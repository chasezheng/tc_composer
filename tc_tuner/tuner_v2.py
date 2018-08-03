import asyncio
import logging
import multiprocessing
import os
import time
import traceback
from asyncio import sleep
from collections import abc
from datetime import datetime
from functools import lru_cache
from typing import Sequence

import numpy as np
import tensor_comprehensions as tc
import torch
from torch import Tensor

from tc_composer import settings
from tc_composer.func.function_with_params import FunctionWithParams
from .gpu_lock import gpu_lock
from .message_bus import MessageBus, Worker
from .modules import Vectorizer, Evaluator
from .option_result import OptionResult
from .settings import EVENT_LOOP, get_configured_logger, SAVE_DIR
from .stats import TunerStats


class ChildTuner(Worker):
    __slots__ = 'tuner_config', '_f', '_start_option_res', '_inps', '_correct_out', 'progress', 'msg_bus'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE: float = float('inf')
    SAMPLE_SIZE = 10

    TUNER_CONFIG = tc.TunerConfig().pop_size(2).generations(1).crossover_rate(1).mutation_rate(3)

    def __init__(self,
                 round: int,
                 msg_bus: MessageBus,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 start_option_res: OptionResult):
        super(Worker, self).__init__()
        self._start_option_res = start_option_res
        self._f = f
        self._inps = inps
        self._correct_out = correct_outs

        self.msg_bus = msg_bus
        self.progress = self.msg_bus._manager.Namespace()
        self.progress.remaining_round = round
        self.progress.option_res = start_option_res
        self.progress.done = False

    @property
    def done(self):
        return self.progress.done

    def run(self):
        f = self._f
        inps = self._inps
        start_option_res = self._start_option_res

        for _ in range(self.progress.remaining_round):
            self.msg_bus.started(start_option_res, self.pid)

            with gpu_lock:
                option = f.tune_options(
                    inps,
                    start_option=start_option_res.option,
                    tuner_config=self.TUNER_CONFIG,
                    save_result=False
                )
                option_res = OptionResult(option)
            try:
                f.recompile(*inps, option=option)
            except RuntimeError as e:
                option_res.exc = e
                option_res.speed = self.NOT_VIABLE
                self.msg_bus.add_result(option_res)
                traceback.print_exc()
                del e
            else:
                # Check correctness
                try:
                    with gpu_lock:
                        out = f(*inps)
                        if isinstance(out, abc.Sequence):
                            for t, t0 in zip(out, self._correct_out):
                                np.testing.assert_allclose(t.data, t0.data, rtol=self.RTOL)
                        else:
                            np.testing.assert_allclose(out.data, self._correct_out[0].data,
                                                       rtol=self.RTOL)
                except AssertionError as e:
                    option_res.exc = e
                    option_res.speed = self.NOT_VIABLE
                    self.msg_bus.add_result(option_res)
                    traceback.print_exc()
                    del e
                else:
                    synchronize = torch.cuda.synchronize
                    timer = time.perf_counter

                    with gpu_lock:
                        synchronize()
                        start = timer()
                        for _ in range(self.SAMPLE_SIZE):
                            f(*inps)
                        synchronize()
                        end = timer()

                    option_res.speed = end - start
                    self.msg_bus.add_result(option_res)
                    start_option_res = option_res

                    self.progress.remaining_round -= 1
                    self.progress.option_res = start_option_res

        self.progress.done = True

    def reincarnate(self):
        return Worker(
            round=self.progress.remaining_round,
            msg_bus=self.msg_bus,
            f=self._f,
            inps=self._inps,
            correct_outs=self._correct_out,
            start_option_res=self.progress.option_res, )


class Tuner:
    __slots__ = '_workers', '_evaluator', '_f', '_inps', '_correct_outs', \
                '_collect_results_config', 'msg_bus', \
                '_save_result_config', '_stats', '_main_async_config', 'start_option_attr'

    NUM_RUNNER = multiprocessing.cpu_count()
    NOT_VIABLE: float = Worker.NOT_VIABLE

    def __init__(self,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 start_option: tc.MappingOptions,
                 name: str = None):
        super(Tuner, self).__init__()
        self.start_option_attr = Vectorizer.parse_option_str(str(start_option))

        self._f = f
        self._inps = inps
        self._correct_outs = correct_outs
        self._f.logger.setLevel(logging.WARNING)
        self.msg_bus = MessageBus()
        self._evaluator = Evaluator()

        self._main_async_config = [3, 1.001, 3]
        self._save_result_config = [30, 1.001, 1 / 8]

        self._stats = TunerStats(name=(name or self._f.entry_point))
        self._workers = []

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return get_configured_logger(
            self._f.entry_point + '_tuner',
            format='[%(levelname)s] %(name)s.%(funcName)s - %(message)s')

    @property
    def save_dir(self):
        return os.path.join(SAVE_DIR, self._stats.name)

    def _make_worker(self, round: int, start_option_attr: Vectorizer.OptionAttr = None) -> ChildTuner:
        return ChildTuner(f=self._f, inps=self._inps, correct_outs=self._correct_outs,
                          start_option_res=start_option_attr or self.start_option_attr,
                          round=round, msg_bus=self.msg_bus)

    @staticmethod
    def check_exception(f: asyncio.Future):
        if f.cancelled() or (not f.done()):
            return
        e = f.exception()
        if e is not None:
            raise e

    async def save_results(self):
        stats = self._stats
        evaluator = self._evaluator
        delay, backoff, recovery_bias = self._save_result_config

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        while True:
            while self.msg_bus.result_len() > 0:
                delay /= backoff ** recovery_bias

                with open(datetime.now().strftime('%Y-%m-%d %X'), 'w') as f:
                    opt_res = self.msg_bus.pop_result()
                    stats.log_option(opt_res)

                    async with gpu_lock:
                        prediteced_t = evaluator(opt_res.option)
                        if opt_res.speed == self.NOT_VIABLE:
                            prediteced_t.neg().backward(retain_graph=True)
                        else:
                            (prediteced_t - opt_res.speed).abs().backward(retain_graph=True)

                    f.write(str(opt_res.option))

            async with gpu_lock:
                evaluator.apply_grad()
            await sleep(delay)
            delay *= backoff

    async def monitor(self):
        stats = self._stats

        gpu_lock_queue = gpu_lock._LOCK_QUEUE

        while True:
            stats.resource_usage()
            stats.gauge('workers_alive', sum(1 for w in self._workers if w.is_alive()))
            stats.async_stats('gpu_lock',
                              gpu_lock_queue._get_retry_delay,
                              gpu_lock_queue._exponential_backoff,
                              gpu_lock_queue.RECOVERY_BIAS)

            await asyncio.sleep(.2)

    async def main(self, num: int):
        self._stats.start()
        work = asyncio.gather(self.monitor(),
                              self.save_results(),
                              loop=EVENT_LOOP)
        asyncio.ensure_future(work, loop=EVENT_LOOP)

        try:
            self._workers.extend(self._make_worker(num) for _ in range(self.NUM_RUNNER))
            tuple(w.start() for w in self._workers)

            while True:
                if any(w.is_alive() for w in self._workers):
                    self.check_exception(work)
                    await asyncio.sleep(1)
                else:
                    self._workers.clear()
                    break
        finally:
            self._stats.flush()
            self._stats.stop()

            self.check_exception(work)

    def run(self, num: int):
        main = asyncio.Task(self.main(num), loop=EVENT_LOOP)
        try:
            EVENT_LOOP.run_until_complete(main)
        finally:
            self.check_exception(main)
