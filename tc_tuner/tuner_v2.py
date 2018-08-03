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
from multiprocessing import sharedctypes
from typing import Sequence, MutableSequence

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
    __slots__ = 'tuner_config', '_f', '_inps', '_correct_out', '_start_option_res', '_msg_bus', '_done', '_started_at'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE: float = float('inf')
    SAMPLE_SIZE = 10

    TUNER_CONFIG = tc.TunerConfig().pop_size(2).generations(1).crossover_rate(1).mutation_rate(3)

    def __init__(self,
                 num: int,
                 msg_bus: MessageBus,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 start_option_res: OptionResult):
        super(Worker, self).__init__()
        self._f = f
        self._inps = inps
        self._correct_out = correct_outs
        self._msg_bus = msg_bus

        self._done = multiprocessing.sharedctypes.RawValue('b', False)
        self._num = multiprocessing.sharedctypes.RawValue('i', num)
        self.__start_option_res: MutableSequence[float] \
            = multiprocessing.sharedctypes.RawArray('f', Vectorizer.LEN)

        if isinstance(start_option_res, tc.MappingOptions):
            # todo logging warning
            start_option_res = OptionResult(start_option_res)
        self._start_option_res = start_option_res
        self._started_at = sharedctypes.RawValue('f', float('nan'))

    @property
    def done(self):
        return self._done.value

    @property
    def started_at(self):
        return self._started_at.value

    def run(self):
        f = self._f
        inps = self._inps
        start_option_res = self._start_option_res

        for _ in range(self._num.value):
            self._started_at.value = int(time.time())

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
                self._msg_bus.add_result(option_res)
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
                    self._msg_bus.add_result(option_res)
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
                    self._msg_bus.add_result(option_res)
                    start_option_res = option_res

                    self._num.value -= 1
                    self._start_option_res = start_option_res

        self._done = True

    def reincarnate(self):
        return ChildTuner(
            num=self._num,
            msg_bus=self._msg_bus,
            f=self._f,
            inps=self._inps,
            correct_outs=self._correct_out,
            start_option_res=self._start_option_res)


class Tuner:
    __slots__ = '_workers', '_evaluator', '_f', '_inps', '_correct_outs', \
                '_collect_results_config', '_msg_bus', \
                '_save_result_config', '_stats', '_main_async_config', 'start_option_res'

    NUM_RUNNER = multiprocessing.cpu_count()
    NOT_VIABLE: float = ChildTuner.NOT_VIABLE

    def __init__(self,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 start_option: tc.MappingOptions,
                 name: str = None):
        name = (name or f.entry_point) + '_tuner'

        self.start_option_res = OptionResult(start_option)

        self._f = f
        self._inps = inps
        self._correct_outs = correct_outs
        self._f.logger.setLevel(logging.WARNING)
        self._msg_bus = MessageBus(name=name)
        self._evaluator = Evaluator()

        self._main_async_config = [3, 1.001, 3]
        self._save_result_config = [30, 1.001, 1 / 8]

        self._stats = TunerStats(name=name)
        self._workers: MutableSequence[ChildTuner] = []

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return get_configured_logger(
            self._f.entry_point + '_tuner',
            format='[%(levelname)s] %(name)s.%(funcName)s - %(message)s')

    @property
    def save_dir(self):
        return os.path.join(SAVE_DIR, self._stats.name)

    def _make_worker(self, num: int, start_option_res: OptionResult = None) -> ChildTuner:
        return ChildTuner(f=self._f, inps=self._inps, correct_outs=self._correct_outs,
                          start_option_res=start_option_res or self.start_option_res,
                          num=num, msg_bus=self._msg_bus)

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
            while self._msg_bus.result_len() > 0:
                delay /= backoff ** recovery_bias

                with open(datetime.now().strftime('%Y-%m-%d %X'), 'w') as f:
                    opt_res = self._msg_bus.pop_result()
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
            self._save_result_config[0] = delay

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
            stats.async_stats('save_results', *self._save_result_config)

            await asyncio.sleep(.2)

    async def main(self, num: int):
        work = asyncio.gather(self.monitor(),
                              self._msg_bus.monitor(),
                              self._msg_bus.kill_zombie(),
                              self.save_results(),
                              loop=EVENT_LOOP)
        asyncio.ensure_future(work, loop=EVENT_LOOP)

        try:
            self._workers.extend(
                self._make_worker(num, start_option_res=self.start_option_res) for _ in range(self.NUM_RUNNER)
            )
            self.logger.info('Starting workers...')
            tuple(w.start() for w in self._workers)
            self.logger.info('Workers started')

            while True:
                if any(w.is_alive() for w in self._workers):
                    self.check_exception(work)
                    await asyncio.sleep(3)
                else:
                    self._workers.clear()
                    break
        finally:
            self._stats.flush()

            self.check_exception(work)

    def run(self, num: int):
        main = asyncio.Task(self.main(num), loop=EVENT_LOOP)
        try:
            EVENT_LOOP.run_until_complete(main)
        finally:
            self.check_exception(main)
