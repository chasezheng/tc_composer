import asyncio
import logging
import multiprocessing
import os
import sys
import time
from collections import abc
from functools import lru_cache
from itertools import chain
from multiprocessing import sharedctypes
from typing import Sequence

import numpy as np
import tensor_comprehensions as tc
import torch
from torch import Tensor, autograd
import shelve
from tc_composer import settings
import math
from tc_composer.func.function_with_params import FunctionWithParams
from . import async_util
from .gpu_lock import gpu_lock
from .message_bus import MessageBus, Worker
from .modules import Proposer, Evaluator, Vectorizer
from .option_result import OptionResult
from .settings import EVENT_LOOP, get_configured_logger, SAVE_DIR
from .stats import TunerStats


class ChildTuner(Worker):
    __slots__ = '_f', '_inps', '_correct_outs', '_msg_bus', '_started_at', '_working'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE = float('inf')
    SAMPLE_SIZE = 1000

    def __init__(self,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 msg_bus: MessageBus):
        super(ChildTuner, self).__init__()
        self._f = f
        self._inps = inps
        self._correct_outs = correct_outs
        self._msg_bus = msg_bus
        self._started_at = sharedctypes.RawValue('f', float('nan'))
        self._working = sharedctypes.RawValue('b', False)
        self._current_opt = sharedctypes.RawArray('f', Vectorizer.LEN)

    @property
    def done(self):
        return False

    @property
    def current_opt(self) -> tc.MappingOptions:
        return Vectorizer.to_mapping_options(Tensor(self._current_opt))

    @property
    def started_at(self) -> float:
        return self._started_at.value

    def reincarnate(self):
        return ChildTuner(f=self._f, inps=self._inps,
                          correct_outs=self._correct_outs, msg_bus=self._msg_bus)

    def run(self):
        f = self._f
        inps = self._inps
        f.logger.setLevel(logging.WARNING)

        while True:
            task = asyncio.ensure_future(self._msg_bus.aget_queue())
            EVENT_LOOP.run_until_complete(task)  # todo abstract this
            opt_res: OptionResult = task.result()
            self._current_opt[:Vectorizer.LEN] = opt_res.to_tensor().data.tolist()
            self._working.value = True

            self._started_at.value = int(time.perf_counter())

            opt_res.speed = self.NOT_VIABLE  # Until proven otherwise
            try:
                f.recompile(*inps, option=opt_res.option)
            except RuntimeError:
                opt_res.exc = sys.exc_info()[1]
                opt_res.speed = self.NOT_VIABLE
            except:
                opt_res.exc = sys.exc_info()[1]
                opt_res.speed = self.NOT_VIABLE
                raise
            else:
                try:
                    # Check correctness
                    with gpu_lock:
                        out = f(*inps)
                        if isinstance(out, abc.Sequence):
                            for t, t0 in zip(out, self._correct_outs):
                                np.testing.assert_allclose(t.data, t0.data, rtol=self.RTOL)
                        else:
                            np.testing.assert_allclose(out.data, self._correct_outs[0].data,
                                                       rtol=self.RTOL)
                except AssertionError:
                    opt_res.exc = sys.exc_info()[1]
                    opt_res.speed = self.NOT_VIABLE
                else:
                    # Benchmark
                    with gpu_lock:
                        synchronize = torch.cuda.synchronize
                        timer = time.perf_counter

                        synchronize()
                        start = timer()
                        for _ in range(self.SAMPLE_SIZE):
                            f(*inps)
                        synchronize()
                        end = timer()

                    opt_res.speed = end - start
            finally:
                opt_res.compile_time = int(time.perf_counter()) - self.started_at
                self._msg_bus.add_result(opt_res)
                self._working.value = False

    def terminate(self):
        if self._working.value:
            opt_res = OptionResult(self.current_opt)
            opt_res.speed = self.NOT_VIABLE
            self._msg_bus.add_result(opt_res)
        super(ChildTuner, self).terminate()


class Tuner:
    __slots__ = '_proposer', '_evaluator', '_f', '_inps', '_correct_outs', '_in_sizes', \
                '_stats', '_make_options_async_config', '_save_result_config', '_msg_bus', \
                '_prediction_error', '_opt_res_persist'

    MAX_COMPILE_TIME = 180
    NUM_RUNNER = multiprocessing.cpu_count()

    NOT_VIABLE = ChildTuner.NOT_VIABLE

    def __init__(self, f: FunctionWithParams,
                 inps: Sequence[Tensor], correct_outs: Sequence[Tensor],
                 start_option: tc.MappingOptions = None,
                 prefix: str = None):
        super(Tuner, self).__init__()
        name = (prefix or f.entry_point) + '_tuner'
        self._msg_bus = MessageBus(name=name)

        in_sizes = tuple(chain(*(inp.shape for inp in inps)))
        self._in_sizes = Tensor(in_sizes)
        self._proposer = Proposer(len(in_sizes), start_option=start_option)
        self._evaluator = Evaluator()

        self._f = f
        f.logger.setLevel(logging.WARNING)
        self._inps = inps
        self._correct_outs = correct_outs

        self._make_options_async_config = [3, 1.001, 3]
        self._save_result_config = [30, 1.001, 1 / 8]

        self._stats = TunerStats(name=name)

        self._prediction_error = 10

        os.makedirs(self.save_dir, exist_ok=True)
        self._opt_res_persist = shelve.open(os.path.join(self.save_dir, name), writeback=True)

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return get_configured_logger(
            self._f.entry_point + '_tuner',
            format='[%(levelname)s] %(name)s.%(funcName)s L.%(lineno)d - %(message)s')

    @property
    def save_dir(self):
        return os.path.join(SAVE_DIR, self._stats.name)

    async def _apply_grad(self):
        async with gpu_lock:
            proposer_grad = sum(
                p.grad.abs().sum().item() for p in self._proposer.parameters()
                if p.grad is not None
            )
            evaluator_grad = sum(
                p.grad.abs().sum().item() for p in self._evaluator.parameters()
                if p.grad is not None
            )
            self._proposer.apply_grad()
            self._evaluator.apply_grad()
        self._stats.gauge('gradient', proposer_grad, tags=[f'key:proposer'])
        self._stats.gauge('gradient', evaluator_grad, tags=[f'key:evaluator'])

    def _make_worker(self) -> ChildTuner:
        return ChildTuner(msg_bus=self._msg_bus, f=self._f, inps=self._inps,
                          correct_outs=self._correct_outs)

    async def monitor(self):
        stats = self._stats

        while True:
            stats.resource_usage()
            self._stats.gauge('prediction_error', self._prediction_error)

            await asyncio.sleep(.02)

    async def evaluate_option(self, t: Tensor, opt: tc.MappingOptions) -> None:
        assert len(t.shape) == 1 and t.shape[0] == Vectorizer.LEN, \
            f"t.shape = {t.shape}; Vectorizer.LEN = {Vectorizer.LEN}"
        self._stats.increment('progress', tags=['key:starting'])

        opt_res = OptionResult(opt)

        if str(opt) not in self._opt_res_persist:
            self._msg_bus.queue(opt_res)
            opt_res = await self._msg_bus.find_and_pop(opt_res)
            self._opt_res_persist[str(opt)] = opt_res.speed
        else:
            opt_res.speed = self._opt_res_persist[str(opt)]
            self._stats.queue_passage(f'Retrieved option')

        if opt_res.speed is not self.NOT_VIABLE:
            self._stats.log_option(opt_res)

        actual_t = opt_res.speed

        async with gpu_lock:
            predicted_t: Tensor = self._evaluator(t)
            try:
                for p in self._proposer.parameters():
                    # Freezing params
                    p.requires_grad = False
                if actual_t == self.NOT_VIABLE:
                    predicted_t.neg().abs().backward(retain_graph=True)
                else:
                    diff = (math.log(actual_t) - predicted_t).abs()
                    diff.backward(retain_graph=True)
                    self._prediction_error = .95 * self._prediction_error + .05 * diff.item()
            finally:
                for p in self._proposer.parameters():
                    p.requires_grad = True
            try:
                for p in self._evaluator.parameters():
                    # Freezing params
                    p.requires_grad = False

                if actual_t == self.NOT_VIABLE:
                    predicted_t.backward(retain_graph=True)
                else:
                    with autograd.set_grad_enabled(False):
                        r = actual_t / predicted_t
                    (predicted_t * r).backward(retain_graph=True)
            finally:
                for p in self._evaluator.parameters():
                    p.requires_grad = True

    @async_util.repeat(recovery_bias=100)
    async def make_options(self):
        diff = self._msg_bus.queue_msg_num() - 2 * self.NUM_RUNNER
        if diff > 0:
            return async_util.Repeat.DelayFactor(diff)

        async with gpu_lock:
            ts, opts = self._proposer(self._in_sizes)
            with torch.autograd.set_grad_enabled(False):
                ts_with_noise = ts + torch.normal(0, std=ts / 2)
                opts_with_noise = (Vectorizer.to_mapping_options(t) for t in ts_with_noise)

        return (*zip(ts, opts), *zip(ts_with_noise, opts_with_noise))

    async def main(self, num: int):
        workers = tuple(self._make_worker() for _ in range(self.NUM_RUNNER))
        for w in workers:
            w.start()
        house_keeping = asyncio.gather(self.monitor(),
                                       self._msg_bus.monitor(),
                                       async_util.Repeat.monitor(),
                                       self._msg_bus.kill_zombie(*workers),
                                       loop=EVENT_LOOP)
        house_keeping = asyncio.ensure_future(house_keeping, loop=EVENT_LOOP)
        del workers

        for _ in range(num):
            out = await self.make_options()
            async_util.check_exception(house_keeping)
            await asyncio.gather(*(self.evaluate_option(t, o) for t, o in out))
            await self._apply_grad()
            self._stats.queue_passage('tasks')
            self._opt_res_persist.sync()

    def run(self, num: int):
        main = asyncio.Task(self.main(num), loop=EVENT_LOOP)

        try:
            EVENT_LOOP.run_until_complete(main)
        finally:
            self._stats.flush()
            self._opt_res_persist.sync()

            async_util.check_exception(main)
