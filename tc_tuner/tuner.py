import asyncio
import logging
import time
import traceback
from asyncio import sleep
from collections import abc
from datetime import datetime
from functools import lru_cache
from itertools import chain
from typing import Sequence, Tuple, MutableSequence

import numpy as np
import tensor_comprehensions as tc
import torch
from torch import Tensor, autograd, multiprocessing

from tc_composer import settings
from tc_composer.func.function_with_params import FunctionWithParams
from .async_queue import AsyncQueue
from .gpu_lock import gpu_lock
from .modules import Proposer, Evaluator, Vectorizer
from .settings import EVENT_LOOP, get_configured_logger
from .stats import TunerStats


class Worker(multiprocessing.Process):
    __slots__ = '_result_queue', '_progress_queue', '_f', '_option_queue', '_inps', '_correct_out'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE = float('inf')
    SAMPLE_SIZE = 10

    def __init__(self,
                 option_queue: AsyncQueue[Vectorizer.OptionAttr],
                 progress_queue: AsyncQueue[Tuple[Vectorizer.OptionAttr, datetime, int]],
                 result_queue: AsyncQueue[Vectorizer.OptionAttr, float],
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor]):
        super(Worker, self).__init__()
        self._option_queue = option_queue
        self._progress_queue = progress_queue
        self._result_queue = result_queue
        self._f = f
        self._inps = inps
        self._correct_out = correct_outs

    def run(self):
        f = self._f
        inps = self._inps
        try:
            while True:
                option_attrs = self._option_queue.get()
                self._progress_queue.put((option_attrs, datetime.now(), self.pid))

                try:
                    f.recompile(*inps, option=Vectorizer.from_attr_to_opt(option_attrs))
                except RuntimeError:
                    self._result_queue.put((option_attrs, self.NOT_VIABLE))
                else:
                    # Check correctness
                    with gpu_lock:
                        try:
                            out = f(*inps)
                            if isinstance(out, abc.Sequence):
                                for t, t0 in zip(out, self._correct_out):
                                    np.testing.assert_allclose(t.data, t0.data, rtol=self.RTOL)
                            else:
                                np.testing.assert_allclose(out.data, self._correct_out[0].data,
                                                           rtol=self.RTOL)
                        except AssertionError:
                            self._result_queue.put((option_attrs, self.NOT_VIABLE))

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

                    self._result_queue.put((option_attrs, start - end))
        finally:
            print('Exiting')


class Tuner:
    __slots__ = '_option_queue', '_progress_queue', '_result_queue', '_workers', '_proposer', \
                '_evaluator', '_f', '_inps', '_correct_outs', '_in_sizes', '_collected_results', \
                '_collected_progress', '_collect_results_config', '_get_result_config', \
                '_stats', '_make_options_async_config'

    MAX_COMPILE_TIME = 180
    NUM_RUNNER = multiprocessing.cpu_count()

    NOT_VIABLE = Worker.NOT_VIABLE

    def __init__(self, f: FunctionWithParams,
                 inps: Sequence[Tensor], correct_outs: Sequence[Tensor],
                 start_option: tc.MappingOptions = None,
                 prefix: str = None):
        super(Tuner, self).__init__()
        self._option_queue = AsyncQueue()
        self._progress_queue = AsyncQueue()  # todo when will this queue size be larger than NUM_RUNNER?
        self._result_queue = AsyncQueue()
        self._collected_progress: MutableSequence[Tuple[Vectorizer.OptionAttr, datetime, int]] = []
        self._collected_results: MutableSequence[Tuple[Vectorizer.OptionAttr, float]] = []

        in_sizes = tuple(chain(*(inp.shape for inp in inps)))
        self._in_sizes = Tensor(in_sizes)
        self._proposer = Proposer(len(in_sizes), start_option=start_option)
        self._evaluator = Evaluator()

        self._f = f
        f.logger.setLevel(logging.WARNING)
        self._inps = inps
        self._correct_outs = correct_outs

        self._make_options_async_config = [3, 1.001, 3]
        self._collect_results_config = [.2, 1.0005, 3]
        self._get_result_config = [.2, 1.0005, 3]

        self._workers = [self._make_worker() for _ in range(self.NUM_RUNNER)]
        for w in self._workers:
            w.start()

        self._stats = TunerStats(name=(prefix or f.entry_point) + '_tuner')

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return get_configured_logger(
            self._f.entry_point + '_tuner',
            format='[%(levelname)s] %(name)s.%(funcName)s L.%(lineno)d - %(message)s')

    def _make_worker(self) -> Worker:
        return Worker(option_queue=self._option_queue, progress_queue=self._progress_queue,
                      result_queue=self._result_queue, f=self._f, inps=self._inps,
                      correct_outs=self._correct_outs)

    @staticmethod
    def check_exception(f: asyncio.Future):
        if f.cancelled() or (not f.done()):
            return
        e = f.exception()
        if e is not None:
            raise e

    def put_result(self, opt, t):
        self._collected_results.append((opt, t))

    def pop_progress(self, opt) -> Tuple[datetime, int]:
        for n, (o, dt, pid) in enumerate(self._collected_progress):
            if opt == o:
                self._collected_progress.pop(n)
                return dt, pid
        raise Exception

    async def get_result(self, opt) -> float:
        delay, backoff, recovery_bias = self._get_result_config
        while True:
            for o, t in self._collected_results:
                if o == opt:
                    self._get_result_config[0] = delay / (backoff ** recovery_bias)
                    return t
            await sleep(delay)
            delay *= backoff

    async def collect_results(self):
        stats = self._stats
        progress_queue = self._progress_queue
        result_queue = self._result_queue
        collected_progress = self._collected_progress
        delay, backoff, recovery_bias = self._collect_results_config

        mean_compile_time = 300
        moment_2nd_compile_time = mean_compile_time ** 2

        while True:
            progress_qsize = progress_queue.qsize()
            self.logger.info(f'progress_qsize: {progress_qsize}')
            for _ in range(progress_qsize):
                collected_progress.append(await progress_queue.aget())
            stats.queue_passage('progress', value=progress_qsize)
            delay /= (backoff ** progress_qsize)

            result_qsize = result_queue.qsize()
            self.logger.info(f'result_qsize: {result_qsize}')
            for _ in range(result_qsize):
                opt, t = await result_queue.aget()
                now = datetime.now()
                mean_compile_time = .99 * mean_compile_time + .01 * (
                        now - self.pop_progress(opt)[0]).total_seconds()
                moment_2nd_compile_time = .99 * moment_2nd_compile_time + .01 * (
                        (now - self.pop_progress(opt)[0]).total_seconds() ** 2)
                self.put_result(opt, t)
            stats.queue_passage('result', value=result_qsize)
            delay /= (backoff ** result_qsize)

            now = datetime.now()
            for n, (opt, dt, pid) in tuple(enumerate(collected_progress)):
                if (now - dt).total_seconds() > mean_compile_time + 2 * (
                        (moment_2nd_compile_time - mean_compile_time ** 2) ** .5):
                    worker = self._workers[tuple(w.pid for w in self._workers).index(pid)]
                    worker.terminate()
                    self._workers[self._workers.index(worker)] = self._make_worker()
                    collected_progress.pop(n)
                    self.put_result(opt, self.NOT_VIABLE)
                    stats.increment('worker_died')

            await asyncio.sleep(delay)
            delay *= backoff ** (1 / recovery_bias)
            self._collect_results_config[0] = delay
            stats.async_stats('progress', delay, backoff, recovery_bias)
            stats.gauge('compile_time', mean_compile_time, tags=['key:mean'])
            stats.gauge('compile_time',
                        mean_compile_time - ((moment_2nd_compile_time - mean_compile_time ** 2) ** .5),
                        tags=['key:-sd'])
            stats.gauge('compile_time',
                        mean_compile_time + ((moment_2nd_compile_time - mean_compile_time ** 2) ** .5),
                        tags=['key:+sd'])

    async def monitor(self):
        stats = self._stats
        option = self._option_queue
        progress = self._progress_queue
        result = self._result_queue

        collect_results_config = self._collect_results_config

        gpu_lock_queue = gpu_lock._LOCK_QUEUE

        while True:
            stats.resource_usage()

            stats.queue_size('option', option.qsize())
            stats.queue_size('progress', progress.qsize())
            stats.queue_size('result', result.qsize())
            stats.gauge('workers_alive', sum(1 for w in self._workers if w.is_alive()))

            stats.gauge('collected', len(self._collected_progress), tags=['key:progress'])
            stats.gauge('collected', len(self._collected_results), tags=['key:result'])

            stats.async_stats('evaluate_option0', *collect_results_config)
            stats.async_stats('gpu_lock',
                              gpu_lock_queue.get_delay,
                              gpu_lock_queue.exponential_backoff,
                              gpu_lock_queue.RECOVERY_BIAS)

            await asyncio.sleep(.2)

    async def evaluate_option(self, t: Tensor) -> None:
        assert len(t.shape) == 1 and t.shape[0] == Vectorizer.LEN, \
            f"t.shape = {t.shape}; Vectorizer.LEN = {Vectorizer.LEN}"
        self._stats.increment('progress', tags=['key:starting'])
        try:
            option_attr = Vectorizer.parse_option_str(str(Vectorizer.to_mapping_options(t)))
        except RuntimeError:
            traceback.print_exc()
            return
        else:
            self._option_queue.put(option_attr)

        actual_t = await self.get_result(option_attr)
        self._stats.increment('progress', tags=['key:result_collected'])

        async with gpu_lock:
            predicted_t: Tensor = self._evaluator(t)
            try:
                for p in self._proposer.parameters():
                    # Freezing params
                    p.requires_grad = False
                if actual_t == self.NOT_VIABLE:
                    predicted_t.neg().abs().backward(retain_graph=True)
                else:
                    (actual_t - predicted_t).mul(100).abs().backward(retain_graph=True)
            finally:
                for p in self._proposer.parameters():
                    p.requires_grad = True
            try:
                for p in self._evaluator.parameters():
                    # Freezing params
                    p.requires_grad = False

                if actual_t == self.NOT_VIABLE:
                    with autograd.set_grad_enabled(False):
                        r = actual_t / predicted_t
                    (predicted_t * r).backward(retain_graph=True)
                else:
                    predicted_t.backward(retain_graph=True)
            finally:
                for p in self._evaluator.parameters():
                    p.requires_grad = True

        self._stats.increment('progress', tags=['key:option_evaluated'])

    async def make_options(self, num: int):
        monitor = asyncio.gather(self.monitor(),
                                 self.collect_results(),
                                 loop=EVENT_LOOP)
        asyncio.ensure_future(monitor, loop=EVENT_LOOP)

        stats = self._stats
        delay, backoff, recovery_bias = self._make_options_async_config
        tasks = []

        for _ in range(num):
            async with gpu_lock:
                options = self._proposer(self._in_sizes).view(-1, Vectorizer.LEN)
                with torch.autograd.set_grad_enabled(False):
                    options_with_noise = options + torch.normal(0, std=options / 10)

            task = asyncio.gather(*(self.evaluate_option(o) for o in (*options, *options_with_noise)),
                                  loop=EVENT_LOOP)
            tasks.append(task)

            def done_callback(*x):
                if tuple(self._evaluator.parameters())[-1].grad.abs().sum().item() == 0:
                    raise Exception
                if tuple(self._proposer.parameters())[-1].grad.abs().sum().item() == 0:
                    raise Exception
                if not tuple(self._proposer.parameters())[0].requires_grad:
                    raise Exception
                if not tuple(self._evaluator.parameters())[0].requires_grad:
                    raise Exception
                self._proposer.apply_grad()
                self._evaluator.apply_grad()

            task.add_done_callback(done_callback)
            asyncio.ensure_future(task, loop=EVENT_LOOP)

            while self._option_queue.qsize() > len(options):
                await asyncio.sleep(delay)
                delay *= backoff
            delay /= backoff ** recovery_bias
            self._make_options_async_config[0] = delay
            stats.async_stats('main', delay, backoff, recovery_bias)

            self.check_exception(monitor)
            if tasks[0].cancelled():
                del tasks[0]
            elif tasks[0].done():
                self.check_exception(tasks.pop(0))

            self.logger.info(f'round {_} done')
            stats.flush()
            await asyncio.sleep(.1)

        while len(tasks) > 0:
            await asyncio.sleep(1)
            if tasks[0].cancelled():
                del tasks[0]
            elif tasks[0].done():
                self.check_exception(tasks.pop(0))
        self.check_exception(monitor)

    def run(self, num: int):
        self._stats.start()
        main = asyncio.Task(self.make_options(num), loop=EVENT_LOOP)

        try:
            EVENT_LOOP.run_until_complete(main)
        finally:
            self._stats.flush()
            self._stats.stop()

            self.check_exception(main)
