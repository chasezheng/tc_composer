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
from typing import Tuple, MutableSequence, Callable, Sequence

import numpy as np
import tensor_comprehensions as tc
import torch
from torch import Tensor

from tc_composer import settings
from tc_composer.func.function_with_params import FunctionWithParams
from .async_queue import AsyncQueue
from .gpu_lock import gpu_lock
from .modules import Vectorizer, Evaluator
from .settings import EVENT_LOOP, get_configured_logger, SAVE_DIR
from .stats import TunerStats


class Worker(multiprocessing.Process):
    __slots__ = '_result_queue', '_progress_queue', 'tuner_config', \
                '_f', '_start_option_attr', '_inps', '_correct_out'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE: float = float('inf')
    SAMPLE_SIZE = 10

    TUNER_CONFIG = tc.TunerConfig().pop_size(2).generations(1).crossover_rate(1).mutation_rate(3)

    def __init__(self,
                 round: int,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor],
                 start_option_attr: Vectorizer.OptionAttr,
                 progress_queue: AsyncQueue,
                 result_queue: AsyncQueue):
        super(Worker, self).__init__()
        self._round = round
        self._start_option_attr = start_option_attr
        self._progress_queue = progress_queue
        self._result_queue = result_queue
        self._f = f
        self._inps = inps
        self._correct_out = correct_outs

    def run(self):
        EVENT_LOOP.stop()
        f = self._f
        inps = self._inps
        option_attrs = self._start_option_attr

        for _ in range(self._round):
            self._progress_queue.put((option_attrs, datetime.now(), self.pid))

            try:
                with gpu_lock:
                    option = f.tune_options(
                        inps,
                        start_option=Vectorizer.from_attr_to_opt(option_attrs),
                        tuner_config=self.TUNER_CONFIG,
                        save_result=False
                    )
                f.recompile(*inps, option=option)
            except RuntimeError:
                self._result_queue.put((option_attrs, self.NOT_VIABLE))
                traceback.print_exc()
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
                except AssertionError:
                    self._result_queue.put((Vectorizer.parse_option_str(str(option)),
                                            self.NOT_VIABLE))
                    traceback.print_exc()
                else:
                    option_attrs = Vectorizer.parse_option_str(str(option))

                    synchronize = torch.cuda.synchronize
                    timer = time.perf_counter

                    with gpu_lock:
                        synchronize()
                        start = timer()
                        for _ in range(self.SAMPLE_SIZE):
                            f(*inps)
                        synchronize()
                        end = timer()

                    self._result_queue.put((option_attrs, start - end))


class Tuner:
    __slots__ = '_progress_queue', '_result_queue', '_workers', '_evaluator', '_f', '_inps', '_correct_outs', \
                '_collected_results', '_collected_progress', '_collect_results_config', \
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
        self._progress_queue: AsyncQueue[Tuple[Vectorizer.OptionAttr, datetime, int]] \
            = AsyncQueue()  # todo when will this queue size be larger than NUM_RUNNER?
        self._result_queue: AsyncQueue[Tuple[Vectorizer.OptionAttr, float]] = AsyncQueue()
        self._collected_progress: MutableSequence[Tuple[Vectorizer.OptionAttr, datetime, int]] = []
        self._collected_results: MutableSequence[Tuple[Vectorizer.OptionAttr, float]] = []

        self._f = f
        self._inps = inps
        self._correct_outs = correct_outs
        self._f.logger.setLevel(logging.WARNING)

        self._evaluator = Evaluator()

        self._main_async_config = [3, 1.001, 3]
        self._collect_results_config = [.2, 1.0005, 3]
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

    def _make_worker(self, round: int, start_option_attr: Vectorizer.OptionAttr = None) -> Worker:
        return Worker(f=self._f, inps=self._inps, correct_outs=self._correct_outs,
                      start_option_attr=start_option_attr or self.start_option_attr,
                      progress_queue=self._progress_queue, round=round,
                      result_queue=self._result_queue, )

    @staticmethod
    def check_exception(f: asyncio.Future):
        if f.cancelled() or (not f.done()):
            return
        e = f.exception()
        if e is not None:
            raise e

    def put_result(self, opt: Vectorizer.OptionAttr, t: float):
        self._collected_results.append((opt, t))

    def pop_progress(self, opt: Vectorizer.OptionAttr) -> Tuple[datetime, int]:
        for n, (o, dt, pid) in enumerate(self._collected_progress):
            if opt == o:
                self._collected_progress.pop(n)
                return dt, pid
        raise Exception

    async def kill_stalled(self):
        collected_progress = self._collected_progress

        now = datetime.now()
        for n, (opt_attr, dt, pid) in tuple(enumerate(collected_progress)):
            if (now - dt).total_seconds() > mean_work_time \
                    + 2 * ((moment_2nd_work_time - mean_work_time ** 2) ** .5):
                worker = self._workers[tuple(w.pid for w in self._workers).index(pid)]
                worker.terminate()
                round = worker_rounds.pop(pid)

                new_worker = self._make_worker(round=round, start_option_attr=opt_attr)
                worker_rounds[new_worker.pid] = round
                self._workers[self._workers.index(worker)] = new_worker

                collected_progress.pop(n)
                self.put_result(opt_attr, self.NOT_VIABLE)
                stats.increment('worker_died')

    async def collect_results(self):
        stats = self._stats
        progress_queue = self._progress_queue
        result_queue = self._result_queue
        collected_progress = self._collected_progress
        delay, backoff, recovery_bias = self._collect_results_config

        mean_work_time = 100
        moment_2nd_work_time = mean_work_time ** 2

        worker_rounds = dict((w.pid, 0) for w in self._workers)

        while True:
            print('collect results')
            progress_qsize = progress_queue.qsize()
            self.logger.info(f'progress_qsize: {progress_qsize}')
            for _ in range(progress_qsize):
                collected_progress.append(await progress_queue.aget())
                print('got progress')
            stats.queue_passage('progress', value=progress_qsize)
            delay /= (backoff ** progress_qsize)

            result_qsize = result_queue.qsize()
            self.logger.info(f'result_qsize: {result_qsize}')
            for _ in range(result_qsize):
                opt_attr, t = await result_queue.aget()
                now = datetime.now()
                start_time, pid = self.pop_progress(opt_attr)
                self.put_result(opt_attr, t)
                worker_rounds[pid] += 1
                mean_work_time = .99 * mean_work_time \
                                 + .01 * (now - start_time).total_seconds()
                moment_2nd_work_time = .99 * moment_2nd_work_time \
                                       + .01 * ((now - start_time).total_seconds() ** 2)
            stats.queue_passage('result', value=result_qsize)
            delay /= (backoff ** result_qsize)

            now = datetime.now()
            for n, (opt_attr, dt, pid) in tuple(enumerate(collected_progress)):
                if (now - dt).total_seconds() > mean_work_time \
                        + 2 * ((moment_2nd_work_time - mean_work_time ** 2) ** .5):
                    worker = self._workers[tuple(w.pid for w in self._workers).index(pid)]
                    worker.terminate()
                    round = worker_rounds.pop(pid)

                    new_worker = self._make_worker(round=round, start_option_attr=opt_attr)
                    worker_rounds[new_worker.pid] = round
                    self._workers[self._workers.index(worker)] = new_worker

                    collected_progress.pop(n)
                    self.put_result(opt_attr, self.NOT_VIABLE)
                    stats.increment('worker_died')

            await asyncio.sleep(delay)
            delay *= backoff ** (1 / recovery_bias)
            self._collect_results_config[0] = delay
            stats.gauge('work_time', mean_work_time, tags=['key:mean'])
            stats.gauge('work_time',
                        mean_work_time - ((moment_2nd_work_time - mean_work_time ** 2) ** .5),
                        tags=['key:-sd'])
            stats.gauge('work_time',
                        mean_work_time + ((moment_2nd_work_time - mean_work_time ** 2) ** .5),
                        tags=['key:+sd'])

    async def save_results(self):
        stats = self._stats
        evaluator = self._evaluator
        delay, backoff, recovery_bias = self._save_result_config

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        while True:
            while len(self._collected_results) > 0:
                delay /= backoff ** recovery_bias

                with open(datetime.now().strftime('%Y-%m-%d %X'), 'w') as f:
                    opt_attr, t = self._collected_results.pop()
                    option = Vectorizer.from_attr_to_opt(opt_attr)
                    stats.log_option(str(option), t)

                    async with gpu_lock:
                        prediteced_t = evaluator(
                            Vectorizer.from_attr_to_tensor(opt_attr)
                        )
                        if t == self.NOT_VIABLE:
                            prediteced_t.neg().backward(retain_graph=True)
                        else:
                            (prediteced_t - t).abs().backward(retain_graph=True)

                    f.write(str(option))

            async with gpu_lock:
                evaluator.apply_grad()
            await sleep(delay)
            delay *= backoff

    async def monitor(self):
        stats = self._stats
        progress = self._progress_queue
        result = self._result_queue

        collect_results_config = self._collect_results_config

        gpu_lock_queue = gpu_lock._LOCK_QUEUE

        while True:
            stats.resource_usage()

            stats.queue_size('progress', progress.qsize())
            stats.queue_size('result', result.qsize())
            stats.gauge('workers_alive', sum(1 for w in self._workers if w.is_alive()))

            stats.gauge('collected', len(self._collected_progress), tags=['key:progress'])
            stats.gauge('collected', len(self._collected_results), tags=['key:result'])

            stats.async_stats('collect_results', *collect_results_config)
            stats.async_stats('gpu_lock',
                              gpu_lock_queue._get_retry_delay,
                              gpu_lock_queue._exponential_backoff,
                              gpu_lock_queue.RECOVERY_BIAS)

            await asyncio.sleep(.2)

    async def main(self, num: int):
        self._stats.start()
        work = asyncio.gather(self.monitor(),
                              self.collect_results(),
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
