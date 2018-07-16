import asyncio
import multiprocessing
import time
from collections import abc
from datetime import datetime
from itertools import chain
from multiprocessing import Process, Queue, queues
from typing import Sequence, MutableMapping

import numpy as np
import tensor_comprehensions as tc
import torch
from torch import Tensor, autograd

from tc_composer import settings
from tc_composer.func.function_with_params import FunctionWithParams
from .evaluator import Evaluator
from .option_vectorizer import Vectorizer
from .proposer import Proposer


class Runner(Process):
    __slots__ = '_gpu_lock', '_result_queue', '_progress_queue', '_f', '_option_queue', '_inps', '_correct_out'

    RTOL = 1e-3 if settings.DEFAULT_TYPE.lower() == 'float' else 1e-9
    NOT_VIABLE = 300
    SAMPLE_SIZE = 10

    def __init__(self,
                 gpu_lock: Queue,
                 option_queue: Queue,
                 progress_queue: Queue,
                 result_queue: Queue,
                 f: FunctionWithParams,
                 inps: Sequence[Tensor],
                 correct_outs: Sequence[Tensor]):
        super(Runner, self).__init__()
        self._gpu_lock = gpu_lock
        self._option_queue = option_queue
        self._progress_queue = progress_queue
        self._result_queue = result_queue
        self._f = f
        self._inps = inps
        self._correct_out = correct_outs

    def run(self):
        f = self._f
        inps = self._inps
        while True:
            option = self._option_queue.get()
            self._progress_queue.put((self, option, datetime.now()))
            f.recompile(*inps, option=option)

            # Check correctness
            try:
                gpu_lock = self._gpu_lock.get()
                out = f(inps)
                if isinstance(out, abc.Sequence):
                    for t, t0 in zip(out, self._correct_out):
                        np.testing.assert_allclose(t.data, t0.data)
                else:
                    np.testing.assert_allclose(out.data, self._correct_out[0].data, rtol=self.RTOL)
            except AssertionError:
                return self.NOT_VIABLE
            finally:
                self._gpu_lock.put(gpu_lock)

            try:
                gpu_lock = self._gpu_lock.get()
                synchronize = torch.cuda.synchronize
                timer = time.perf_counter
                times = []
                out = ()
                for _ in range(self.SAMPLE_SIZE):
                    if not isinstance(out, abc.Sequence):
                        out = (out,)
                    synchronize()
                    t0 = timer()
                    out = f(*inps, out=out)
                    synchronize()
                    times.append(timer() - t0)
            finally:
                self._gpu_lock.put(gpu_lock)

            self._result_queue.put((option, sum(times) / len(times)))


class Tuner:
    __slots__ = '_gpu_lock', '_option_queue', '_progress_queue', '_result_queue', '_workers', '_proposer', \
                '_evaluator', '_f', '_inps', '_correct_outs', 'in_sizes', '_collected_results', '_collected_progress', '_event_loop'

    NUM_RUNNER = multiprocessing.cpu_count()

    def __init__(self, f: FunctionWithParams, inps: Sequence[Tensor], correct_outs: Sequence[Tensor]):
        super(Tuner, self).__init__()
        self._gpu_lock = Queue(maxsize=1)
        self._gpu_lock.put('lock')
        self._option_queue = Queue()
        self._progress_queue = Queue(maxsize=self.NUM_RUNNER)
        self._result_queue = Queue()
        self._collected_progress: MutableMapping[tc.MappingOptions, tuple[datetime, Runner]] = {}
        self._collected_results: MutableMapping[tc.MappingOptions, float] = {}
        self._event_loop = asyncio.get_event_loop()

        self._workers = tuple(
            Runner(gpu_lock=self._gpu_lock, option_queue=self._option_queue, progress_queue=self._progress_queue,
                   result_queue=self._result_queue, f=f, inps=inps, correct_outs=correct_outs) for _ in self.NUM_RUNNER)

        for w in self._workers:
            w.start()

        in_sizes = tuple(chain(*(inp.shape for inp in inps)))
        self.in_sizes = Tensor(in_sizes)
        self._proposer = Proposer(len(in_sizes))
        self._evaluator = Evaluator()

        self._f = f
        self._inps = inps
        self._correct_outs = correct_outs

    async def collect_from_queues(self) -> None:
        while True:
            while not self._progress_queue.empty():
                runner, opt, t = await self._get(self._progress_queue)
                self._progress_queue[opt] = (t, runner)
            while not self._result_queue.empty():
                opt, t = await self._get(self._result_queue)
                self._collected_results[opt] = t
            asyncio.sleep(.2)

    async def evaluate_option(self, t: Tensor) -> None:
        assert len(t.shape) == 1 and t.shape[0] == Vectorizer.LEN
        wait_time = .3

        option = Vectorizer.to_mapping_options(t)
        self._option_queue.put(option)

        while option not in self._progress_queue:
            asyncio.sleep(wait_time)
        start_time, runner = self._collected_progress.pop(option)

        try:
            lock = await self._get(self._gpu_lock)
            predicted_t: Tensor = self._evaluator(option)
        finally:
            self._gpu_lock.put(lock)

        while option not in self._collected_results:
            if (datetime.now() - start_time).seconds > 300:
                runner.terminate()
                runner.start()
                self._collected_results[option] = runner.NOT_VIABLE
                break
            else:
                asyncio.sleep(wait_time)

        actual_t = self._collected_results.pop(option)

        with autograd.set_grad_enabled(False):
            r = actual_t / predicted_t

        (predicted_t * r).backward()

    async def _get(self, queue: Queue, delay: float = .05):
        while True:
            try:
                return queue.get_nowait()
            except multiprocessing.queues.Empty:
                asyncio.sleep(delay)

    async def make_options(self, round: int):
        for _ in range(round):
            try:
                lock = await self._get(self._gpu_lock, delay=1)
                options = self._proposer(self.in_sizes)
                with torch.autograd.set_grad_enabled(False):
                    options_with_noise = options + torch.normal(0, std=options / 10)
            finally:
                self._gpu_lock.put(lock)

            for o0, o1 in zip(options_with_noise, options):
                asyncio.ensure_future(self.evaluate_option(o0), loop=self._event_loop)
                asyncio.ensure_future(self.evaluate_option(o1), loop=self._event_loop)

            # todo optimizer and step
