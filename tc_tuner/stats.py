import json
import os

import datadog
import psutil
from datadog import ThreadStats
from typing import List
from .option_result import OptionResult

with open(os.path.join(os.path.expanduser('~'), '.datadog_key'), 'rb') as f:
    datadog.initialize(**json.load(f))


class TunerStats:
    __slots__ = 'name', 'best_10_options'
    _PREFIX = set()

    _STATS = ThreadStats()

    def __init__(self, name: str):
        assert name not in self._PREFIX, f"name={name} already exists."
        self.name = name
        self.best_10_options: List[float] = []

    def gauge(self, metric_name, value, timestamp=None, tags=(), sample_rate=1, host=None):
        return self._STATS.gauge(
            metric_name=f"{metric_name}",
            value=value,
            timestamp=timestamp,
            tags=(*tags, 'name:{}'.format(self.name)),
            sample_rate=sample_rate,
            host=host
        )

    def increment(self, metric_name, value=1, timestamp=None, tags=None, sample_rate=1, host=None):
        return self._STATS.increment(
            metric_name=f"{self.name}.{metric_name}",
            value=value,
            timestamp=timestamp,
            tags=tags,
            sample_rate=sample_rate,
            host=host
        )

    def async_stats(self, key: str, delay: float, backoff: float, recovery_bias: float) -> None:
        self.gauge("async_stats.delay", delay, tags=[f'key:{key}'])
        self.gauge("async_stats.backoff", backoff, tags=[f'key:{key}'])
        self.gauge("async_stats.recovery_bias", recovery_bias, tags=[f'key:{key}'])

    def queue_size(self, key: str, val: int) -> None:
        self.gauge(f"queue_size", val, tags=[f'key:{key}'])

    def queue_passage(self, key: str, value: int = 1) -> None:
        self.increment(f"queue_passage", tags=[f"key:{key}"], value=value)

    def resource_usage(self) -> None:
        for n, p in enumerate(psutil.cpu_percent(percpu=True)):
            self.gauge(f'cpu_usage', p, tags=[f'key:{n}'])

        mem_stats = psutil.virtual_memory()
        self.gauge('memory_usage', mem_stats.total, tags=[f'key:total'])
        self.gauge('memory_usage', mem_stats.available, tags=[f'key:available'])

    def log_option(self, opt_res: OptionResult) -> None:
        self.gauge(f"option_perf", opt_res.speed)

        if len(self.best_10_options) < 10 or opt_res.speed < self.best_10_options[-1].speed:
            self.best_10_options.append(opt_res)
            self.best_10_options.sort(key=lambda x: x.speed)
            self.best_10_options: List[OptionResult] = self.best_10_options[:10]
            self._STATS.event(
                title=f"New best 10 option: {opt_res.speed}",
                text=str(opt_res.option),
                alert_type='success',
                aggregation_key='log_option'
            )
