import json
import os
import psutil
import torch
import datadog
from datadog import ThreadStats

with open(os.path.join(os.path.expanduser('~'), '.datadog_key'), 'rb') as f:
    datadog.initialize(**json.load(f))


class TunerStats(ThreadStats):
    _PREFIX = set()

    def __init__(self, name: str):
        super(TunerStats, self).__init__()
        assert name not in self._PREFIX, f"name={name} already exists."
        self.name = name
        self.best_10_options = []

    def gauge(self, metric_name, value, timestamp=None, tags=(), sample_rate=1, host=None):
        return super(TunerStats, self).gauge(
            metric_name=f"{metric_name}",
            value=value,
            timestamp=timestamp,
            tags=(*tags, 'name:{}'.format(self.name)),
            sample_rate=sample_rate,
            host=host
        )

    def increment(self, metric_name, value=1, timestamp=None, tags=None, sample_rate=1, host=None):
        return super(TunerStats, self).increment(
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
        for n,p in enumerate(psutil.cpu_percent(percpu=True)):
            self.gauge(f'cpu_usage', p, tags=[f'key:{n}'])

        mem_stats = psutil.virtual_memory()
        self.gauge('memory_usage', mem_stats.total, tags=[f'key:total'])
        self.gauge('memory_usage', mem_stats.available, tags=[f'key:available'])

    def log_option(self, opt: str, t: float) -> None:
        self.gauge(f"option_perf", t)

        if len(self.best_10_options) < 10 or t < self.best_10_options[-1][0]:
            self.best_10_options.append((t, opt))
            self.best_10_options.sort()
            self.best_10_options = self.best_10_options[:10]
            self.event(
                title=f"New best 10 option: {t}",
                text=opt,
                alert_type='success',
                aggregation_key='log_option'
            )
