import asyncio

from tc_tuner import settings
from tc_tuner.async_util import Repeat, repeat
from ..torch_test_case import TorchTestCase


class TestAsyncUtil(TorchTestCase):
    def test_smaller_delay(self):
        def smaller():
            return Repeat.DelayFactor(-2)

        f = Repeat(smaller)

        delay0 = f.retry_delay

        asyncio.ensure_future(f())
        settings.EVENT_LOOP.run_until_complete(asyncio.sleep(.3))

        self.assertLess(f.retry_delay, delay0)
        self.assertGreater(f.retry_delay, 0)

    def test_same_delay(self):
        def zero():
            return Repeat.DelayFactor(0)

        f = Repeat(zero)

        delay0 = f.retry_delay

        asyncio.ensure_future(f())
        settings.EVENT_LOOP.run_until_complete(asyncio.sleep(1))

        self.assertEqual(delay0, f.retry_delay)

    def test_greater_delay(self):
        def one():
            return Repeat.DelayFactor(1)

        f = Repeat(one)

        delay0 = f.retry_delay

        asyncio.ensure_future(f())
        settings.EVENT_LOOP.run_until_complete(asyncio.sleep(1))

        self.assertGreater(f.retry_delay, delay0)

    def test_backoff(self):
        ints = (i for i in (1, 1, 1))

        def get_int():
            return Repeat.DelayFactor(next(ints))

        f = Repeat(get_int)
        backoff0 = f.backoff

        asyncio.ensure_future(f())
        try:
            settings.EVENT_LOOP.run_until_complete(asyncio.sleep(.7))
        except StopIteration:
            pass

        self.assertGreater(f.backoff, backoff0)

    def test_smaller_backoff(self):
        ints = (i for i in (1, -1, 1))

        def get_int():
            return Repeat.DelayFactor(next(ints))

        f = Repeat(get_int)
        backoff0 = f.backoff

        asyncio.ensure_future(f())
        try:
            settings.EVENT_LOOP.run_until_complete(asyncio.sleep(.25))
        except StopIteration:
            pass

        self.assertLess(f.backoff, backoff0)

    def test_decorator(self):
        def one():
            return Repeat.DelayFactor(1)

        main = repeat(one)
        delay0 = main.retry_delay
        asyncio.ensure_future(main())
        settings.EVENT_LOOP.run_until_complete(asyncio.sleep(.5))

        self.assertGreater(main.retry_delay, delay0)

    def test_return(self):
        out = 2
        ints = (i for i in (Repeat.DelayFactor(1), out))

        def test_f():
            return next(ints)

        main = asyncio.ensure_future(repeat(test_f)())
        settings.EVENT_LOOP.run_until_complete(main)

        self.assertEqual(main.result(), out)

    def test_await(self):
        out = 2
        ints = (i for i in (Repeat.DelayFactor(1), out))

        async def test_async():
            return next(ints)

        main = asyncio.ensure_future(repeat(test_async)())
        settings.EVENT_LOOP.run_until_complete(main)

        self.assertEqual(main.result(), out)
