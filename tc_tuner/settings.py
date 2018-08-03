import asyncio
import os
import pickle

from torch import multiprocessing

from tc_composer.settings import get_configured_logger

EVENT_LOOP: asyncio.BaseEventLoop = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)
SAVE_DIR = os.path.join(os.path.expanduser('~'), __package__)

LOGGER = get_configured_logger(__name__)
LOGGER.info("Setting multiprocessing `start_method` to 'spawn'.")
LOGGER.info(f'Setting default pickle protocol to {pickle.HIGHEST_PROTOCOL}')

pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

multiprocessing.set_start_method('spawn', force=True)
