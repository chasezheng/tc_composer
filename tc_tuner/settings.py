import asyncio
import os
from multiprocessing import Manager
from torch import multiprocessing
from tc_composer.settings import get_configured_logger
import pickle

SYNC_MANAGER = Manager()
EVENT_LOOP: asyncio.BaseEventLoop = asyncio.get_event_loop()
SAVE_DIR = os.path.join(os.path.expanduser('~'), __package__)
DEFAULT_PICKLE_PROTOCOL = 4

LOGGER = get_configured_logger(__name__)
LOGGER.info("Setting multiprocessing `start_method` to 'spawn'.")
LOGGER.info(f'Setting default pickle protocol to {DEFAULT_PICKLE_PROTOCOL}')

pickle.DEFAULT_PROTOCOL = DEFAULT_PICKLE_PROTOCOL

multiprocessing.set_start_method('spawn', force=True)
