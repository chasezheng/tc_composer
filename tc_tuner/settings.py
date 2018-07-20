import asyncio
import os
from torch import multiprocessing
from tc_composer.settings import get_configured_logger

EVENT_LOOP: asyncio.BaseEventLoop = asyncio.get_event_loop()
SAVE_DIR = os.path.join(os.path.expanduser('~'), __package__)

LOGGER = get_configured_logger(__name__)
LOGGER.info("Settings multiprocessing `start_method` to 'spawn'.")

multiprocessing.set_start_method('spawn', force=True)
