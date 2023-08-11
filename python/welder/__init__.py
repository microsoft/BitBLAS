import torch
import sys

from . import arch, relay
from .code_generator import CodeGenerator
from .IRpass import *

import logging

def set_log_level(level):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

def _init_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s [welder:%(levelname)s]: %(message)s", datefmt='%F %T')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level(logging.INFO)

_init_logger()
