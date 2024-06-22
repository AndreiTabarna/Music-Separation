import logging
import json
import sys
import torch
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager
import tqdm
from pathlib import Path

#This has the purpose of giving real-time feedback in the process of training

def logger(level : str = 'info'):
    ALLOWED_LEVELS = ['debug', 'info', 'warning', 'error', 'critical']
    ALLOWED_LEVELS.extend([x.upper() for x in ALLOWED_LEVELS])
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"logging level must be one of {ALLOWED_LEVELS}")
    
    logging.getLogger('sox').setLevel(logging.ERROR)

    level = getattr(logging, level.upper())
    logging.basicConfig(
        format='%(asctime)s | %(filename)s:%(lineno)d %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=level
    )


