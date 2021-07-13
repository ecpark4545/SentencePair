import logging
import os
import random
import numpy as np
import torch

def init_logger(path: str):
	if not os.path.exists(path):
		os.makedirs(path)
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
	debug_fh.setLevel(logging.DEBUG)
	
	info_fh = logging.FileHandler(os.path.join(path, "info.log"))
	info_fh.setLevel(logging.INFO)
	
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	
	info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
	debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')
	
	ch.setFormatter(info_formatter)
	info_fh.setFormatter(info_formatter)
	debug_fh.setFormatter(debug_formatter)
	
	logger.addHandler(ch)
	logger.addHandler(debug_fh)
	logger.addHandler(info_fh)
	
	return logger


def set_seed(seed=1111):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

