import numpy as np

from CONSTANTS import *


def split_train_dev_test(instances, train_ratio, dev_ratio):
    """
    Generic function to split instances into train, dev, and test.
    """
    dev_split = int(dev_ratio * len(instances))
    train_split = int(train_ratio * len(instances))
    train = instances[: (train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split) :]
    return train, dev, test


def filter_anomalous_by_block(train_instances, block_index=0, num_blocks=100):
    """
    Deterministically pick a block of anomalous instances.

    - block_index: which block to pick (0-based)
    - num_blocks: total number of blocks to split anomalous instances into
    """
    normal_instances = [ins for ins in train_instances if ins.label != "Anomalous"]
    anomalous_instances = [ins for ins in train_instances if ins.label == "Anomalous"]

    if anomalous_instances:
        # Split anomalous instances into blocks
        block_size = max(1, len(anomalous_instances) // num_blocks)
        start = block_index * block_size
        end = start + block_size
        # pick block, handle last block overflow
        anomalous_block = (
            anomalous_instances[start:end]
            if end <= len(anomalous_instances)
            else anomalous_instances[start:]
        )
    else:
        anomalous_block = []

    filtered_train = normal_instances + anomalous_block
    return filtered_train, block_index


# ================== Split Functions ==================


def cut_by_ratio(instances, train_ratio, dev_ratio):
    return split_train_dev_test(instances, train_ratio, dev_ratio)


def cut_all(instances):
    np.random.shuffle(instances)
    return instances, [], []


# ================== Filtered Versions ==================


def cut_by_ratio_filter(
    instances, train_ratio, dev_ratio, block_index=0, num_blocks=100
):
    original_train, dev, test = split_train_dev_test(instances, train_ratio, dev_ratio)
    train, _ = filter_anomalous_by_block(original_train, block_index, num_blocks)
    return train, dev, test


# ================== Examples ==================


# 6:1:3 split (with optional filter)
def cut_by_613(instances):
    return cut_by_ratio(instances, 0.6, 0.1)


def cut_by_613_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.6, 0.1, block_index)


# 3:1:6 split
def cut_by_316(instances):
    return cut_by_ratio(instances, 0.3, 0.1)


def cut_by_316_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.3, 0.1, block_index)


# 4:1:5 split
def cut_by_415(instances):
    return cut_by_ratio(instances, 0.4, 0.1)


def cut_by_415_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.4, 0.1, block_index)


# 5:1:4 split
def cut_by_514(instances):
    return cut_by_ratio(instances, 0.5, 0.1)


def cut_by_514_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.5, 0.1, block_index)


# 2:1:7 split
def cut_by_217(instances):
    return cut_by_ratio(instances, 0.2, 0.1)


def cut_by_172_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.1, 0.7, block_index)


def cut_by_253_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.2, 0.5, block_index)


def cut_by_226_filter(instances, block_index=0):
    return cut_by_ratio_filter(instances, 0.2, 0.2, block_index)
    return cut_by_ratio_filter(instances, 0.2, 0.2, block_index)
