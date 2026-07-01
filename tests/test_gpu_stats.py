"""Tests for NVML device index mapping."""

import os
from unittest import mock

from src.profiling.gpu_stats import nvml_device_index


def test_nvml_device_index_default():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert nvml_device_index() == 0


def test_nvml_device_index_visible_gpu():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
        assert nvml_device_index() == 1
