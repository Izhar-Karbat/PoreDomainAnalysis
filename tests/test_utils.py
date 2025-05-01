import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import base64
import pytest
import os  # Added for setting config env var if needed

# Assuming core utilities are importable
from pore_analysis.core.utils import (
    frames_to_time,
    OneLetter,
    fig_to_base64,
    clean_json_data,
)
# Import config directly to test FRAMES_PER_NS interaction
from pore_analysis.core import config as core_config

# Helper to reset config value if changed by monkeypatch
original_frames_per_ns = core_config.FRAMES_PER_NS

@pytest.fixture(autouse=True)
def reset_config():
    """Reset FRAMES_PER_NS after each test."""
    yield
    core_config.FRAMES_PER_NS = original_frames_per_ns

# Revised test for frames_to_time to use monkeypatch
def test_frames_to_time_basic(monkeypatch):
    # Test with the actual config value loaded initially
    original_fpns = core_config.FRAMES_PER_NS
    arr_original = frames_to_time([0, original_fpns, original_fpns * 2])
    assert np.allclose(arr_original, np.array([0.0, 1.0, 2.0])), \
        f"Test failed with original FRAMES_PER_NS={original_fpns}"

    # Test with a specific patched value
    monkeypatch.setattr(core_config, 'FRAMES_PER_NS', 20.0)
    # OPTIONALLY: If the function still doesn't see the change, patch the utils module directly:
    # monkeypatch.setattr(pore_analysis.core.utils, 'FRAMES_PER_NS', 20.0)
    arr_20 = frames_to_time([0, 20, 40])
    assert np.allclose(arr_20, np.array([0.0, 1.0, 2.0])), \
        f"Test failed with patched FRAMES_PER_NS=20.0, got {arr_20}"


def test_frames_to_time_invalid(monkeypatch):
    # Set FRAMES_PER_NS directly on the imported config module
    monkeypatch.setattr(core_config, 'FRAMES_PER_NS', 0)
    with pytest.raises(ValueError, match="FRAMES_PER_NS must be positive"):
        frames_to_time([0, 1, 2])

    monkeypatch.setattr(core_config, 'FRAMES_PER_NS', -10)
    with pytest.raises(ValueError, match="FRAMES_PER_NS must be positive"):
        frames_to_time([0, 1, 2])


def test_oneletter_basic():
    assert OneLetter('CYSASPGLY') == 'CDG'
    assert OneLetter('TRPVALGLUALA') == 'WVEA'
    assert OneLetter('HIS') == 'H'  # Test single
    assert OneLetter('HSE') == 'H'  # Test variant


def test_oneletter_case_insensitive():
    assert OneLetter('cysAspgLy') == 'CDG'


def test_oneletter_error_unknown():
    with pytest.raises(ValueError, match="Unknown amino acid code 'XXX'"):
        OneLetter('GLYXXXALA')


def test_oneletter_error_length():
    with pytest.raises(ValueError, match="Input length.*must be a multiple of three"):
        OneLetter('GLYA')
    with pytest.raises(ValueError, match="Input length.*must be a multiple of three"):
        OneLetter('GL')


def test_fig_to_base64():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100  # Should be a reasonably long string
        # Check if it decodes and looks like PNG header
        data = base64.b64decode(b64)
        assert data.startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close(fig)  # Ensure figure is closed


def test_fig_to_base64_error(monkeypatch):
    # Simulate an error during fig.savefig
    def mock_savefig(*args, **kwargs):
        raise IOError("Simulated save error")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    monkeypatch.setattr(fig, 'savefig', mock_savefig)
    try:
        b64 = fig_to_base64(fig)
        assert b64 == "", "Should return empty string on error"
    finally:
        plt.close(fig)


def test_clean_json_data_numpy_types():
    data = {
        'int_': np.int64(10),
        'float_': np.float32(3.14),
        'bool_': np.bool_(True),
        'array_': np.array([1, 2, 3]),
        'nan_': np.nan,
        'inf_': np.inf,
        'neg_inf_': -np.inf,
        'nested_array': np.array([np.nan, 4.0])
    }
    cleaned = clean_json_data(data)
    assert cleaned['int_'] == 10 and isinstance(cleaned['int_'], int)
    assert cleaned['float_'] == pytest.approx(3.14) and isinstance(cleaned['float_'], float) 
    assert cleaned['bool_'] is True and isinstance(cleaned['bool_'], bool)
    assert cleaned['array_'] == [1, 2, 3] and isinstance(cleaned['array_'], list)
    assert cleaned['nan_'] is None
    assert cleaned['inf_'] is None
    assert cleaned['neg_inf_'] is None
    assert cleaned['nested_array'] == [None, 4.0]


def test_clean_json_data_standard_types():
    data = {
        'a': math.nan,
        'b': [1, 2, math.inf],
        'c': None,
        'd': "string",
        'e': 123,
        'f': 4.56
    }
    cleaned = clean_json_data(data)
    assert cleaned['a'] is None
    assert cleaned['b'] == [1, 2, None]
    assert cleaned['c'] is None
    assert cleaned['d'] == "string"
    assert cleaned['e'] == 123
    assert cleaned['f'] == 4.56


def test_clean_json_data_nested():
    data = {
        'level1': {
            'list1': [np.int16(1), np.nan, {'key': np.array([5.0, np.inf])}]
        },
        'tuple1': (np.float64(9.9), None)
    }
    cleaned = clean_json_data(data)
    assert cleaned == {
        'level1': {
            'list1': [1, None, {'key': [5.0, None]}]
        },
        'tuple1': [9.9, None]  # Tuples become lists
    }


def test_clean_json_data_datetime():
    from datetime import datetime
    now = datetime.now()
    data = {'timestamp': now}
    cleaned = clean_json_data(data)
    assert cleaned['timestamp'] == now.isoformat()
