import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import pytest

from pore_analysis.core.utils import (
    frames_to_time,
    OneLetter,
    fig_to_base64,
    clean_json_data,
    FRAMES_PER_NS as UTILS_FRAMES_PER_NS
)

def test_frames_to_time_basic():
    arr = frames_to_time([0, UTILS_FRAMES_PER_NS, UTILS_FRAMES_PER_NS * 2])
    assert np.allclose(arr, np.array([0.0, 1.0, 2.0]))

def test_frames_to_time_invalid(monkeypatch):
    import pore_analysis.core.utils as utils
    monkeypatch.setattr(utils, 'FRAMES_PER_NS', 0)
    with pytest.raises(ValueError):
        frames_to_time([0, 1, 2])

def test_oneletter_basic():
    assert OneLetter('CYSASPGLY') == 'CDG'

def test_oneletter_error():
    with pytest.raises(ValueError):
        OneLetter('XXX')

def test_fig_to_base64():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    b64 = fig_to_base64(fig)
    assert isinstance(b64, str) and len(b64) > 0
    data = base64.b64decode(b64)
    # PNG files start with the byte signature 89 50 4E 47 0D 0A 1A 0A
    assert data[:8] == b'\x89PNG\r\n\x1a\n'

def test_clean_json_data():
    data = {
        'a': np.nan,
        'b': np.array([1, 2, np.inf]),
        'c': [np.int32(5), np.float64(3.14), {'x': np.bool_(True)}]
    }
    cleaned = clean_json_data(data)
    assert cleaned['a'] is None
    assert cleaned['b'] == [1, 2, None]
    assert cleaned['c'][0] == 5
    assert isinstance(cleaned['c'][1], float)
    assert cleaned['c'][2]['x'] is True
