import os
import sys
import pytest

pytest.importorskip("numpy")
pytest.importorskip("nnfs")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset_utils import get_dataset


def test_spiral_validate_shapes():
    X, y = get_dataset('spiral', classes=2, validate=True)
    assert X.shape == (200, 2)
    assert y.shape == (200,)


def test_invalid_dataset_name():
    with pytest.raises(ValueError):
        get_dataset('invalid', classes=2)
