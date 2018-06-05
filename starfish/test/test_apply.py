from copy import deepcopy

import numpy as np

from starfish.test.dataset_fixtures import synthetic_stack, gold_standard_dataset


def multiply(array, value):
    return array * value


def test_apply(synthetic_stack):
    synthetic_stack = deepcopy(synthetic_stack)
    assert (synthetic_stack.numpy_array == 1).all()
    synthetic_stack.apply(multiply, value=2)
    assert (synthetic_stack.numpy_array == 2).all()


def test_apply_3d(synthetic_stack):
    synthetic_stack = deepcopy(synthetic_stack)
    assert np.all(synthetic_stack.numpy_array == 1)
    synthetic_stack.apply(multiply, value=4, is_volume=True)
    assert (synthetic_stack.numpy_array == 4).all()


def test_apply_gold_standard(gold_standard_dataset):
    stack, codebook = gold_standard_dataset
    synthetic_stack = deepcopy(stack)
    synthetic_stack = synthetic_stack.image
    synthetic_stack.apply(multiply, value=2)
    assert np.all(synthetic_stack.numpy_array == stack.image.numpy_array * 2)

