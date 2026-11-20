import numpy as np
import pytest

from team2.rq22_fit import fit, predict


@pytest.mark.parametrize("objective", ["log_huber", "log_mse", "raw_mse"])
def test_recovers_synthetic_relationship(objective):
    n = np.repeat([50e6, 100e6, 200e6, 350e6], 3)
    d = n * np.tile([10, 20, 30], 4)
    known = np.array([1.5, 0.8, 0.6, 0.35, 0.28])
    loss = predict(known, n, d)

    result = fit(n, d, loss, objective=objective)

    np.testing.assert_allclose(
        predict(result.x, n, d), loss, atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        predict(result.x, 1.15e9, 20e9),
        predict(known, 1.15e9, 20e9),
        rtol=1e-3,
    )


def test_rejects_two_model_sizes():
    n = np.repeat([50e6, 100e6], 3)
    d = n * np.tile([10, 20, 30], 2)
    loss = predict([1.5, 0.8, 0.6, 0.35, 0.28], n, d)

    with pytest.raises(ValueError, match="three distinct model sizes"):
        fit(n, d, loss)


@pytest.mark.parametrize("objective", ["log_huber", "log_mse", "raw_mse"])
def test_predicts_held_out_model_size(objective):
    n = np.repeat([50e6, 100e6, 200e6, 350e6], 3)
    d = n * np.tile([10, 20, 30], 4)
    known = [1.5, 0.8, 0.6, 0.35, 0.28]
    loss = predict(known, n, d)

    train = n < 350e6
    held_out = ~train
    result = fit(n[train], d[train], loss[train], objective=objective)

    np.testing.assert_allclose(
        predict(result.x, n[held_out], d[held_out]),
        loss[held_out],
        rtol=1e-3,
        atol=0,
    )
