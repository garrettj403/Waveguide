"""Tests for utilities sub-module."""

import numpy as np

import waveguide as wg


def test_db2np():
    """Test converting linear to dB and Np."""

    # Test array
    value = np.array([1, 10, 100])

    # Test dB value
    value_db = wg.db10(value)
    np.testing.assert_almost_equal(value_db, [0, 10, 20], decimal=10)

    # Convert to Np
    value_np = wg.db2np(value_db)
    true_np = np.array([0, 1.1512925465, 2.302585093])
    np.testing.assert_almost_equal(value_np, true_np, decimal=9)

    # Test dB 20 value
    value_db20 = wg.db20(value)
    np.testing.assert_almost_equal(value_db, value_db20 / 2)

    # Test back to dB
    value_db_back = wg.np2db(value_np)
    np.testing.assert_almost_equal(value_db_back, value_db)


def test_deg2rad():
    """Test converting degrees to radians."""

    # Test array
    angles = np.array([0, 90, 180])

    # Convert to radians
    angles_rad = wg.deg2rad(angles)
    true_rad = np.array([0, np.pi / 2, np.pi])
    np.testing.assert_almost_equal(angles_rad, true_rad)

    # Convert back to degrees
    angles_deg = wg.rad2deg(angles_rad)
    np.testing.assert_almost_equal(angles_deg, angles)


if __name__ == "__main__":

    test_db2np()
    test_deg2rad()
