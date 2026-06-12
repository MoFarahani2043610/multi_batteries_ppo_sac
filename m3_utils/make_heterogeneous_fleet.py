"""
m3_utils/make_heterogeneous_fleet.py
=====================================
Milestone 3 — heterogeneous battery fleet generator.

Draws per-battery Pmax, Rmax, eta from a seeded distribution so all
algorithms see identical setups across the scalability sweep.

Usage
-----
    from m3_utils.make_heterogeneous_fleet import make_heterogeneous_fleet

    fleet = make_heterogeneous_fleet(N=5, seed=42)
    # fleet is a list of 5 BatteryConfig objects with different specs
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

import numpy as np
from typing import Optional

from storage_arbitrage_env import BatteryConfig


def make_heterogeneous_fleet(
    n: int,
    seed: int = 0,
    capacity_range: tuple[float, float] = (0.5, 2.0),
    p_charge_range: tuple[float, float] = (0.2, 1.0),
    p_discharge_range: tuple[float, float] = (0.2, 1.0),
    eta_charge_range: tuple[float, float] = (0.88, 0.98),
    eta_discharge_range: tuple[float, float] = (0.88, 0.98),
    market_node: int = 0,
) -> list[BatteryConfig]:
    """
    Generate N heterogeneous BatteryConfig objects drawn from a
    seeded uniform distribution.

    CRITICAL: seed is fixed per (N, seed) cell so all algorithms in
    the M3 sweep see identical battery configurations. Never change
    the sampling logic after M3 experiments begin.

    Parameters
    ----------
    n : int
        Number of batteries.
    seed : int
        Master seed — use the experiment seed (0-4) for reproducibility.
    capacity_range : (float, float)
        Uniform range for capacity_mwh in MWh. Default (0.5, 2.0).
    p_charge_range : (float, float)
        Uniform range for p_charge_max in MW. Default (0.2, 1.0).
    p_discharge_range : (float, float)
        Uniform range for p_discharge_max in MW. Default (0.2, 1.0).
    eta_charge_range : (float, float)
        Uniform range for eta_charge. Default (0.88, 0.98).
    eta_discharge_range : (float, float)
        Uniform range for eta_discharge. Default (0.88, 0.98).
    market_node : int
        Market node index for all batteries. Default 0.

    Returns
    -------
    list[BatteryConfig]
        N BatteryConfig objects with heterogeneous specifications.

    Examples
    --------
    >>> fleet = make_heterogeneous_fleet(N=3, seed=0)
    >>> for b in fleet:
    ...     print(f"cap={b.capacity_mwh:.2f} MWh, p_chg={b.p_charge_max:.2f} MW, eta={b.eta_charge:.3f}")
    """
    rng = np.random.default_rng(seed)

    capacities      = rng.uniform(*capacity_range,      size=n)
    p_charges       = rng.uniform(*p_charge_range,       size=n)
    p_discharges    = rng.uniform(*p_discharge_range,    size=n)
    eta_charges     = rng.uniform(*eta_charge_range,     size=n)
    eta_discharges  = rng.uniform(*eta_discharge_range,  size=n)

    fleet = []
    for i in range(n):
        fleet.append(BatteryConfig(
            capacity_mwh      = float(capacities[i]),
            p_charge_max      = float(p_charges[i]),
            p_discharge_max   = float(p_discharges[i]),
            eta_charge        = float(eta_charges[i]),
            eta_discharge     = float(eta_discharges[i]),
            soc_min           = 0.0,
            soc_max           = 1.0,
            initial_soc       = None,   # randomised at each episode reset
            market_node       = market_node,
        ))

    return fleet


def print_fleet_summary(fleet: list[BatteryConfig]) -> None:
    """Print a readable summary of a fleet."""
    print(f"\nFleet summary ({len(fleet)} batteries):")
    print(f"  {'Battery':<10} {'Cap (MWh)':<12} {'P_chg (MW)':<12} {'P_dis (MW)':<12} {'eta_c':<8} {'eta_d':<8}")
    print("  " + "-" * 62)
    for i, b in enumerate(fleet):
        print(f"  {i:<10} {b.capacity_mwh:<12.3f} {b.p_charge_max:<12.3f} "
              f"{b.p_discharge_max:<12.3f} {b.eta_charge:<8.3f} {b.eta_discharge:<8.3f}")
    total_cap = sum(b.capacity_mwh for b in fleet)
    total_pwr = sum(b.p_charge_max for b in fleet)
    print(f"  {'TOTAL':<10} {total_cap:<12.3f} {total_pwr:<12.3f}")


if __name__ == "__main__":
    # Quick test — verify determinism and print example fleets
    print("=" * 60)
    print("make_heterogeneous_fleet — sanity check")
    print("=" * 60)

    for n in [1, 2, 5, 10, 20]:
        fleet = make_heterogeneous_fleet(n, seed=0)
        assert len(fleet) == n
        # verify determinism: same seed → same fleet
        fleet2 = make_heterogeneous_fleet(n, seed=0)
        for b1, b2 in zip(fleet, fleet2):
            assert b1.capacity_mwh == b2.capacity_mwh
        print(f"N={n:2d}: determinism check PASSED")

    print("\nExample fleet (N=5, seed=0):")
    print_fleet_summary(make_heterogeneous_fleet(5, seed=0))

    print("\nExample fleet (N=5, seed=1):")
    print_fleet_summary(make_heterogeneous_fleet(5, seed=1))

    print("\nAll checks passed.")