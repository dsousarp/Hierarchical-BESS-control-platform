"""Clean apples-to-apples strategy comparison for v4_electrical_rc_model.

Runs four strategies through the EXACT SAME simulation loop:
  1. Full optimizer  (EMS + MPC + EKF/MHE)
  2. EMS + estimator (EMS + EKF/MHE, no MPC)
  3. EMS only        (EMS, no MPC, no estimator — true plant SOC)
  4. Rule-based      (price-sorted schedule, no optimization)

All use the same:
  - BatteryPack plant model (identical initial conditions)
  - Profit accounting formula (single code path)
  - Time resolution (dt_sim, dt_mpc)
  - Price scenarios and realized prices

Usage:
    uv run python v4_electrical_rc_model/run_comparison.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import pathlib
import sys
import time
from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.parameters import (
    BatteryParams, EKFParams, ElectricalParams, EMSParams,
    MHEParams, MPCParams, PackParams, ThermalParams, TimeParams,
)
from data.real_price_loader import RealPriceLoader
from ems.economic_ems import EconomicEMS
from estimation.ekf import ExtendedKalmanFilter
from estimation.mhe import MovingHorizonEstimator
from models.battery_model import BatteryPack
from mpc.tracking_mpc import TrackingMPC
from simulation.simulator import interpolate_ems_to_mpc

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
ENERGY_CSV = PROJECT_ROOT / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = PROJECT_ROOT / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = PROJECT_ROOT.parent / "results"

# ---------------------------------------------------------------------------
#  Regulation delivery penalty — parameters from EMSParams
#  (reg_soc_margin, reg_penalty_mult)
# ---------------------------------------------------------------------------


# ===========================================================================
#  Strategy base class
# ===========================================================================

class ControlStrategy(ABC):
    """Base class for control strategies. All return u = [P_chg, P_dis, P_reg]."""

    @abstractmethod
    def get_command(
        self,
        sim_step: int,
        x_true: np.ndarray,
        y_meas: np.ndarray,
        u_prev: np.ndarray,
        steps_per_ems: int,
        steps_per_mpc: int,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Return u = [P_chg, P_dis, P_reg] for this step."""
        ...


# ===========================================================================
#  Strategy 1: Full Optimizer (EMS + MPC + estimator)
# ===========================================================================

class FullOptimizerStrategy(ControlStrategy):
    def __init__(self, bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.mpc = TrackingMPC(bp, tp, mp, thp, elp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self.mhe = MovingHorizonEstimator(bp, tp, mhe_p, thp, elp)
        self.mp = mp
        self._mpc_refs = None
        self._mpc_ref_base = 0
        self._u_current = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        bp, tp, ep = self.bp, self.tp, self.ep

        # --- EMS update (every dt_ems) ---
        if sim_step % steps_per_ems == 0:
            x_est = self.ekf.get_estimate()
            ems_hour = sim_step // steps_per_ems
            remaining = min(ep.N_ems, energy_scenarios.shape[1] - ems_hour)
            remaining = max(remaining, 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]

            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            ems_result = self.ems.solve(
                soc_init=x_est[0], soh_init=x_est[1], t_init=x_est[2],
                vrc1_init=x_est[3], vrc2_init=x_est[4],
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            self._mpc_refs = interpolate_ems_to_mpc(ems_result, tp.dt_ems, tp.dt_mpc)
            self._mpc_ref_base = 0

        # --- MPC + estimator update (every dt_mpc) ---
        if sim_step % steps_per_mpc == 0 and self._mpc_refs is not None:
            # Estimator
            if sim_step > 0:
                ekf_est = self.ekf.step(self._u_current, y_meas)
                self.mhe.step(self._u_current, y_meas)
            else:
                ekf_est = self.ekf.get_estimate()

            # MPC reference windows
            refs = self._mpc_refs
            N = self.mp.N_mpc
            b = self._mpc_ref_base
            pc_win = refs["P_chg_ref_mpc"][b:b + N]
            pd_win = refs["P_dis_ref_mpc"][b:b + N]
            pr_win = refs["P_reg_ref_mpc"][b:b + N]
            soc_win = refs["SOC_ref_mpc"][b:b + N + 1]
            soh_win = refs["SOH_ref_mpc"][b:b + N + 1]
            temp_win = refs["TEMP_ref_mpc"][b:b + N + 1]

            # Pad if near end
            def _pad(arr, target_len, is_state=False):
                if len(arr) >= target_len:
                    return arr[:target_len]
                return np.pad(arr, (0, target_len - len(arr)), mode="edge")

            pc_win = _pad(pc_win, N)
            pd_win = _pad(pd_win, N)
            pr_win = _pad(pr_win, N)
            soc_win = _pad(soc_win, N + 1, True)
            soh_win = _pad(soh_win, N + 1, True)
            temp_win = _pad(temp_win, N + 1, True)

            try:
                self._u_current = self.mpc.solve(
                    x_est=ekf_est, soc_ref=soc_win, soh_ref=soh_win,
                    temp_ref=temp_win, p_chg_ref=pc_win, p_dis_ref=pd_win,
                    p_reg_ref=pr_win, u_prev=self._u_current,
                )
            except Exception:
                self._u_current = np.zeros(3)

            self._mpc_ref_base += 1

        return self._u_current.copy()


# ===========================================================================
#  Strategy 2: EMS + Estimator (no MPC)
# ===========================================================================

class EMSEstimatorStrategy(ControlStrategy):
    def __init__(self, bp, tp, ep, ekf_p, mhe_p, thp, elp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self.mhe = MovingHorizonEstimator(bp, tp, mhe_p, thp, elp)
        self._hourly_cmd = np.zeros(3)
        self._u_current = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        ep = self.ep

        # Estimator update at MPC rate (for state feedback to EMS)
        if sim_step % steps_per_mpc == 0:
            if sim_step > 0:
                self.ekf.step(self._u_current, y_meas)
                self.mhe.step(self._u_current, y_meas)

        # EMS update (every dt_ems) — uses estimated state
        if sim_step % steps_per_ems == 0:
            x_est = self.ekf.get_estimate()
            ems_hour = sim_step // steps_per_ems
            remaining = max(min(ep.N_ems, energy_scenarios.shape[1] - ems_hour), 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]
            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            ems_result = self.ems.solve(
                soc_init=x_est[0], soh_init=x_est[1], t_init=x_est[2],
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            # Apply first hour's command directly (no MPC tracking)
            self._hourly_cmd = np.array([
                float(ems_result["P_chg_ref"][0]),
                float(ems_result["P_dis_ref"][0]),
                float(ems_result["P_reg_ref"][0]),
            ])

        self._u_current = self._hourly_cmd.copy()
        return self._u_current.copy()


# ===========================================================================
#  Strategy 3: EMS Only (no MPC, no estimator — uses true plant SOC)
# ===========================================================================

class EMSOnlyStrategy(ControlStrategy):
    def __init__(self, bp, tp, ep, thp, elp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self._hourly_cmd = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        ep = self.ep

        if sim_step % steps_per_ems == 0:
            ems_hour = sim_step // steps_per_ems
            remaining = max(min(ep.N_ems, energy_scenarios.shape[1] - ems_hour), 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]
            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            # Uses TRUE plant SOC (ideal BMS, no estimation noise)
            ems_result = self.ems.solve(
                soc_init=float(x_true[0]),
                soh_init=float(x_true[1]),
                t_init=float(x_true[2]),
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            self._hourly_cmd = np.array([
                float(ems_result["P_chg_ref"][0]),
                float(ems_result["P_dis_ref"][0]),
                float(ems_result["P_reg_ref"][0]),
            ])

        return self._hourly_cmd.copy()


# ===========================================================================
#  Strategy 4: Rule-Based (price-sorted, fixed schedule)
# ===========================================================================

class RuleBasedStrategy(ControlStrategy):
    def __init__(self, bp, ep):
        self.bp = bp
        self.reg_fraction = ep.regulation_fraction
        self._schedule = None  # shape (24, 3)

    def plan(
        self,
        forecast_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Build a fixed 24h schedule from the expected price profile.

        Uses the probability-weighted average across forecast scenarios
        (NOT the realized prices — no oracle).
        """
        bp = self.bp
        # Expected price = weighted average across scenarios
        expected_prices = np.average(forecast_scenarios[:, :24], axis=0, weights=probabilities)

        sorted_hours = np.argsort(expected_prices)
        usable = (bp.SOC_max - bp.SOC_min) * bp.E_nom_kwh
        n_ch = int(np.ceil(usable / bp.P_max_kw))  # hours at full power
        charge_hours = set(sorted_hours[:n_ch])
        discharge_hours = set(sorted_hours[-n_ch:])
        overlap = charge_hours & discharge_hours
        charge_hours -= overlap
        discharge_hours -= overlap

        self._schedule = np.zeros((24, 3))
        for h in range(24):
            if h in charge_hours:
                self._schedule[h, 0] = bp.P_max_kw       # P_chg
            elif h in discharge_hours:
                self._schedule[h, 1] = bp.P_max_kw       # P_dis
            # Fixed regulation commitment every hour
            self._schedule[h, 2] = bp.P_max_kw * self.reg_fraction

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        hour = min(sim_step // steps_per_ems, 23)
        return self._schedule[hour].copy()


# ===========================================================================
#  Unified simulation loop (SINGLE profit accounting for all strategies)
# ===========================================================================

def unified_sim_loop(
    strategy: ControlStrategy,
    plant: BatteryPack,
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    realized_energy_prices: np.ndarray,   # shape (n_hours,) — actual $/kWh
    realized_reg_prices: np.ndarray,      # shape (n_hours,) — actual $/kW/h
    forecast_energy_scen: np.ndarray,     # shape (n_scen, n_hours) — for EMS planning
    forecast_reg_scen: np.ndarray,        # shape (n_scen, n_hours)
    forecast_probs: np.ndarray,           # shape (n_scen,)
) -> dict:
    """Run one 24h simulation. Returns profit and operational metrics.

    Profit formula (IDENTICAL for all strategies, from simulator.py:386-400):
        e_profit = price_e * (P_dis - P_chg) * dt_h
        r_profit = price_r * P_reg * dt_h     [capacity payment]
        d_cost   = deg_cost * alpha_deg * (P_chg + P_dis + P_reg) * dt_mpc
        reg_penalty = penalty if SOC near limits and P_reg > 0

    The plant is stepped at dt_sim resolution.
    Commands update at dt_mpc resolution.
    EMS updates at dt_ems resolution (strategy-dependent).
    """
    total_seconds = int(tp.sim_hours * 3600)
    steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)
    steps_per_ems = int(tp.dt_ems / tp.dt_sim)
    N_sim_steps = int(total_seconds / tp.dt_sim)
    dt_h = tp.dt_mpc / 3600.0

    cum_profit = 0.0
    u_current = np.zeros(3)
    soh_init = plant.get_state()[1]

    for sim_step in range(N_sim_steps):
        x_true = plant.get_state()
        y_meas = plant.get_measurement()

        # --- Get control command from strategy (at dt_mpc rate) ---
        if sim_step % steps_per_mpc == 0:
            u_current = strategy.get_command(
                sim_step=sim_step,
                x_true=x_true,
                y_meas=y_meas,
                u_prev=u_current,
                steps_per_ems=steps_per_ems,
                steps_per_mpc=steps_per_mpc,
                energy_scenarios=forecast_energy_scen,
                reg_scenarios=forecast_reg_scen,
                probabilities=forecast_probs,
            )

            # --- Profit accounting (ONCE, identical for all strategies) ---
            ems_hour = min(sim_step // steps_per_ems,
                           len(realized_energy_prices) - 1)
            price_e = float(realized_energy_prices[ems_hour])
            price_r = float(realized_reg_prices[ems_hour])

            e_profit = price_e * (u_current[1] - u_current[0]) * dt_h
            r_profit = price_r * u_current[2] * dt_h
            d_cost = (ep.degradation_cost * bp.alpha_deg
                      * (u_current[0] + u_current[1] + u_current[2])
                      * tp.dt_mpc)

            # Regulation delivery penalty
            soc = x_true[0]
            if u_current[2] > 0.1:
                can_deliver = (soc > bp.SOC_min + ep.reg_soc_margin
                               and soc < bp.SOC_max - ep.reg_soc_margin)
                if not can_deliver:
                    r_profit = -ep.reg_penalty_mult * price_r * u_current[2] * dt_h

            cum_profit += e_profit + r_profit - d_cost

        # --- Step the plant ---
        plant.step(u_current)

    x_final = plant.get_state()
    soh_final = x_final[1]

    return {
        "profit": float(cum_profit),
        "soh_degradation": float(soh_init - soh_final),
        "final_soc": float(x_final[0]),
        "final_soh": float(x_final[1]),
    }


# ===========================================================================
#  Per-day worker
# ===========================================================================

def _run_single_day(args: tuple) -> dict:
    """Run all 4 strategies on one day. Designed for multiprocessing.Pool."""
    (day_idx, realized_energy_24, realized_reg_24,
     forecast_energy_48, forecast_reg_48, forecast_probs,
     bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp) = args

    logging.disable(logging.WARNING)

    results = {"day_idx": day_idx}
    t0_total = time.perf_counter()

    strategies = {
        "rule_based": lambda: _make_rule_based(bp, ep, forecast_energy_48, forecast_probs),
        "ems_only": lambda: EMSOnlyStrategy(bp, tp, ep, thp, elp),
        "ems_est": lambda: EMSEstimatorStrategy(bp, tp, ep, ekf_p, mhe_p, thp, elp),
        "optimizer": lambda: FullOptimizerStrategy(bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp),
    }

    # Pad realized prices to cover lookahead
    n_hours_needed = int(tp.sim_hours) + ep.N_ems
    re_padded = np.pad(realized_energy_24, (0, max(0, n_hours_needed - len(realized_energy_24))), mode="edge")
    rr_padded = np.pad(realized_reg_24, (0, max(0, n_hours_needed - len(realized_reg_24))), mode="edge")

    for name, make_strategy in strategies.items():
        t0 = time.perf_counter()
        plant = BatteryPack(bp, tp, thp, elp, pp)
        strategy = make_strategy()

        res = unified_sim_loop(
            strategy=strategy,
            plant=plant,
            bp=bp, tp=tp, ep=ep,
            realized_energy_prices=re_padded,
            realized_reg_prices=rr_padded,
            forecast_energy_scen=forecast_energy_48,
            forecast_reg_scen=forecast_reg_48,
            forecast_probs=forecast_probs,
        )
        res["wall_time"] = time.perf_counter() - t0
        results[name] = res

    results["total_wall_time"] = time.perf_counter() - t0_total
    return results


def _make_rule_based(bp, ep, forecast_energy, forecast_probs):
    """Create and plan a rule-based strategy."""
    strategy = RuleBasedStrategy(bp, ep)
    strategy.plan(forecast_energy, forecast_probs)
    return strategy


# ===========================================================================
#  Scenario generation (realized prices NOT in forecast set)
# ===========================================================================

def build_day_scenarios(
    loader: RealPriceLoader,
    day_idx: int,
    n_forecast_scenarios: int,
    rng: np.random.Generator,
    n_hours: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forecast scenarios that do NOT contain the realized day.

    Returns:
        realized_energy_24: actual day's prices (24,) [$/kWh]
        realized_reg_24:    actual day's regulation prices (24,) [$/kWh]
        forecast_energy:    (n_scenarios, n_hours) — other days, not the actual
        forecast_reg:       (n_scenarios, n_hours)
        forecast_probs:     (n_scenarios,) — uniform
    """
    realized_energy_24 = loader.get_day(day_idx)

    # Build regulation for realized day
    if loader.has_real_regulation:
        realized_reg_24 = loader._daily_reg[day_idx].copy()
    else:
        realized_reg_24 = 0.4 * realized_energy_24 + 0.006

    # Sample OTHER days for forecast (exclude the realized day)
    other_days = [i for i in range(loader.n_days) if i != day_idx]
    chosen = rng.choice(other_days, size=n_forecast_scenarios, replace=False)

    forecast_energy = np.zeros((n_forecast_scenarios, n_hours))
    forecast_reg = np.zeros((n_forecast_scenarios, n_hours))

    for s_idx, d_idx in enumerate(chosen):
        e48, r48 = loader._build_48h(d_idx)
        forecast_energy[s_idx, :n_hours] = e48[:n_hours]
        forecast_reg[s_idx, :n_hours] = r48[:n_hours]

    forecast_energy = np.maximum(forecast_energy, 0.001)
    forecast_reg = np.maximum(forecast_reg, 0.0)
    forecast_probs = np.ones(n_forecast_scenarios) / n_forecast_scenarios

    return (realized_energy_24, realized_reg_24,
            forecast_energy, forecast_reg, forecast_probs)


# ===========================================================================
#  Main
# ===========================================================================

def main() -> None:
    N_DAYS = 84

    # ---- Calibrated parameters (same as validate_real_prices.py) ----
    bp = dataclasses.replace(BatteryParams(), alpha_deg=4.76e-11)
    tp = dataclasses.replace(TimeParams(), dt_mpc=300.0, dt_estimator=300.0, dt_sim=10.0)
    ep = dataclasses.replace(EMSParams(), degradation_cost=36_500.0)
    mp = dataclasses.replace(MPCParams(), N_mpc=12, Nc_mpc=4)
    ekf_p = EKFParams()
    mhe_p = dataclasses.replace(MHEParams(), N_mhe=6)
    thp = dataclasses.replace(ThermalParams(), R_internal=0.072, h_cool=150.0, C_thermal=300_000.0)
    elp = dataclasses.replace(ElectricalParams(), R0=0.0324, R1=0.0216, R2=0.0180)
    pp = PackParams()

    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)
    rng = np.random.default_rng(seed=777)
    n_hours_total = int(tp.sim_hours) + ep.N_ems

    print("=" * 70)
    print("  CLEAN STRATEGY COMPARISON [v4_electrical_rc_model]")
    print("=" * 70)
    stats = loader.price_stats
    print(f"  Data:       German day-ahead + FCR, Q1 2024 ({stats['n_days']} days)")
    print(f"  Strategies: optimizer, EMS+est, EMS only, rule-based")
    print(f"  Days:       {N_DAYS}")
    print(f"  Scenarios:  forecast = 5 other days (realized NOT in set)")
    print(f"  Reg penalty: {ep.reg_penalty_mult}x capacity price when SOC near limits")
    print("=" * 70)
    print()

    # ---- Build jobs ----
    day_indices = list(range(min(N_DAYS, loader.n_days)))
    jobs = []
    for day_idx in day_indices:
        re24, rr24, fe, fr, fp = build_day_scenarios(
            loader, day_idx, n_forecast_scenarios=5, rng=rng, n_hours=n_hours_total,
        )
        jobs.append((day_idx, re24, rr24, fe, fr, fp,
                      bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp))

    # ---- Run ----
    n_workers = min(len(jobs), multiprocessing.cpu_count(), 3)
    print(f"  Running {len(jobs)} days x 4 strategies across {n_workers} workers...\n")
    t0 = time.perf_counter()

    with multiprocessing.Pool(n_workers) as pool:
        all_results = pool.map(_run_single_day, jobs)

    wall = time.perf_counter() - t0
    print(f"\n  Done in {wall:.0f}s ({wall/len(jobs):.0f}s/day)\n")

    # ---- Aggregate ----
    EUR = 1 / 1.08
    strat_names = ["rule_based", "ems_only", "ems_est", "optimizer"]

    summary = {"n_days": len(all_results), "per_day": []}
    agg = {s: {"profits": [], "soh_degs": []} for s in strat_names}

    for r in all_results:
        day_entry = {"day_idx": r["day_idx"]}
        for s in strat_names:
            sr = r[s]
            day_entry[s] = sr
            agg[s]["profits"].append(sr["profit"])
            agg[s]["soh_degs"].append(sr["soh_degradation"])
        summary["per_day"].append(day_entry)

    # Print table
    print(f"  {'':22s}", end="")
    for s in strat_names:
        print(f"  {s:>14s}", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in strat_names:
        print(f"  {'─'*14}", end="")
    print()

    for label, fn in [
        ("Mean EUR/day", lambda v: np.mean(v) * EUR),
        ("Median EUR/day", lambda v: np.median(v) * EUR),
        ("Std EUR/day", lambda v: np.std(v) * EUR),
        ("P5 EUR/day", lambda v: np.percentile(v, 5) * EUR),
        ("P95 EUR/day", lambda v: np.percentile(v, 95) * EUR),
        ("Worst EUR/day", lambda v: np.min(v) * EUR),
        ("Best EUR/day", lambda v: np.max(v) * EUR),
        ("Loss days", lambda v: int(np.sum(np.array(v) < 0))),
        ("SOH %/day", lambda v: np.mean(v) * 100),
    ]:
        print(f"  {label:22s}", end="")
        for s in strat_names:
            val = fn(agg[s]["profits"] if "EUR" in label or "Loss" in label else agg[s]["soh_degs"])
            if isinstance(val, int):
                print(f"  {val:14d}", end="")
            else:
                print(f"  {val:14.2f}", end="")
        print()

    # Advantage summary
    opt = np.array(agg["optimizer"]["profits"]) * EUR
    rb = np.array(agg["rule_based"]["profits"]) * EUR
    adv = opt - rb
    print(f"\n  Optimizer vs Rule-Based:")
    print(f"    Advantage:  EUR {adv.mean():.2f}/day  ({(adv > 0).mean()*100:.0f}% win rate)")
    print(f"    Annual (200kWh): EUR {adv.mean()*365:.0f}")
    print(f"    Annual (10MWh):  EUR {adv.mean()*365*50:,.0f}")
    print(f"    Annual (50MWh):  EUR {adv.mean()*365*250:,.0f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "v4_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
