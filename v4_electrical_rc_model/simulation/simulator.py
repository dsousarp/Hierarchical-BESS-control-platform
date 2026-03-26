"""Multi-rate simulation coordinator with 2RC electrical model and multi-cell pack.

v4_electrical_rc_model: 5-state dynamics (SOC, SOH, T, V_rc1, V_rc2),
3-measurement (SOC, T, V_term).

Time scales
-----------
  dt_sim  = 1 s         plant integration  (BatteryPack.step)
  dt_mpc  = 60 s        MPC solve  +  EKF / MHE update
  dt_ems  = 3 600 s     EMS re-solve
"""

from __future__ import annotations

import logging
import time

import numpy as np

from config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    ThermalParams,
    TimeParams,
)
from ems.economic_ems import EconomicEMS
from estimation.ekf import ExtendedKalmanFilter
from estimation.mhe import MovingHorizonEstimator
from models.battery_model import BatteryPack, BatteryPlant
from mpc.tracking_mpc import TrackingMPC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Reference interpolation
# ---------------------------------------------------------------------------

def interpolate_ems_to_mpc(
    ems_result: dict,
    dt_ems: float,
    dt_mpc: float,
) -> dict:
    """Interpolate hourly EMS references to MPC resolution."""
    ratio = int(round(dt_ems / dt_mpc))

    p_chg = np.repeat(ems_result["P_chg_ref"], ratio)
    p_dis = np.repeat(ems_result["P_dis_ref"], ratio)
    p_reg = np.repeat(ems_result["P_reg_ref"], ratio)

    soc_hourly = ems_result["SOC_ref"]
    soh_hourly = ems_result["SOH_ref"]
    temp_hourly = ems_result["TEMP_ref"]
    N_hours = len(soc_hourly) - 1
    N_mpc_pts = N_hours * ratio + 1

    t_hourly = np.arange(N_hours + 1, dtype=np.float64)
    t_mpc = np.linspace(0.0, N_hours, N_mpc_pts)

    soc_mpc = np.interp(t_mpc, t_hourly, soc_hourly)
    soh_mpc = np.interp(t_mpc, t_hourly, soh_hourly)
    temp_mpc = np.interp(t_mpc, t_hourly, temp_hourly)

    return {
        "P_chg_ref_mpc": p_chg,
        "P_dis_ref_mpc": p_dis,
        "P_reg_ref_mpc": p_reg,
        "SOC_ref_mpc": soc_mpc,
        "SOH_ref_mpc": soh_mpc,
        "TEMP_ref_mpc": temp_mpc,
    }


# ---------------------------------------------------------------------------
#  Multi-rate simulator
# ---------------------------------------------------------------------------

class MultiRateSimulator:
    """Coordinates plant, EMS, MPC, EKF, and MHE at their respective rates.

    v4: accepts ElectricalParams; uses 5-state model with 3 measurements.
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        mp: MPCParams,
        ekf_p: EKFParams,
        mhe_p: MHEParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        pp: PackParams | None = None,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.mp = mp
        self.thp = thp
        self.elp = elp
        self.pp = pp

        # Plant: multi-cell pack or single cell
        if pp is not None:
            self.plant = BatteryPack(bp, tp, thp, elp, pp)
            self.n_cells = pp.n_cells
        else:
            self.plant = BatteryPlant(bp, tp, thp, elp)
            self.n_cells = 1

        # Optimizer-level components (pack-level, 5-state)
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.mpc = TrackingMPC(bp, tp, mp, thp, elp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self.mhe = MovingHorizonEstimator(bp, tp, mhe_p, thp, elp)

    # ------------------------------------------------------------------
    #  Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:
        """Execute the full multi-rate closed-loop simulation."""
        bp = self.bp
        tp = self.tp
        ep = self.ep
        thp = self.thp

        # Timing
        total_seconds = int(tp.sim_hours * 3600)
        steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)
        steps_per_ems = int(tp.dt_ems / tp.dt_sim)
        N_sim_steps = int(total_seconds / tp.dt_sim)
        N_mpc_steps = int(total_seconds / tp.dt_mpc)

        # Pre-allocate storage (pack-level)
        soc_true = np.zeros(N_sim_steps + 1)
        soh_true = np.zeros(N_sim_steps + 1)
        temp_true = np.zeros(N_sim_steps + 1)
        vrc1_true = np.zeros(N_sim_steps + 1)
        vrc2_true = np.zeros(N_sim_steps + 1)
        vterm_true = np.zeros(N_sim_steps + 1)

        # Estimator states (dt_mpc resolution)
        soc_ekf = np.zeros(N_mpc_steps + 1)
        soh_ekf = np.zeros(N_mpc_steps + 1)
        temp_ekf = np.zeros(N_mpc_steps + 1)
        vrc1_ekf = np.zeros(N_mpc_steps + 1)
        vrc2_ekf = np.zeros(N_mpc_steps + 1)
        soc_mhe = np.zeros(N_mpc_steps + 1)
        soh_mhe = np.zeros(N_mpc_steps + 1)
        temp_mhe = np.zeros(N_mpc_steps + 1)
        vrc1_mhe = np.zeros(N_mpc_steps + 1)
        vrc2_mhe = np.zeros(N_mpc_steps + 1)

        power_applied = np.zeros((N_mpc_steps, 3))

        # Timing instrumentation
        mpc_solve_times = np.zeros(N_mpc_steps)
        est_solve_times = np.zeros(N_mpc_steps)

        # Reference tracking
        soc_ref_at_mpc = np.zeros(N_mpc_steps)
        power_ref_at_mpc = np.zeros((N_mpc_steps, 3))

        # EMS-level reference storage
        ems_p_chg_refs: list[np.ndarray] = []
        ems_p_dis_refs: list[np.ndarray] = []
        ems_p_reg_refs: list[np.ndarray] = []
        ems_soc_refs: list[np.ndarray] = []

        # Cell-level arrays (only when multi-cell)
        n_cells = self.n_cells
        has_cells = n_cells > 1
        if has_cells:
            cell_socs = np.zeros((n_cells, N_sim_steps + 1))
            cell_sohs = np.zeros((n_cells, N_sim_steps + 1))
            cell_temps = np.zeros((n_cells, N_sim_steps + 1))
            cell_vrc1s = np.zeros((n_cells, N_sim_steps + 1))
            cell_vrc2s = np.zeros((n_cells, N_sim_steps + 1))
            balancing_power_log = np.zeros((n_cells, N_mpc_steps))

        # Initialise
        x_true = self.plant.get_state()
        soc_true[0] = x_true[0]
        soh_true[0] = x_true[1]
        temp_true[0] = x_true[2]
        vrc1_true[0] = x_true[3]
        vrc2_true[0] = x_true[4]
        vterm_true[0] = self.plant.get_terminal_voltage()

        if has_cells:
            cs = self.plant.get_cell_states()
            cell_socs[:, 0] = cs[:, 0]
            cell_sohs[:, 0] = cs[:, 1]
            cell_temps[:, 0] = cs[:, 2]
            cell_vrc1s[:, 0] = cs[:, 3]
            cell_vrc2s[:, 0] = cs[:, 4]

        ekf_est = self.ekf.get_estimate()
        soc_ekf[0] = ekf_est[0]
        soh_ekf[0] = ekf_est[1]
        temp_ekf[0] = ekf_est[2]
        vrc1_ekf[0] = ekf_est[3]
        vrc2_ekf[0] = ekf_est[4]

        mhe_est = self.mhe.get_estimate()
        soc_mhe[0] = mhe_est[0]
        soh_mhe[0] = mhe_est[1]
        temp_mhe[0] = mhe_est[2]
        vrc1_mhe[0] = mhe_est[3]
        vrc2_mhe[0] = mhe_est[4]

        # Current control command
        u_current = np.zeros(3)

        # Interpolated MPC-resolution references
        soc_ref_mpc_local = np.full(N_mpc_steps + 1, bp.SOC_init)
        soh_ref_mpc_local = np.full(N_mpc_steps + 1, bp.SOH_init)
        temp_ref_mpc_local = np.full(N_mpc_steps + 1, thp.T_init)
        p_chg_ref_mpc_local = np.zeros(N_mpc_steps)
        p_dis_ref_mpc_local = np.zeros(N_mpc_steps)
        p_reg_ref_mpc_local = np.zeros(N_mpc_steps)

        mpc_ref_base = 0
        mpc_idx = 0
        cum_profit = 0.0
        cum_profit_arr = np.zeros(N_mpc_steps)
        energy_profit_arr = np.zeros(N_mpc_steps)
        reg_profit_arr = np.zeros(N_mpc_steps)
        deg_cost_arr = np.zeros(N_mpc_steps)

        for sim_step in range(N_sim_steps):
            # ===========================================================
            #  EMS update  (every dt_ems = 3 600 s)
            # ===========================================================
            if sim_step % steps_per_ems == 0:
                ems_hour = sim_step // steps_per_ems
                x_est = self.ekf.get_estimate()

                remaining_hours = min(
                    ep.N_ems, energy_scenarios.shape[1] - ems_hour
                )
                if remaining_hours < 1:
                    remaining_hours = 1

                e_scen = energy_scenarios[:, ems_hour : ems_hour + remaining_hours]
                r_scen = reg_scenarios[:, ems_hour : ems_hour + remaining_hours]

                if e_scen.shape[1] < ep.N_ems:
                    pad_w = ep.N_ems - e_scen.shape[1]
                    e_scen = np.pad(e_scen, ((0, 0), (0, pad_w)), mode="edge")
                    r_scen = np.pad(r_scen, ((0, 0), (0, pad_w)), mode="edge")

                logger.info(
                    "EMS solve at t=%d s (hour %d), SOC=%.3f, SOH=%.6f, T=%.1f",
                    sim_step, ems_hour, x_est[0], x_est[1], x_est[2],
                )

                ems_result = self.ems.solve(
                    soc_init=x_est[0],
                    soh_init=x_est[1],
                    t_init=x_est[2],
                    vrc1_init=x_est[3],
                    vrc2_init=x_est[4],
                    energy_scenarios=e_scen,
                    reg_scenarios=r_scen,
                    probabilities=probabilities,
                )

                ems_p_chg_refs.append(ems_result["P_chg_ref"].copy())
                ems_p_dis_refs.append(ems_result["P_dis_ref"].copy())
                ems_p_reg_refs.append(ems_result["P_reg_ref"].copy())
                ems_soc_refs.append(ems_result["SOC_ref"].copy())

                # Blending
                if ems_hour > 0:
                    off = min(mpc_ref_base, len(p_chg_ref_mpc_local) - 1)
                    prev_p_chg_end = float(p_chg_ref_mpc_local[off])
                    prev_p_dis_end = float(p_dis_ref_mpc_local[off])
                    prev_p_reg_end = float(p_reg_ref_mpc_local[off])

                refs = interpolate_ems_to_mpc(ems_result, tp.dt_ems, tp.dt_mpc)
                mpc_ref_base = 0

                soc_ref_mpc_local = refs["SOC_ref_mpc"]
                soh_ref_mpc_local = refs["SOH_ref_mpc"]
                temp_ref_mpc_local = refs["TEMP_ref_mpc"]
                p_chg_ref_mpc_local = refs["P_chg_ref_mpc"]
                p_dis_ref_mpc_local = refs["P_dis_ref_mpc"]
                p_reg_ref_mpc_local = refs["P_reg_ref_mpc"]

                if ems_hour > 0:
                    Nb = min(self.mp.n_blend_steps, len(p_chg_ref_mpc_local))
                    alpha = np.linspace(1.0 / (Nb + 1), 1.0, Nb)
                    p_chg_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_chg_end
                        + alpha * p_chg_ref_mpc_local[:Nb]
                    )
                    p_dis_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_dis_end
                        + alpha * p_dis_ref_mpc_local[:Nb]
                    )
                    p_reg_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_reg_end
                        + alpha * p_reg_ref_mpc_local[:Nb]
                    )

            # ===========================================================
            #  MPC + Estimation update  (every dt_mpc = 60 s)
            # ===========================================================
            if sim_step % steps_per_mpc == 0:
                y_meas = self.plant.get_measurement()  # shape (3,)

                t0_est = time.perf_counter()
                if sim_step > 0:
                    ekf_est = self.ekf.step(u_current, y_meas)
                    mhe_est = self.mhe.step(u_current, y_meas)
                else:
                    ekf_est = self.ekf.get_estimate()
                    mhe_est = self.mhe.get_estimate()
                est_solve_times[mpc_idx] = time.perf_counter() - t0_est

                soc_ekf[mpc_idx] = ekf_est[0]
                soh_ekf[mpc_idx] = ekf_est[1]
                temp_ekf[mpc_idx] = ekf_est[2]
                vrc1_ekf[mpc_idx] = ekf_est[3]
                vrc2_ekf[mpc_idx] = ekf_est[4]
                soc_mhe[mpc_idx] = mhe_est[0]
                soh_mhe[mpc_idx] = mhe_est[1]
                temp_mhe[mpc_idx] = mhe_est[2]
                vrc1_mhe[mpc_idx] = mhe_est[3]
                vrc2_mhe[mpc_idx] = mhe_est[4]

                off = mpc_ref_base
                N_pred = self.mp.N_mpc

                soc_win = self._extract_ref(soc_ref_mpc_local, off, N_pred + 1)
                soh_win = self._extract_ref(soh_ref_mpc_local, off, N_pred + 1)
                temp_win = self._extract_ref(temp_ref_mpc_local, off, N_pred + 1)
                pc_win = self._extract_ref(p_chg_ref_mpc_local, off, N_pred)
                pd_win = self._extract_ref(p_dis_ref_mpc_local, off, N_pred)
                pr_win = self._extract_ref(p_reg_ref_mpc_local, off, N_pred)

                if mpc_idx < N_mpc_steps:
                    soc_ref_at_mpc[mpc_idx] = soc_win[0]
                    power_ref_at_mpc[mpc_idx] = [pc_win[0], pd_win[0], pr_win[0]]

                t0_mpc = time.perf_counter()
                u_current = self.mpc.solve(
                    x_est=ekf_est,      # shape (5,)
                    soc_ref=soc_win,
                    soh_ref=soh_win,
                    temp_ref=temp_win,
                    p_chg_ref=pc_win,
                    p_dis_ref=pd_win,
                    p_reg_ref=pr_win,
                    u_prev=u_current,
                )
                mpc_solve_times[mpc_idx] = time.perf_counter() - t0_mpc

                if mpc_idx < N_mpc_steps:
                    power_applied[mpc_idx] = u_current

                    # Log balancing power
                    if has_cells:
                        balancing_power_log[:, mpc_idx] = self.plant.get_balancing_power()

                    ems_hour_now = sim_step // steps_per_ems
                    if ems_hour_now < energy_scenarios.shape[1]:
                        price_e = float(energy_scenarios[0, ems_hour_now])
                        price_r = float(reg_scenarios[0, ems_hour_now])
                    else:
                        price_e = float(energy_scenarios[0, -1])
                        price_r = float(reg_scenarios[0, -1])

                    dt_h = tp.dt_mpc / 3600.0
                    e_profit = price_e * (u_current[1] - u_current[0]) * dt_h
                    r_profit = price_r * u_current[2] * dt_h
                    d_cost = (
                        ep.degradation_cost
                        * bp.alpha_deg
                        * (u_current[0] + u_current[1] + u_current[2])
                        * tp.dt_mpc
                    )

                    # Regulation delivery penalty: if SOC is too close
                    # to limits, the battery cannot deliver regulation
                    # and pays a penalty instead of earning revenue.
                    soc_now = soc_true[sim_step]
                    if u_current[2] > 0.1:
                        can_deliver = (
                            soc_now > bp.SOC_min + ep.reg_soc_margin
                            and soc_now < bp.SOC_max - ep.reg_soc_margin
                        )
                        if not can_deliver:
                            r_profit = (
                                -ep.reg_penalty_mult
                                * price_r
                                * u_current[2]
                                * dt_h
                            )

                    cum_profit += e_profit + r_profit - d_cost
                    cum_profit_arr[mpc_idx] = cum_profit
                    energy_profit_arr[mpc_idx] = e_profit
                    reg_profit_arr[mpc_idx] = r_profit
                    deg_cost_arr[mpc_idx] = d_cost

                mpc_ref_base += 1
                mpc_idx += 1

            # ===========================================================
            #  Plant step  (every dt_sim = 1 s)
            # ===========================================================
            x_new, _ = self.plant.step(u_current)
            soc_true[sim_step + 1] = x_new[0]
            soh_true[sim_step + 1] = x_new[1]
            temp_true[sim_step + 1] = x_new[2]
            vrc1_true[sim_step + 1] = x_new[3]
            vrc2_true[sim_step + 1] = x_new[4]
            vterm_true[sim_step + 1] = self.plant.get_terminal_voltage()

            # Log cell-level states
            if has_cells:
                cs = self.plant.get_cell_states()
                cell_socs[:, sim_step + 1] = cs[:, 0]
                cell_sohs[:, sim_step + 1] = cs[:, 1]
                cell_temps[:, sim_step + 1] = cs[:, 2]
                cell_vrc1s[:, sim_step + 1] = cs[:, 3]
                cell_vrc2s[:, sim_step + 1] = cs[:, 4]

        # Trim estimator arrays
        soc_ekf = soc_ekf[:mpc_idx]
        soh_ekf = soh_ekf[:mpc_idx]
        temp_ekf = temp_ekf[:mpc_idx]
        vrc1_ekf = vrc1_ekf[:mpc_idx]
        vrc2_ekf = vrc2_ekf[:mpc_idx]
        soc_mhe = soc_mhe[:mpc_idx]
        soh_mhe = soh_mhe[:mpc_idx]
        temp_mhe = temp_mhe[:mpc_idx]
        vrc1_mhe = vrc1_mhe[:mpc_idx]
        vrc2_mhe = vrc2_mhe[:mpc_idx]

        logger.info(
            "Simulation complete: profit=$%.2f, SOH loss=%.4f%%, T_max=%.1f degC, "
            "V_term range=[%.1f, %.1f] V",
            cum_profit,
            (soh_true[0] - soh_true[-1]) * 100,
            np.max(temp_true),
            np.min(vterm_true[1:]),
            np.max(vterm_true[1:]),
        )

        result = {
            # Plant (dt_sim resolution)
            "time_sim": np.arange(N_sim_steps + 1) * tp.dt_sim,
            "soc_true": soc_true,
            "soh_true": soh_true,
            "temp_true": temp_true,
            "vrc1_true": vrc1_true,
            "vrc2_true": vrc2_true,
            "vterm_true": vterm_true,
            # Estimators (dt_mpc resolution)
            "time_mpc": np.arange(len(soc_ekf)) * tp.dt_mpc,
            "soc_ekf": soc_ekf,
            "soh_ekf": soh_ekf,
            "temp_ekf": temp_ekf,
            "vrc1_ekf": vrc1_ekf,
            "vrc2_ekf": vrc2_ekf,
            "soc_mhe": soc_mhe,
            "soh_mhe": soh_mhe,
            "temp_mhe": temp_mhe,
            "vrc1_mhe": vrc1_mhe,
            "vrc2_mhe": vrc2_mhe,
            # Applied power
            "power_applied": power_applied[:mpc_idx],
            # Profit
            "cumulative_profit": cum_profit_arr[:mpc_idx],
            "energy_profit": energy_profit_arr[:mpc_idx],
            "reg_profit": reg_profit_arr[:mpc_idx],
            "deg_cost": deg_cost_arr[:mpc_idx],
            "total_profit": cum_profit,
            "soh_degradation": soh_true[0] - soh_true[-1],
            # EMS references
            "ems_p_chg_refs": ems_p_chg_refs,
            "ems_p_dis_refs": ems_p_dis_refs,
            "ems_p_reg_refs": ems_p_reg_refs,
            "ems_soc_refs": ems_soc_refs,
            # Prices
            "prices_energy": energy_scenarios[0],
            "prices_reg": reg_scenarios[0],
            # Timing
            "mpc_solve_times": mpc_solve_times[:mpc_idx],
            "est_solve_times": est_solve_times[:mpc_idx],
            # Reference tracking
            "soc_ref_at_mpc": soc_ref_at_mpc[:mpc_idx],
            "power_ref_at_mpc": power_ref_at_mpc[:mpc_idx],
        }

        # Cell-level data
        if has_cells:
            result["cell_socs"] = cell_socs
            result["cell_sohs"] = cell_sohs
            result["cell_temps"] = cell_temps
            result["cell_vrc1s"] = cell_vrc1s
            result["cell_vrc2s"] = cell_vrc2s
            result["balancing_power"] = balancing_power_log[:, :mpc_idx]
            result["soc_imbalance"] = np.max(cell_socs, axis=0) - np.min(cell_socs, axis=0)
            result["n_cells"] = n_cells

        return result

    @staticmethod
    def _extract_ref(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
        """Extract a window from *arr* starting at *offset*, padding if needed."""
        end = offset + length
        if end <= len(arr):
            return arr[offset:end].copy()
        available = arr[offset:].copy()
        pad_val = available[-1] if len(available) > 0 else 0.0
        return np.concatenate([available, np.full(length - len(available), pad_val)])
