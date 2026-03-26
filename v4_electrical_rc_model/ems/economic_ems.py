"""Stochastic Energy Management System (EMS) with regulation market.

v4_electrical_rc_model: uses 3-state dynamics (SOC, SOH, T) at the EMS
layer.  V_rc1/V_rc2 are omitted because both RC time constants (10 s,
400 s) are far shorter than the EMS time step (3 600 s), so the
transients fully decay within each planning interval.  This is
standard hierarchical control practice — the planning layer uses a
coarse model, the fast controller (MPC) uses the detailed model.

Solves a scenario-based NLP every ``dt_ems`` (3 600 s) over a 24-hour
rolling horizon using CasADi Opti / IPOPT.

Decision variables per scenario *s*, per time step *k*
-------------------------------------------------------
  P_chg[s, k], P_dis[s, k], P_reg[s, k]   :  power commands  [kW]
  SOC[s, k],   SOH[s, k],   TEMP[s, k]    :  state trajectories

Objective
---------
  Maximise   E_s [ energy_revenue + regulation_revenue
                   - degradation_cost - regulation_delivery_penalty ]

Regulation delivery feasibility
-------------------------------
  When SOC is within ``reg_soc_margin`` of its limits, the battery
  cannot reliably deliver regulation services (insufficient headroom
  for up/down response).  Linear constraints force P_reg to zero as
  SOC approaches limits, preventing infeasible capacity commitments.

Non-anticipativity: first-stage decisions agree across all scenarios.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    ThermalParams,
    TimeParams,
)
from models.battery_model import build_casadi_rk4_integrator_3state

logger = logging.getLogger(__name__)


class EconomicEMS:
    """Stochastic economic EMS with energy arbitrage and regulation revenue.

    Uses 3-state dynamics (SOC, SOH, T) — V_rc states are irrelevant at
    hourly resolution since tau_1=10s, tau_2=400s << dt_ems=3600s.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    ep  : EMSParams
    thp : ThermalParams
    elp : ElectricalParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        thp: ThermalParams,
        elp: ElectricalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.thp = thp
        self.elp = elp

        # 3-state RK4 integrator at EMS time step (3 600 s)
        # No sub-stepping needed — all dynamics are slow at this timescale
        self._F_ems = build_casadi_rk4_integrator_3state(
            bp, thp, elp, tp.dt_ems,
        )

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
        vrc1_init: float = 0.0,
        vrc2_init: float = 0.0,
    ) -> dict:
        """Solve the stochastic EMS optimisation.

        Parameters
        ----------
        soc_init : float
        soh_init : float
        t_init   : float   Temperature [degC]
        energy_scenarios : ndarray, shape (n_scenarios, N_hours)
        reg_scenarios : ndarray, shape (n_scenarios, N_hours)
        probabilities : ndarray, shape (n_scenarios,)
        vrc1_init : float   Ignored (kept for interface compatibility).
        vrc2_init : float   Ignored (kept for interface compatibility).

        Returns
        -------
        dict with keys:
            P_chg_ref, P_dis_ref, P_reg_ref : ndarray (N,)
            SOC_ref, SOH_ref, TEMP_ref : ndarray (N+1,)
            VRC1_ref, VRC2_ref : ndarray (N+1,)   (zeros — V_rc not modeled)
            expected_profit : float
        """
        bp = self.bp
        ep = self.ep
        thp = self.thp
        N = min(ep.N_ems, energy_scenarios.shape[1])
        S = len(probabilities)

        # Clip initial state to strictly feasible region
        soc_init = float(np.clip(soc_init, bp.SOC_min + 0.001, bp.SOC_max - 0.001))
        soh_init = float(np.clip(soh_init, 0.51, 1.0))
        t_init = float(np.clip(t_init, thp.T_min + 0.1, thp.T_max - 0.1))

        opti = ca.Opti()

        # ---- Decision variables (per scenario) ----
        P_chg: list[ca.MX] = []
        P_dis: list[ca.MX] = []
        P_reg: list[ca.MX] = []
        SOC: list[ca.MX] = []
        SOH: list[ca.MX] = []
        TEMP: list[ca.MX] = []
        eps_soc: list[ca.MX] = []

        for _ in range(S):
            P_chg.append(opti.variable(N))
            P_dis.append(opti.variable(N))
            P_reg.append(opti.variable(N))
            SOC.append(opti.variable(N + 1))
            SOH.append(opti.variable(N + 1))
            TEMP.append(opti.variable(N + 1))
            eps_soc.append(opti.variable(N + 1))

        total_obj = 0.0

        for s in range(S):
            # Initial conditions
            opti.subject_to(SOC[s][0] == soc_init)
            opti.subject_to(SOH[s][0] == soh_init)
            opti.subject_to(TEMP[s][0] == t_init)

            scenario_profit = 0.0

            for k in range(N):
                # Dynamics (3-state)
                x_k = ca.vertcat(SOC[s][k], SOH[s][k], TEMP[s][k])
                u_k = ca.vertcat(P_chg[s][k], P_dis[s][k], P_reg[s][k])
                x_next = self._F_ems(x_k, u_k)

                opti.subject_to(SOC[s][k + 1] == x_next[0])
                opti.subject_to(SOH[s][k + 1] == x_next[1])
                opti.subject_to(TEMP[s][k + 1] == x_next[2])

                # Energy arbitrage revenue  [$ for this hour]
                dt_hours = self.tp.dt_ems / 3600.0
                energy_rev = float(energy_scenarios[s, k]) * (
                    P_dis[s][k] - P_chg[s][k]
                ) * dt_hours

                # Regulation capacity payment  [$ for this hour]
                reg_rev = float(reg_scenarios[s, k]) * P_reg[s][k] * dt_hours

                # Degradation cost (Arrhenius effect is embedded in dynamics)
                deg_cost = ep.degradation_cost * bp.alpha_deg * (
                    P_chg[s][k] + P_dis[s][k] + P_reg[s][k]
                ) * self.tp.dt_ems

                scenario_profit += energy_rev + reg_rev - deg_cost

            # Terminal penalties
            scenario_profit -= ep.terminal_soc_weight * (
                SOC[s][N] - bp.SOC_terminal
            ) ** 2
            scenario_profit -= ep.terminal_soh_weight * (
                SOH[s][N] - soh_init
            ) ** 2

            # Soft SOC constraint penalty
            for k in range(N + 1):
                scenario_profit -= 1e5 * eps_soc[s][k] ** 2

            # ---- Bounds for this scenario ----
            opti.subject_to(eps_soc[s] >= 0)
            for k in range(N + 1):
                opti.subject_to(SOC[s][k] >= bp.SOC_min - eps_soc[s][k])
                opti.subject_to(SOC[s][k] <= bp.SOC_max + eps_soc[s][k])
            opti.subject_to(opti.bounded(0.5, SOH[s], 1.001))
            opti.subject_to(opti.bounded(thp.T_min, TEMP[s], thp.T_max))
            opti.subject_to(opti.bounded(0.0, P_chg[s], bp.P_max_kw))
            opti.subject_to(opti.bounded(0.0, P_dis[s], bp.P_max_kw))
            opti.subject_to(
                opti.bounded(0.0, P_reg[s], bp.P_max_kw * ep.regulation_fraction)
            )

            # Power budget: charge + reg <= P_max,  discharge + reg <= P_max
            for k in range(N):
                opti.subject_to(P_chg[s][k] + P_reg[s][k] <= bp.P_max_kw)
                opti.subject_to(P_dis[s][k] + P_reg[s][k] <= bp.P_max_kw)

            # Regulation delivery feasibility: P_reg must reduce when
            # SOC approaches limits (insufficient headroom for
            # symmetric up/down response).  Linear in (P_reg, SOC).
            P_reg_max = bp.P_max_kw * ep.regulation_fraction
            for k in range(N):
                opti.subject_to(
                    P_reg[s][k] * ep.reg_soc_margin
                    <= P_reg_max * (SOC[s][k] - bp.SOC_min)
                )
                opti.subject_to(
                    P_reg[s][k] * ep.reg_soc_margin
                    <= P_reg_max * (bp.SOC_max - SOC[s][k])
                )

            # Accumulate expected cost (minimise negative profit)
            total_obj += float(probabilities[s]) * (-scenario_profit)

        # ---- Non-anticipativity (first-stage) ----
        for s in range(1, S):
            opti.subject_to(P_chg[s][0] == P_chg[0][0])
            opti.subject_to(P_dis[s][0] == P_dis[0][0])
            opti.subject_to(P_reg[s][0] == P_reg[0][0])

        opti.minimize(total_obj)

        # ---- Initial guesses ----
        for s in range(S):
            opti.set_initial(SOC[s], np.linspace(soc_init, bp.SOC_terminal, N + 1))
            opti.set_initial(SOH[s], soh_init)
            opti.set_initial(TEMP[s], t_init)
            opti.set_initial(P_chg[s], 0.0)
            opti.set_initial(P_dis[s], 0.0)
            opti.set_initial(P_reg[s], 0.0)
            opti.set_initial(eps_soc[s], 0.0)

        # ---- Solver options ----
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.linear_solver": "mumps",
        }
        opti.solver("ipopt", opts)

        # ---- Solve ----
        try:
            sol = opti.solve()
        except RuntimeError as exc:
            logger.error("EMS solver failed: %s", exc)
            return self._fallback_result(N, soc_init, soh_init, t_init)

        # ---- Extract probability-weighted references ----
        p_chg_ref = np.zeros(N)
        p_dis_ref = np.zeros(N)
        p_reg_ref = np.zeros(N)
        soc_ref = np.zeros(N + 1)
        soh_ref = np.zeros(N + 1)
        temp_ref = np.zeros(N + 1)

        for s in range(S):
            w = float(probabilities[s])
            p_chg_ref += w * np.array(sol.value(P_chg[s])).flatten()
            p_dis_ref += w * np.array(sol.value(P_dis[s])).flatten()
            p_reg_ref += w * np.array(sol.value(P_reg[s])).flatten()
            soc_ref += w * np.array(sol.value(SOC[s])).flatten()
            soh_ref += w * np.array(sol.value(SOH[s])).flatten()
            temp_ref += w * np.array(sol.value(TEMP[s])).flatten()

        expected_profit = float(-sol.value(total_obj))

        logger.info(
            "EMS solved: expected profit = $%.2f  |  "
            "SOC [%.3f -> %.3f]  |  SOH [%.6f -> %.6f]  |  T [%.1f -> %.1f]",
            expected_profit,
            soc_ref[0], soc_ref[-1],
            soh_ref[0], soh_ref[-1],
            temp_ref[0], temp_ref[-1],
        )

        return {
            "P_chg_ref": p_chg_ref,
            "P_dis_ref": p_dis_ref,
            "P_reg_ref": p_reg_ref,
            "SOC_ref": soc_ref,
            "SOH_ref": soh_ref,
            "TEMP_ref": temp_ref,
            "VRC1_ref": np.zeros(N + 1),   # V_rc not modeled at EMS layer
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": expected_profit,
        }

    # ------------------------------------------------------------------
    #  Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_result(
        N: int, soc_init: float, soh_init: float, t_init: float,
    ) -> dict:
        """Return zero-power references when the solver fails."""
        return {
            "P_chg_ref": np.zeros(N),
            "P_dis_ref": np.zeros(N),
            "P_reg_ref": np.zeros(N),
            "SOC_ref": np.full(N + 1, soc_init),
            "SOH_ref": np.full(N + 1, soh_init),
            "TEMP_ref": np.full(N + 1, t_init),
            "VRC1_ref": np.zeros(N + 1),
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": 0.0,
        }
