"""Configuration parameters for the hierarchical BESS control platform.

v4_electrical_rc_model: adds ElectricalParams for 2RC equivalent circuit
with NMC OCV polynomial.  Extends state vector from 3 to 5:
    x = [SOC, SOH, T, V_rc1, V_rc2]

All physical units are SI-consistent and explicitly documented:
  - Energy:  kWh
  - Power:   kW
  - Time:    seconds (except sim_hours)
  - Price:   $/kWh  (energy),  $/kW/h  (regulation)
  - SOC/SOH: dimensionless  [0, 1]
  - Temperature: degC
  - Voltage: V
  - Resistance: Ohm
  - Capacitance: F
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryParams:
    """Physical parameters of the battery energy storage system with degradation.

    Degradation model (v2: thermally coupled via Arrhenius factor)
    ---------------------------------------------------------------
    dSOH/dt = -alpha_deg * kappa(T) * (P_chg + P_dis + |P_reg|)

    kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))

    At T_ref (25 degC): kappa = 1.0 (identical to v1 baseline).
    At 45 degC: kappa ~ 1.66 (66 % faster degradation).
    """

    E_nom_kwh: float = 200.0           # Nominal energy capacity  [kWh]
    P_max_kw: float = 100.0            # Maximum charge / discharge power  [kW]
    SOC_min: float = 0.10              # Minimum allowable state of charge  [-]
    SOC_max: float = 0.90              # Maximum allowable state of charge  [-]
    SOC_init: float = 0.50             # Initial state of charge  [-]
    SOH_init: float = 1.00             # Initial state of health  [-]
    SOC_terminal: float = 0.50         # Terminal SOC target for EMS  [-]
    eta_charge: float = 0.95           # Charging efficiency  [-]
    eta_discharge: float = 0.95        # Discharging efficiency  [-]
    alpha_deg: float = 2.78e-9         # Degradation rate  [1/(kW*s)]


@dataclass(frozen=True)
class ThermalParams:
    """Lumped-parameter thermal model for a 200 kWh / 100 kW utility-scale BESS.

    Thermal dynamics
    ----------------
    dT/dt = (I^2 * R_total_dc - h_cool * (T - T_ambient)) / C_thermal   [degC/s]

    In v4, R_internal is superseded by the sum R0 + R1 + R2 from ElectricalParams
    for Joule heating.  Kept here for backward compatibility.

    Current derived from quadratic solve (v4):
        R0 * I^2 - V_oc_eff * I + P_net * 1000 = 0

    Arrhenius coupling
    ------------------
    kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))

    At 25 degC: kappa = 1.00.  At 35 degC: kappa = 1.30.  At 45 degC: kappa = 1.66.
    """

    R_internal: float = 0.010          # Total DC resistance  [Ohm]  (= R0+R1+R2 in v4)
    C_thermal: float = 150_000.0       # Thermal mass (heat capacity)  [J/K]
    h_cool: float = 50.0               # Cooling coefficient  [W/K]
    T_ambient: float = 25.0            # Ambient temperature  [degC]
    T_init: float = 25.0               # Initial cell temperature  [degC]
    T_max: float = 45.0                # Maximum allowable temperature  [degC]
    T_min: float = 5.0                 # Minimum allowable temperature  [degC]
    V_nominal: float = 800.0           # Nominal pack voltage  [V]
    E_a: float = 20_000.0             # Arrhenius activation energy  [J/mol]
    R_gas: float = 8.314               # Universal gas constant  [J/(mol*K)]
    T_ref: float = 25.0                # Reference temperature for Arrhenius  [degC]


@dataclass(frozen=True)
class ElectricalParams:
    """2RC equivalent circuit model parameters (v4).

    Circuit model
    -------------
    V_term = OCV(SOC) - V_rc1 - V_rc2 - I * R0

    where I > 0 for discharge, I < 0 for charge.

    RC dynamics
    -----------
    dV_rc1/dt = -V_rc1 / tau_1 + I / C1       (fast transient,  tau_1 = 10 s)
    dV_rc2/dt = -V_rc2 / tau_2 + I / C2       (slow diffusion,  tau_2 = 400 s)

    Total DC resistance: R0 + R1 + R2 = 0.010 Ohm  (matches v3 R_internal).

    OCV polynomial (NMC, single cell)
    ---------------------------------
    OCV_cell(SOC) = sum(ocv_coeffs[k] * SOC^k, k=0..7)
    Range: ~3.0 V (SOC=0) to ~4.19 V (SOC=1), monotonically increasing.

    Pack scaling: each module contains n_series_cells single cells in series.
    OCV_module(SOC) = n_series_cells * OCV_cell(SOC).
    """

    # Series resistance  [Ohm, pack-level]
    R0: float = 0.005

    # First RC pair (charge-transfer / double-layer)
    R1: float = 0.003                  # [Ohm, pack-level]
    tau_1: float = 10.0                # Time constant  [s]

    # Second RC pair (solid-state diffusion)
    R2: float = 0.002                  # [Ohm, pack-level]
    tau_2: float = 400.0               # Time constant  [s]

    # Single cells in series per module  (V_module_nom / V_cell_nom)
    n_series_cells: int = 54           # 200 V / 3.7 V ≈ 54

    # Cell voltage limits  [V, single cell]
    V_min_cell: float = 2.8            # NMC lower cutoff
    V_max_cell: float = 4.25           # NMC upper cutoff

    # Initial RC voltages  [V, pack-level]
    V_rc1_init: float = 0.0
    V_rc2_init: float = 0.0

    # NMC single-cell OCV polynomial coefficients  [V]
    # OCV(SOC) = a0 + a1*SOC + a2*SOC^2 + ... + a7*SOC^7
    # Fitted to NMC data: monotonic, 3.0 V (SOC=0) to 4.19 V (SOC=1).
    ocv_coeffs: tuple[float, ...] = (
        3.001186,
        7.405750,
        -44.120594,
        139.808956,
        -237.151740,
        209.935290,
        -84.647602,
        9.959177,
    )

    # Voltage measurement noise  [V, pack-level]
    sigma_v_meas: float = 1.0

    @property
    def C1(self) -> float:
        """First RC capacitance [F]: C1 = tau_1 / R1."""
        return self.tau_1 / self.R1

    @property
    def C2(self) -> float:
        """Second RC capacitance [F]: C2 = tau_2 / R2."""
        return self.tau_2 / self.R2

    @property
    def R_total_dc(self) -> float:
        """Total DC resistance [Ohm]: R0 + R1 + R2."""
        return self.R0 + self.R1 + self.R2

    @property
    def V_min_pack(self) -> float:
        """Minimum pack voltage [V]."""
        return self.V_min_cell * self.n_series_cells * 4  # 4 modules

    @property
    def V_max_pack(self) -> float:
        """Maximum pack voltage [V]."""
        return self.V_max_cell * self.n_series_cells * 4  # 4 modules


@dataclass(frozen=True)
class TimeParams:
    """Time discretisation for multi-rate control.

    Every time quantity is in **seconds** except ``sim_hours``.
    """

    dt_ems: float = 3600.0             # EMS sampling period  [s]
    dt_mpc: float = 60.0               # MPC sampling period  [s]
    dt_estimator: float = 60.0         # Estimator sampling period  [s]  (= dt_mpc)
    dt_sim: float = 5.0                # Plant simulation step  [s]  (RK4-stable for tau_min=10s)
    sim_hours: float = 24.0            # Total simulation duration  [hours]


@dataclass(frozen=True)
class EMSParams:
    """Parameters for the stochastic Energy Management System optimizer."""

    N_ems: int = 24                    # Planning horizon  [hours / steps at dt_ems]
    Nc_ems: int = 24                   # Control horizon  (= N_ems)
    n_scenarios: int = 5               # Number of price scenarios
    regulation_fraction: float = 0.3   # Max fraction of P_max for regulation  [-]
    degradation_cost: float = 50.0     # Cost of SOH loss  [$/unit SOH lost]
    reg_soc_margin: float = 0.05      # SOC margin for regulation delivery  [-]
    reg_penalty_mult: float = 2.0     # Penalty multiplier for non-delivery  [-]
    terminal_soc_weight: float = 1e4   # Terminal SOC deviation penalty
    terminal_soh_weight: float = 1e4   # Terminal SOH deviation penalty


@dataclass(frozen=True)
class MPCParams:
    """Tuning parameters for the nonlinear tracking MPC.

    v4: adds soft voltage constraints.
    """

    N_mpc: int = 60                    # Prediction horizon  [steps at dt_mpc]
    Nc_mpc: int = 20                   # Control horizon  [steps at dt_mpc]
    Q_soc: float = 1e4                 # SOC tracking weight
    Q_soh: float = 1e2                 # SOH tracking weight
    Q_temp: float = 1e2                # Temperature tracking weight
    R_power: float = 1.0               # Power reference tracking weight (per input)
    R_delta: float = 10.0              # Control rate-of-change penalty
    Q_terminal: float = 1e5            # Terminal SOC penalty
    slack_penalty: float = 1e6         # Soft SOC constraint violation penalty
    slack_penalty_temp: float = 1e5    # Soft temperature constraint penalty
    slack_penalty_volt: float = 1e5    # Soft voltage constraint penalty  [v4]
    n_blend_steps: int = 5             # EMS boundary reference smoothing  [MPC steps]


@dataclass(frozen=True)
class EKFParams:
    """Tuning parameters for the Extended Kalman Filter.

    v4: 5-state, 3-measurement.  Adds V_rc1, V_rc2 process noise and
    terminal voltage measurement noise.
    """

    # Process noise covariance  Q  (5x5 diagonal in v4)
    q_soc: float = 1e-6               # SOC process noise variance
    q_soh: float = 1e-12              # SOH process noise variance (extremely slow dynamics)
    q_temp: float = 1e-4               # Temperature process noise variance  [degC^2]
    q_vrc1: float = 1e-4               # V_rc1 process noise variance  [V^2]  [v4]
    q_vrc2: float = 1e-5               # V_rc2 process noise variance  [V^2]  [v4]

    # Measurement noise covariance  R  (3x3 diagonal in v4)
    r_soc_meas: float = 1e-4          # SOC measurement noise variance (sigma ~ 0.01)
    r_temp_meas: float = 0.25         # Temperature measurement noise variance  [degC^2]
    r_vterm_meas: float = 4.0         # V_term measurement noise variance  [V^2, sigma=2V]  [v4]

    # Initial state error covariance  P_0  (5x5 diagonal in v4)
    p0_soc: float = 1e-3              # Initial SOC uncertainty
    p0_soh: float = 1e-2              # Initial SOH uncertainty (larger — unknown)
    p0_temp: float = 1.0              # Initial temperature uncertainty  [degC^2]
    p0_vrc1: float = 1.0              # Initial V_rc1 uncertainty  [V^2]  [v4]
    p0_vrc2: float = 1.0              # Initial V_rc2 uncertainty  [V^2]  [v4]


@dataclass(frozen=True)
class MHEParams:
    """Tuning parameters for Moving Horizon Estimation.

    v4: adds V_rc1/V_rc2 arrival cost, V_term measurement weight,
    and V_rc process noise weights.
    """

    N_mhe: int = 30                    # Estimation window  [steps at dt_estimator]

    # Arrival cost weights  (inverse of prior covariance)
    arrival_soc: float = 1e3           # Weight on SOC arrival cost
    arrival_soh: float = 1e4           # Weight on SOH arrival cost (high — SOH barely observable)
    arrival_temp: float = 1e2          # Weight on temperature arrival cost
    arrival_vrc1: float = 1e2          # Weight on V_rc1 arrival cost  [v4]
    arrival_vrc2: float = 1e2          # Weight on V_rc2 arrival cost  [v4]

    # Stage cost weights
    w_soc_meas: float = 1e4            # SOC measurement residual weight
    w_temp_meas: float = 1e3           # Temperature measurement residual weight
    w_vterm_meas: float = 1e3          # V_term measurement residual weight  [v4]
    w_process_soc: float = 1e2         # Process disturbance weight (SOC)
    w_process_soh: float = 1e8         # Process disturbance weight (SOH, very high)
    w_process_temp: float = 1e3        # Temperature process disturbance weight
    w_process_vrc1: float = 1e3        # V_rc1 process disturbance weight  [v4]
    w_process_vrc2: float = 1e4        # V_rc2 process disturbance weight  [v4]


@dataclass(frozen=True)
class PackParams:
    """Multi-cell battery pack configuration.

    Models a pack of N cells in series, each with slightly different
    physical parameters to simulate manufacturing variation.  The active
    balancing controller equalises cell SOCs via proportional control.

    Per-cell scaling (for N cells in series)
    -----------------------------------------
    E_nom_cell  = E_nom_pack / N * (1 +/- capacity_spread)
    P_max_cell  = P_max_pack / N
    R_cell      = R_pack / N * (1 +/- resistance_spread)
    C_th_cell   = C_th_pack / N
    h_cool_cell = h_cool_pack / N
    V_cell      = V_pack / N
    alpha_cell  = alpha_pack * (1 +/- degradation_spread)
    """

    n_cells: int = 4                          # Number of cells in pack
    capacity_spread: float = 0.03             # +/-3% E_nom variation  [-]
    resistance_spread: float = 0.08           # +/-8% R_internal variation  [-]
    degradation_spread: float = 0.05          # +/-5% alpha_deg variation  [-]
    initial_soc_spread: float = 0.02          # +/-2% initial SOC variation  [-]
    balancing_enabled: bool = True             # Enable active cell balancing
    balancing_gain: float = 50.0              # Proportional gain  [kW / unit SOC error]
    max_balancing_power: float = 1.0          # Max balancing power per cell  [kW]
    seed: int = 123                           # RNG seed for cell variation
