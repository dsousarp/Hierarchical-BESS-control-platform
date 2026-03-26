# v4_electrical_rc_model — 2RC Equivalent Circuit with NMC OCV Polynomial

Extends the 3-state thermal model to a **5-state model** by adding two RC voltage states:

    x = [SOC, SOH, T, V_rc1, V_rc2]

Terminal voltage `V_term = OCV(SOC) - V_rc1 - V_rc2 - I*R0` is now explicitly modeled and measured, providing a new observation channel for state estimation via voltage feedback.

## What Changed from v3

| Component | v3_pack_model | v4_electrical_rc_model |
|-----------|--------------|----------------------|
| **State vector** | 3 states: `[SOC, SOH, T]` | 5 states: `[SOC, SOH, T, V_rc1, V_rc2]` |
| **Measurements** | 2: `[SOC, T]` | 3: `[SOC, T, V_term]` |
| **Electrical model** | Simple `I = P/V_nom` | 2RC circuit with OCV polynomial + quadratic current solve |
| **OCV** | Not modeled | 7th-order NMC polynomial: 3.0 V (SOC=0) to 4.19 V (SOC=1) |
| **Voltage constraints** | None | Soft V_term constraints in MPC (V_min_pack, V_max_pack) |
| **EKF** | 3-state, 2-measurement | 5-state, 3-measurement (nonlinear H via OCV) |
| **MHE** | 3-state, 2-measurement | 5-state, 3-measurement with V_rc process weights |
| **MPC** | 3-state | 3-state (V_rc omitted for tractability — see design note) |
| **Plant** | `BatteryPack` with simple current | `BatteryPack` with 2RC dynamics per cell |
| **Visualization** | 4x2 layout | 2x2 layout: SOC, voltage, SOH+RC, power+price |
| **Simulation dt** | 1 s | 5 s (RK4-stable for tau_min=10 s) |
| **Stress tests** | 10 tests | 14 tests (adds OCV, RC step response, solver robustness, voltage limits) |

**Unchanged**: Pack architecture (4 cells, active balancing), price generation.

**EMS enhancement**: Regulation delivery feasibility constraints — P_reg is automatically reduced when SOC approaches limits (insufficient headroom for symmetric up/down response). Prevents infeasible capacity commitments and corresponding non-delivery penalties.

## 2RC Equivalent Circuit Model

```
    I (>0 discharge)
    ─────┬───[R0]───┬───[R1]───┬───[R2]───┬─────
         │          │    ║     │    ║     │
         │          │   [C1]   │   [C2]   │
       V_term       │    ║     │    ║     │
         │          └──────────┘──────────┘
    ─────┴─────────────────────────────────┴─────
                         OCV(SOC)

V_term = OCV(SOC) - V_rc1 - V_rc2 - I * R0

RC dynamics:
  dV_rc1/dt = -V_rc1 / tau_1  +  I / C1    (fast transient,  tau_1 = 10 s)
  dV_rc2/dt = -V_rc2 / tau_2  +  I / C2    (slow diffusion,  tau_2 = 400 s)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `R0` | 0.005 Ohm | Series resistance (pack-level) |
| `R1` | 0.003 Ohm | Charge-transfer resistance |
| `tau_1` | 10 s | Fast time constant (C1 = 3333 F) |
| `R2` | 0.002 Ohm | Diffusion resistance |
| `tau_2` | 400 s | Slow time constant (C2 = 200000 F) |
| `R_total` | 0.010 Ohm | R0 + R1 + R2 (matches v3 R_internal) |

## NMC OCV Polynomial

7th-order polynomial fitted to NMC cell data, evaluated via Horner's method:

```
OCV_cell(SOC) = a0 + a1*SOC + a2*SOC^2 + ... + a7*SOC^7
```

Range: ~3.0 V (SOC=0) to ~4.19 V (SOC=1), monotonically increasing.

Pack-level OCV: `OCV_pack(SOC) = n_series_cells * n_modules * OCV_cell(SOC)`, where `n_series_cells=54` and `n_modules=4` (216 cells total, ~800 V nominal).

## Current Computation

Current is derived from a quadratic equation (power = voltage x current):

```
R0 * I^2 - V_oc_eff * I + P_net * 1000 = 0

where V_oc_eff = OCV(SOC) - V_rc1 - V_rc2
      P_net = P_dis - P_chg  [kW]
```

The physically meaningful root (smaller |I|) is selected. Falls back to `I = P_net*1000 / V_oc_eff` when the discriminant is negative.

## Hierarchical Estimation-Control Separation

A key design decision in v4:

- **EKF/MHE** use the full **5-state** model — the voltage measurement provides additional SOC observability through the OCV curve slope
- **MPC/EMS** use a **3-state** model (SOC, SOH, T) — V_rc dynamics are omitted because their effect on control is negligible (V_rc1_max = I*R1 = 0.375 V at rated current, ~0.05% of pack voltage)

This is standard hierarchical estimation-control separation: the estimator uses all available measurements for accuracy, while the controller uses a reduced model for tractability.

## Pack Voltage Limits

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `V_min_pack` | 604.8 V | 2.8 V/cell x 54 cells x 4 modules |
| `V_max_pack` | 918.0 V | 4.25 V/cell x 54 cells x 4 modules |

Enforced as soft constraints in MPC with penalty weight `slack_penalty_volt = 1e5`.

## Module Structure

```
v4_electrical_rc_model/
├── main.py                   # Entry point: VERSION_TAG="v4_electrical_rc_model"
├── config/
│   └── parameters.py         # All v3 params + ElectricalParams (2RC, OCV, voltage limits)
├── models/
│   └── battery_model.py      # 5-state CasADi + numpy dynamics, OCV polynomial, quadratic solver
├── ems/
│   └── economic_ems.py       # 3-state hourly planning + regulation delivery constraints
├── mpc/
│   └── tracking_mpc.py       # 3-state MPC with OCV-based current + soft voltage constraints
├── estimation/
│   ├── ekf.py                # 5-state, 3-measurement EKF (nonlinear H via OCV)
│   └── mhe.py                # 5-state, 3-measurement MHE with V_rc process weights
├── simulation/
│   └── simulator.py          # Multi-rate coordinator with 2RC pack plant
├── visualization/
│   └── plot_results.py       # 2x2 layout: SOC, voltage, SOH+RC, power+price
├── data/
│   └── price_generator.py    # Unchanged from v3
└── stress_test.py            # 14-test stress suite with electrical-specific tests
```

## Running

```bash
# From repository root
uv run python v4_electrical_rc_model/main.py

# Run stress tests
uv run python v4_electrical_rc_model/stress_test.py

# Compare with v1, v2, v3
uv run python -m comparison.process_results
uv run python -m comparison.compare_versions
```

## Stress Tests

14 tests covering all v3 tests plus 4 new electrical-specific tests:

| # | Test | Category |
|---|------|----------|
| 1 | Max power continuous cycling (100 kW, 4h) | Thermal + electrical |
| 2 | High ambient temperature (40 degC) | Arrhenius coupling |
| 3 | SOC boundary saturation | SOC limits |
| 4 | Rapid power reversals (60s cycle) | Transient response |
| 5 | Thermal decay to ambient | Thermal dynamics |
| 6 | EKF convergence from bad initial estimate (5-state) | Estimation |
| 7 | MPC temperature constraint enforcement | Control safety |
| 8 | Cell imbalance recovery (large initial SOC spread) | Pack balancing |
| 9 | Balancing saturation (extreme cell variation) | Pack balancing |
| 10 | Weakest-cell degradation | Pack degradation |
| 11 | OCV monotonicity verification | **New** — electrical |
| 12 | RC step response (tau_1 and tau_2 settling) | **New** — electrical |
| 13 | Quadratic solver robustness (extreme inputs) | **New** — electrical |
| 14 | Voltage at SOC extremes (V_term within pack limits) | **New** — electrical |

Results plotted to `results/v4_electrical_rc_model_stress_tests.png`.
