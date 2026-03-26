# Energy Storage Optimization Platform

A production-grade Python platform for **optimal scheduling, real-time dispatch, and state estimation** of grid-connected battery energy storage systems (BESS). Simultaneously participates in **energy arbitrage** and **frequency regulation** markets while actively managing battery degradation.

Built with the same hierarchical control architecture deployed in commercial utility-scale installations. Evolves through **incremental, gated upgrades** from baseline to industry-grade digital twin.

## What It Solves

Every BESS operator faces the same question: *Given uncertain market prices, how should I charge, discharge, and bid regulation capacity over the next 24 hours -- and how do I execute that plan in real time while protecting the battery?*

This platform answers it end-to-end:

- **Economic scheduling** -- stochastic 24-hour optimization across 5 price scenarios, balancing arbitrage revenue, regulation capacity payments, and degradation costs
- **Real-time dispatch** -- nonlinear model predictive control executing the schedule at 1-minute resolution, enforcing thermal and SOC constraints
- **State estimation** -- EKF and MHE estimators reconstructing battery internals (SOC, SOH, temperature, voltage) from noisy sensors
- **Multi-cell pack modeling** -- per-cell parameter variation with active balancing, weakest-link SOH tracking

## Architecture

```
 Stochastic Prices ──► EMS (hourly)  ──► MPC (1 min)  ──► Battery Plant (5s)
                       24h horizon        60-step            4-cell pack
                       5 scenarios        warm-started       2RC circuit
                                          ◄── EKF/MHE ◄──  voltage + SOC + T
                                              5-state       measurements
```

| Layer | Time Scale | Model | Purpose |
|-------|-----------|-------|---------|
| EMS | 1 hour | 3-state (SOC, SOH, T) | Economic planning |
| MPC | 1 minute | 3-state + OCV thermal | Real-time tracking |
| EKF/MHE | 1 minute | 5-state (+ V_rc1, V_rc2) | State estimation |
| Plant | 5 seconds | 5-state per cell, 2RC circuit | High-fidelity simulation |

## Versioned Upgrades

Each version adds one major capability, passes a **4-stage gate** (validation, evaluation, comparison, stress testing), and is frozen before the next begins.

| Version | What It Adds | Key Result |
|---------|-------------|------------|
| **v1** Baseline | 2-state EMS + MPC + EKF/MHE | $35 profit, 61ms MPC |
| **v2** Thermal Model | Temperature state, Arrhenius degradation | +3% degradation at elevated T |
| **v3** Pack Model | 4-cell pack, active balancing | SOC spread 2.4% -> 0.2% |
| **v4** Electrical RC | 2RC circuit, NMC OCV, voltage measurement | **47% better SOC estimation** |


## Quick Start

```bash
uv sync                                    # install dependencies
uv run python v4_electrical_rc_model/main.py   # run latest version (~5 min)
```

Each version is independently runnable. Results (`.npz` + `.png`) go to `results/`.

## Typical Results (24h simulation)

| Metric | Value |
|--------|-------|
| Net profit | $35 |
| SOH degradation | 0.26% |
| EKF SOC accuracy | 0.12% RMSE |
| MPC solve time | 222ms avg |
| Solver failures | 0 / 1,440 |
| Terminal voltage range | 738 -- 863 V |

## Technical Stack

CasADi + IPOPT for nonlinear optimization, NumPy for numerics, Matplotlib for visualization. All optimization models use automatic differentiation and warm-started interior-point solving.

## Roadmap

| Upcoming | Description |
|----------|------------|
| v5 | Regulation activation & MPC necessity |
| v6 | Unscented Kalman Filter (UKF) |
| v7 | Online parameter estimation |
| v8 | Real-time NMPC with ACADOS |
| v9 | Degradation-aware MPC |
| v10-v14 | Uncertainty, delays, multi-battery, inverter, market bidding |

---

*Each version contains its own `README.md` with full mathematical formulations and implementation details.*
