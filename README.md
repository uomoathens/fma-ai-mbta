# FMA-AI: Federated Multi-Agent AI for Energy-Aware Public Transport (MBTA GTFS)

Code, data and results for the paper:

> Sargiotis, D. (2026). *Federated Multi-Agent AI for Privacy-Preserving, Energy-Aware Optimisation of Public Transport Systems* (revised version, October 2026). SSRN. https://doi.org/10.2139/ssrn.5577250

Route-level agents learn a shared speed-regulation policy through synchronous federated averaging (FedAvg). The local objective minimises traction, auxiliary and congestion-related energy subject to a bounded loss in on-time performance (OTP). The framework is evaluated in a simulation built on the MBTA GTFS feed, with six training routes, 24 hold-out routes and 20 random seeds, and is compared with a centralised optimiser and with route-isolated learners.

## Main results (MATLAB, 20 seeds, mean ± 95% CI, OTP tolerance 2 pp)

| Regime | Routes | ΔE = ΔCO₂ (%) | OTP (%) | R_conv (median) |
|---|---|---|---|---|
| Timetable baseline | Training | 0 | 71.2 | |
| Centralised | Training | 0.83 ± 0.08 | 70.0 ± 0.3 | 143 |
| Non-federated MARL | Training | 0.90 ± 0.12 | 70.0 ± 0.3 | not met by 400 |
| FMA-AI (FedAvg) | Training | 0.78 ± 0.07 | 70.1 ± 0.3 | 215 |
| Centralised | Hold-out | 0.80 ± 0.06 | 69.5 ± 0.1 | |
| Non-federated MARL | Hold-out | 0 (no model) | 71.1 | |
| FMA-AI (FedAvg) | Hold-out | 0.88 ± 0.06 | 69.3 ± 0.1 | |

## Revision note

The October 2026 revision supersedes the results posted in October 2025 (7.1% reduction). The earlier implementation selected six Sunday commuter-rail trips instead of bus routes, performed no parameter averaging in the federated regime, used a non-optimising centralised baseline, stopped training before convergence from a pre-reduced starting speed, and normalised energy by the product of total passengers and total distance. All of these are corrected in the current code, and all reported figures were regenerated.

## Repository structure

| Path | Content |
|---|---|
| `fma_mbta.m` | Complete experiment: GTFS ingestion, route selection, energy and OTP models, centralised, non-federated and FedAvg training, evaluation, tolerance sweep, CSV output and plots |
| `MBTA_GTFS.zip` | MBTA GTFS feed used in the paper (Summer 2025, version D, valid 31 Jul to 23 Aug 2025) |
| `results/matlab/` | Outputs of `fma_mbta.m` in MATLAB, as reported in the paper |
| `results/octave/` | Independent replication of the same script in GNU Octave 8.4 (different random streams; same route set, overlapping confidence intervals) |
| `analysis/paired_stats_and_figures.py` | Paired differences between regimes, convergence ranges, and the published Figures 1 and 2 |
| `analysis/paired_differences.csv`, `analysis/convergence_rounds.csv` | Outputs of the analysis script for the MATLAB results |
| `figures/` | Figures 1 and 2 as published |
| `paper/` | Revised manuscript (PDF) |

## How to reproduce

1. Place `fma_mbta.m` and `MBTA_GTFS.zip` in the same folder (as in this repository).
2. In MATLAB (or GNU Octave 8 or later), set that folder as the working directory and run `fma_mbta`.
3. Outputs are written to `results/` (CSV files and two PNG figures). A full run takes roughly 20 to 40 minutes on a standard laptop.
4. For the paired statistics and the published figures, run `python analysis/paired_stats_and_figures.py matlab` (requires numpy, pandas, scipy, matplotlib). Use the argument `octave` to analyse the Octave replication instead.

Route selection is deterministic and should reproduce `results/matlab/routes.csv` exactly. Indicators depend on the random-number generator and are expected to match within the reported confidence intervals rather than digit for digit across MATLAB versions or Octave.

## Assumptions and limitations

Arrival disturbances are synthetic and uncalibrated, the average load is fixed at 20 passengers per trip, all vehicles are modelled as battery-electric, and federated averaging is implemented without secure aggregation or differential privacy. See Section 5 of the paper.

## Data attribution

Transit data provided by the Massachusetts Department of Transportation (MassDOT) and the MBTA, redistributed under the MassDOT Developers License Agreement. The data are provided "as is"; MassDOT retains ownership and does not endorse this work.

## License

Code (`fma_mbta.m`, `analysis/`) is released under the MIT License (see `LICENSE`). The GTFS data remain under the MassDOT Developers License Agreement. The manuscript in `paper/` is © the author, all rights reserved.

## Citation

See `CITATION.cff`, or cite the SSRN paper above.
