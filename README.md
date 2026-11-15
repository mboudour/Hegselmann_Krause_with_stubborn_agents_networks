# Hegselmann–Krause dynamics with stubborn agents on networks

Python code for the computations reported in the manuscript *Hegselmann–Krause Dynamics with Stubborn Agents on Networks* (M. Boudourides). The model: synchronous Hegselmann–Krause averaging on a fixed undirected graph, where stubborn agents have confidence bound 0 and all other agents share a common bound σ.

All computations that involve critical edges (pairs at distance exactly σ) or switching of the interaction graph use exact rational arithmetic (`fractions.Fraction`), since rounding destroys ties at distance σ. All scripts use fixed random seeds. Section and table numbers refer to the manuscript.

## Requirements

Python 3.10 or later and the packages in `requirements.txt` (`pygraphviz` needs the Graphviz system library).

```
pip install -r requirements.txt
```

Run each script from the repository root, e.g. `python3 hk_examples.py`. Several scripts import helper functions from `hk_critical_edges.py` and `hk_balanced_2d.py`, so keep all files in one folder.

## Scripts behind the manuscript

| Manuscript item | Script | Notes |
|---|---|---|
| Examples 7.1, 7.2, 7.4 | `hk_examples.py` | exact checks of every number in the examples |
| Example 7.5 | `hk_heterogeneous_counterexamples.py` | exact re-verification of the counterexamples of Hegarty–Ognissanti–Wedin (arXiv:2610.03229) and Etesami–Touri (arXiv:2610.20783), with the endpoint agents made stubborn |
| Figures 1 and 2 | `hk_figures.py` | writes `figures/fig_limit_structure.pdf` and `figures/fig_convergence.pdf`; the d = 2 run is certified frozen and its exact limit is computed from the terminal affine piece |
| Table 2, rows 1–2 (energy inequality; heterogeneous bounds) | `hk_stubborn_check.py` | floating point |
| Table 2, row 3 (critical edges at the limit, d = 1) | `hk_critical_edges.py` | also provides `step`, `frozen_limit`, `solve_exact` used elsewhere |
| Table 2, row 4 (monotonicity of active edges) | `hk_uu_vs_us.py` | 55,000 attempts, about 6 minutes |
| Table 2, row 5 (balanced limits, d = 1) | `hk_balanced_search.py` | |
| Table 2, row 6 (Lemma 5.8 identity, d = 2) | `hk_local_identity.py` | |
| Table 2, row 7 (balanced limits, d = 2, with freezing certificates) | `hk_certify_2d.py` | about 4 minutes; writes `hk_certificates_2d.json` |

## Certificate data

`hk_certificates_2d.json` contains one record for each of the 144 runs of `hk_certify_2d.py`: the graph, the stubborn set, the final time `T`, the time of the last observed change, the exact state `x_T`, the fixed point `y` of the affine piece in use at time `T`, the bound `rbar` (an upper bound for max_i ‖x_i(T) − y_i‖), the state and squared length at `y` of every edge, whether the invariance certificate holds, and whether the candidate limit `y` has a critical edge. All numbers are exact rationals written as strings.

The certificate: if every edge lies on the same side of σ at `y` as in the current pattern, with margin larger than 2·`rbar`, then the ball of radius `rbar` about `y` is invariant under the current affine piece and lies in the region where the update uses that piece, so the interaction graph is constant from time `T` on.

## Exploratory scripts

These were used while developing the results and are not cited for numbers in the manuscript. Some use time budgets rather than fixed trial counts, so their counts can vary between machines.

| Script | Purpose |
|---|---|
| `hk_adversarial.py` | planted ties at distance σ on small graphs; exact re-runs of the worst cases |
| `hk_capture.py`, `hk_capture_large.py`, `hk_capture_followup.py` | tests of edge monotonicity (an edge that switches on never switches off), which fails |
| `hk_perron_coeff.py` | spectral decomposition of the gap along critical edges after freezing |
| `hk_stress.py` | perturbations along negative eigenvectors next to a stubborn agent at distance σ |
| `hk_balanced_2d.py` | time-budgeted search for balanced configurations in d = 2; superseded by `hk_certify_2d.py` |
| `hk_square_2d.py` | an agent at the centre of four stubborn agents at distance σ |
