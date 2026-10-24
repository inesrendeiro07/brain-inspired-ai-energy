# brain-inspired-ai-energy

MATLAB code for the experiments in my report *Thinking on 20 Watts: What can AI learn from the brain's energy-saving mechanisms*.

The code tests three of the brain's energy-saving mechanisms on two small networks trained on the same classification task:

- **Locality**: weights stored next to the arithmetic (on-chip SRAM) instead of in off-chip DRAM
- **Sparsity**: only part of the network active per input (mixture-of-experts routing for the transformer, limited spiking for the SNN)
- **Coding / depth**: stopping early when the answer is clear (early exit for the transformer, time-to-first-spike code for the SNN)

Each mechanism is switched on and off in a 2 x 2 x 2 factorial on both a transformer (conditions C1 to C8) and a spiking neural network (B1 to B8). Energy per trial is estimated from operation and memory-access counts, using the 45 nm figures in Horowitz (2014), and compared with a brain reference based on Attwell and Laughlin (2001).

## Requirements

- MATLAB R2020b or later (uses `pagemtimes` and `exportgraphics`)
- No toolboxes needed

## How to run

From the repository folder in MATLAB:

```matlab
run_part4                 % main investigation, about 2 minutes
run_sensitivity           % sensitivity sweep, about 45 minutes
make_sensitivity_figures  % figures for the sweep
```

Then, for the extra report figures:

```matlab
cd extra_figures
make_extra_figures
```

`run_part4` runs Stages 4.1 to 4.5 in order. Each stage loads what it needs from `results/`, so a single stage can be re-run after the ones before it.

| Script | What it does |
|---|---|
| `stage_4_1.m` | Builds the task and train/validation/test splits, trains all networks |
| `stage_4_2.m` | Sparsity sweep on both networks (validation set) |
| `stage_4_3.m` | Coding / depth: early exit and rate vs timing code (validation set) |
| `stage_4_4.m` | Full factorial, batching and brain reference (test set) |
| `stage_4_5.m` | Summary tables |
| `run_sensitivity.m` | One-at-a-time sensitivity sweep (seeds, task difficulty, network size, memory costs) |
| `explore_task_difficulty.m` | Used to choose the easy/hard noise levels |
| `lib/` | Model, training and energy functions. All settings are in `lib/part4_config.m` |

## Figures

| File | Made by |
|---|---|
| `results/fig_4_2_sparsity_sweep.png` | `stage_4_2.m` |
| `results/fig_4_3_coding_frontier.png` | `stage_4_3.m` |
| `results/fig_4_4_factorial_energy.png` | `stage_4_4.m` |
| `results/fig_4_4_headline.png` | `stage_4_4.m` |
| `results/sensitivity/fig_sens_*.png` | `make_sensitivity_figures.m` |
| `extra_figures/fig7_all_conditions.png` | `make_extra_figures.m` |
| `extra_figures/fig4_landauer.png` | `make_extra_figures.m` |
| `extra_figures/fig_seed_spread.png` | `make_extra_figures.m` |

The sensitivity results are also saved as tables in `results/sensitivity/sensitivity_summary.csv` and `sensitivity_conditions.csv`, and the seed spread in `extra_figures/seed_spread.csv`.

## Licence

MIT, see `LICENSE`.
