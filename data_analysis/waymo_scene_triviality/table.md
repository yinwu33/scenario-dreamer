### train (486995 scenes)

| ego moves less than | net displacement | path length | max speed below | scenes |
| --- | --- | --- | --- | --- |
| 1 m | 21.89% | 21.83% | 0.5 m/s | 20.42% |
| 2 m | 23.57% | 23.53% | 1.0 m/s | 22.29% |
| 5 m | 27.10% | 27.06% | 2.0 m/s | 25.76% |
| 10 m | 31.94% | 31.85% |  |  |
| 20 m | 41.15% | 40.28% |  |  |

| no other agent within | at scene init | over the full 9.1 s |
| --- | --- | --- |
| 5 m (all) | 69.24% | 40.04% |
| 10 m (all) | 30.38% | 11.15% |
| 15 m (all) | 13.81% | 4.28% |
| 20 m (all) | 6.78% | 2.32% |
| 30 m (all) | 1.78% | 1.14% |
| 5 m (vehicles) | 70.54% | 42.15% |
| 10 m (vehicles) | 32.68% | 12.79% |
| 15 m (vehicles) | 15.26% | 4.97% |
| 20 m (vehicles) | 7.62% | 2.70% |
| 30 m (vehicles) | 2.05% | 1.32% |
| 5 m (model_view) | 71.51% | 44.11% |
| 10 m (model_view) | 35.65% | 15.56% |
| 15 m (model_view) | 19.33% | 7.91% |
| 20 m (model_view) | 11.53% | 5.38% |
| 30 m (model_view) | 4.86% | 3.77% |

medians: {"net_disp_m": 31.35, "path_len_m": 32.92, "num_agents": 12.0, "gap_init_all_m": 7.26}

### val (44097 scenes)

| ego moves less than | net displacement | path length | max speed below | scenes |
| --- | --- | --- | --- | --- |
| 1 m | 22.31% | 22.23% | 0.5 m/s | 20.73% |
| 2 m | 24.04% | 24.00% | 1.0 m/s | 22.61% |
| 5 m | 27.49% | 27.48% | 2.0 m/s | 26.34% |
| 10 m | 32.48% | 32.37% |  |  |
| 20 m | 41.39% | 40.64% |  |  |

| no other agent within | at scene init | over the full 9.1 s |
| --- | --- | --- |
| 5 m (all) | 69.14% | 40.18% |
| 10 m (all) | 30.53% | 11.53% |
| 15 m (all) | 13.84% | 4.61% |
| 20 m (all) | 6.86% | 2.46% |
| 30 m (all) | 1.82% | 1.12% |
| 5 m (vehicles) | 70.46% | 42.41% |
| 10 m (vehicles) | 33.00% | 13.28% |
| 15 m (vehicles) | 15.52% | 5.45% |
| 20 m (vehicles) | 7.80% | 2.85% |
| 30 m (vehicles) | 2.07% | 1.33% |
| 5 m (model_view) | 71.45% | 44.36% |
| 10 m (model_view) | 35.68% | 16.04% |
| 15 m (model_view) | 19.45% | 8.27% |
| 20 m (model_view) | 11.63% | 5.62% |
| 30 m (model_view) | 4.92% | 3.78% |

medians: {"net_disp_m": 31.54, "path_len_m": 32.96, "num_agents": 12.0, "gap_init_all_m": 7.25}

### train+val (531092 scenes)

| ego moves less than | net displacement | path length | max speed below | scenes |
| --- | --- | --- | --- | --- |
| 1 m | 21.93% | 21.86% | 0.5 m/s | 20.44% |
| 2 m | 23.61% | 23.57% | 1.0 m/s | 22.32% |
| 5 m | 27.13% | 27.10% | 2.0 m/s | 25.81% |
| 10 m | 31.98% | 31.89% |  |  |
| 20 m | 41.17% | 40.31% |  |  |

| no other agent within | at scene init | over the full 9.1 s |
| --- | --- | --- |
| 5 m (all) | 69.23% | 40.05% |
| 10 m (all) | 30.39% | 11.18% |
| 15 m (all) | 13.81% | 4.31% |
| 20 m (all) | 6.79% | 2.33% |
| 30 m (all) | 1.79% | 1.13% |
| 5 m (vehicles) | 70.53% | 42.17% |
| 10 m (vehicles) | 32.71% | 12.83% |
| 15 m (vehicles) | 15.29% | 5.01% |
| 20 m (vehicles) | 7.64% | 2.71% |
| 30 m (vehicles) | 2.06% | 1.32% |
| 5 m (model_view) | 71.50% | 44.13% |
| 10 m (model_view) | 35.65% | 15.60% |
| 15 m (model_view) | 19.34% | 7.94% |
| 20 m (model_view) | 11.54% | 5.40% |
| 30 m (model_view) | 4.87% | 3.77% |

medians: {"net_disp_m": 31.36, "path_len_m": 32.92, "num_agents": 12.0, "gap_init_all_m": 7.26}

