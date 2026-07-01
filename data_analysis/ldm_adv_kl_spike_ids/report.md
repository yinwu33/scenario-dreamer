# ldm_adv KL Spike Id Mapping

Local logs contain batch-mean `kl_to_base` spikes, but not per-sample ids. I
replayed the deterministic `LDMAdvConditioningPool` sampling sequence with
`seed=0`, `batch_size=64`, `group_size=8`, `pool_size=40000` to map spike
iterations back to train split dataset ids.

Important caveat: these are the 8 conditioning contexts in the high-KL batch, not
per-scene KL contributors. The current train log does not store per-scene KL, so
one of these contexts may dominate the batch mean.

## Highest Local Spikes

| iter | batch KL | train `--id` contexts |
| ---: | ---: | --- |
| 449 | 0.0534 | 359490 65113 156659 325976 483092 151941 214302 351614 |
| 399 | 0.0475 | 320332 281572 381673 61621 186823 345481 374023 74036 |
| 281 | 0.0434 | 58999 271562 240988 23817 85539 120592 127584 420431 |
| 450 | 0.0422 | 191896 149074 84839 370112 485062 296907 14009 352559 |
| 651 | 0.0416 | 50109 346837 371850 483367 482497 135008 198362 478941 |

Close to checkpoint 100:

| iter | train `--id` contexts |
| ---: | --- |
| 99 | 295622 2129 145971 96918 131823 438298 108365 364521 |
| 93 | 376489 183872 471918 288767 464097 257069 194573 254694 |
| 100 | 453768 81679 193123 99337 160126 333104 403558 226742 |
| 101 | 417723 208301 381525 249477 84051 8039 16667 19398 |

The local W&B train-group table for true `iter=99` only logs the most diverse
groups, but it joins cleanly to the reconstructed ids:

| context id | group | observed rewards in logged group rows |
| ---: | ---: | --- |
| 96918 | 3 | -0.2500, -0.0003, -0.0000, 0.0000, 0.0000, 0.0511, 0.1393, 0.8368 |
| 364521 | 7 | -1.0000, -1.0000, -0.4612, -0.3620, -0.2500, -0.2500, 0.0047, 0.0289 |

Those two ids are good first visual checks because the existing training
diagnostic already selected samples from them for media.

## Visualization

I started the requested rollout for the top id with the base ldm_adv checkpoint:

```bash
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py \
  --split train --id 359490 --num 1 --device cpu \
  --out-root data_analysis/ldm_adv_kl_spike_viz \
  --output-prefix kl_spike_base --gif-max-frames 60
```

The GT rollout was written to:

```text
data_analysis/ldm_adv_kl_spike_viz/kl_spike_base_init_adv_359490/gt_rollout.gif
```

I interrupted the generated DDPM sample after about 90s because this machine has
no available CUDA device and `test_rollout_ldm_adv.py` uses the full DDPM sampler.
Run the same command on a CUDA node with `--device cuda` to finish the generated
rollout GIF.

Suggested CUDA commands for a few top contexts:

```bash
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py --split train --id 359490 --num 4 --device cuda --out-root data_analysis/ldm_adv_kl_spike_viz --output-prefix kl_spike_base
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py --split train --id 65113 --num 4 --device cuda --out-root data_analysis/ldm_adv_kl_spike_viz --output-prefix kl_spike_base
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py --split train --id 156659 --num 4 --device cuda --out-root data_analysis/ldm_adv_kl_spike_viz --output-prefix kl_spike_base
```

For checkpoint-100 policy visualization:

```bash
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py \
  --split train --id 96918 --num 4 --device cuda \
  --ckpt data/critical_scene/critical_scene_ddpo_ldm_adv_ddpm_bad_driver/critical_scene_ddpo_ldm_adv_ddpm_bad_driver_00100.ckpt \
  --out-root data_analysis/ldm_adv_kl_spike_viz --output-prefix kl_spike_ddpo100
MPLCONFIGDIR=/tmp/matplotlib python test_scripts/test_rollout_ldm_adv.py \
  --split train --id 364521 --num 4 --device cuda \
  --ckpt data/critical_scene/critical_scene_ddpo_ldm_adv_ddpm_bad_driver/critical_scene_ddpo_ldm_adv_ddpm_bad_driver_00100.ckpt \
  --out-root data_analysis/ldm_adv_kl_spike_viz --output-prefix kl_spike_ddpo100
```
