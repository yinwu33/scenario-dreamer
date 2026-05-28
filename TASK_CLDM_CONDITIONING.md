# CLDM Conditioning Task

## Goal

Build a conditional latent diffusion model for Waymo by adding map-level metadata conditions:

- `num_junctions`
- `curvature_frac`

The intended metadata source is one text file per split, for example `waymo_train.txt`. Each line maps one preprocessed scenario file to its map statistics:

```text
<file_name> <num_junctions> <curvature_frac>
```

The `<file_name>` should correspond to a `.pkl` file under the matching split directory, such as:

```text
data/scenario_dreamer_ae_preprocess_waymo/train
data/scenario_dreamer_ae_preprocess_waymo/val
data/scenario_dreamer_ae_preprocess_waymo/test
```

The CLDM dataset should load the same latent `.pkl` data as LDM and attach these two metadata values as a scene-level conditioning tensor for diffusion.

## Feasibility

This plan is feasible.

The current diffusion model already supports scene-level conditioning through DiT:

- diffusion timestep embedding
- scene/map label embedding from `lg_type` and `map_id`
- number-of-agents embedding
- number-of-lanes embedding

These are added into the DiT conditioning vector:

```python
c = t + y + n
```

`num_junctions` and `curvature_frac` can be embedded as a continuous scene-level condition and added to this same `c` vector after expanding from graph/sample level to lane and agent nodes through `data['lane'].batch` and `data['agent'].batch`.

## Current Code State

The CLDM files are currently mostly copies of LDM and are not wired as an independent conditional model yet.

- `nn_modules/cldm.py` is byte-for-byte identical to `nn_modules/ldm.py`.
- `datasets/waymo/dataset_cldm_waymo.py` is byte-for-byte identical to `datasets/waymo/dataset_ldm_waymo.py`.
- `datamodules/waymo/waymo_datamodule_cldm.py` still imports `WaymoDatasetLDM` from `dataset_ldm_waymo.py`.
- `cfgs/datamodule/waymo_cldm.yaml` still targets `datamodules.waymo.waymo_datamodule_ldm.WaymoDataModuleLDM`.
- `models/scenario_dreamer_cldm.py` still defines `ScenarioDreamerLDM` and imports `LDM` from `nn_modules.ldm`.
- `train.py` and `eval.py` only import and instantiate `models.scenario_dreamer_ldm.ScenarioDreamerLDM`.
- `cfgs/config.yaml` only has an `ldm` config group; there is no separate `cldm` model mode.

So the implementation needs both data/model changes and config/entrypoint wiring.

## Metadata Contract

Recommended split-specific config fields:

```yaml
metadata_dir: ${project_root}/metadata/waymo_map_conditioning
metadata_filename_template: waymo_{split}.txt
condition_dim: 2
condition_names:
  - num_junctions
  - curvature_frac
condition_mean: null
condition_std: null
```

Recommended line parsing:

```text
<file_name> <num_junctions> <curvature_frac>
```

Recommended filename normalization:

- Accept `abc.pkl` and `abc` as equivalent keys.
- Store metadata keys without `.pkl`.
- Match against `os.path.splitext(os.path.basename(self.files[idx]))[0]`.

Recommended dataset behavior:

- Fail fast if the metadata file for a split is missing.
- Fail fast if a `.pkl` sample has no metadata row.
- Validate each metadata row has exactly 3 fields.
- Cast `num_junctions` and `curvature_frac` to `float32`.
- Attach `d['condition'] = torch.tensor([num_junctions, curvature_frac], dtype=torch.float32)`.

## Normalization Decision

The condition values should be normalized before being fed to the network.

Recommended approach:

- Compute condition mean/std from train metadata only.
- Store them in config or a small cached stats file.
- Apply the same stats to train/val/test.
- Clamp `std` with a small epsilon to avoid divide-by-zero.

`curvature_frac` is probably already in a bounded range, but normalizing both fields keeps the embedding scale predictable. `num_junctions` is count-like and likely needs normalization.

## Model Injection Plan

Recommended minimal model change:

1. Add a continuous condition embedder to DiT or a CLDM-specific DiT variant:

   ```python
   self.map_condition_embedder = TwoLayerResMLP(condition_dim, hidden_dim)
   ```

2. In `forward`, read:

   ```python
   scene_cond = data.condition.float()
   ```

3. Normalize if dataset has not already normalized it.

4. Embed per scene:

   ```python
   map_cond = self.map_condition_embedder(scene_cond)
   ```

5. Expand to nodes:

   ```python
   lane_map_cond = map_cond[lane_batch]
   agent_map_cond = map_cond[agent_batch]
   ```

6. Add it into `c` in lane/agent order:

   ```python
   z = torch.cat([lane_map_cond, agent_map_cond], dim=0)
   c = t + y + n + z
   ```

This is the lowest-risk path because it reuses the existing DiT conditioning mechanism.

## Classifier-Free Guidance

Current classifier-free guidance only drops the scene/map label embedding. If the new map condition is always present, then inference guidance will not be fully unconditional with respect to `num_junctions` and `curvature_frac`.

Recommended options:

- Simple first version: no condition dropout for continuous metadata; use it as always-on conditioning.
- Better CLDM version: add `condition_dropout` and replace dropped continuous conditions with a learned null embedding or zero vector.

For true conditional guidance on the new metadata, implement the second option.

## Generation/Evaluation Impact

Training batches can read condition values directly from metadata. Generation is different because `initial_scene` creates synthetic graph placeholders before sampling.

The generation path must decide where conditions come from:

- User-specified fixed condition, for controlled generation.
- Randomly sampled condition from train metadata distribution.
- Condition copied from a lane-conditioned or inpainting source file.

For initial CLDM training, generation can be deferred. For validation visualization during training, `validation_step` calls `forward` on real validation batch data, so metadata-loaded conditions will already exist.

## Files To Change Later

Dataset and datamodule:

- `datasets/waymo/dataset_cldm_waymo.py`
  - Rename class to `WaymoDatasetCLDM`.
  - Load split metadata file once in `__init__`.
  - Attach `d['condition']`.
  - Add validation and normalization.
- `datamodules/waymo/waymo_datamodule_cldm.py`
  - Import `WaymoDatasetCLDM`.
  - Rename class to `WaymoDataModuleCLDM`.
- `cfgs/dataset/waymo_cldm.yaml`
  - Add metadata path/stats fields.
- `cfgs/datamodule/waymo_cldm.yaml`
  - Target the CLDM datamodule.

Model:

- `nn_modules/cldm.py`
  - Rename `LDM` to `CLDM`, or keep diffusion wrapper name but import CLDM-specific DiT.
  - Ensure sampling/loss paths pass `data.condition` through naturally.
- `nn_modules/dit.py` or a new `nn_modules/cdit.py`
  - Add continuous condition embedding.
  - Add optional dropout/null condition for classifier-free guidance.
- `models/scenario_dreamer_cldm.py`
  - Rename class to `ScenarioDreamerCLDM`.
  - Import `CLDM` from `nn_modules.cldm`.

Config/entrypoints:

- `cfgs/config.yaml`
  - Add a `cldm` config group, or intentionally map `model_name=ldm` to CLDM configs during development.
- `train.py`
  - Add `model_name == 'cldm'` path and instantiate `ScenarioDreamerCLDM`.
- `eval.py`
  - Add CLDM loading/generation path.
- `cfgs/train/waymo_cldm.yaml`
  - Update `run_name`.
- `cfgs/eval/waymo_cldm.yaml`
  - Update `run_name`.

## Implementation Checklist

- [ ] Finalize metadata file location and naming convention.
- [ ] Build `waymo_train.txt`, `waymo_val.txt`, and optionally `waymo_test.txt`.
- [x] Add CLDM dataset metadata parsing.
- [x] Add condition stats computation or config values.
- [x] Add `data.condition` to CLDM samples.
- [x] Wire CLDM datamodule config to CLDM datamodule.
- [x] Add continuous condition embedder.
- [x] Decide and implement continuous-condition dropout behavior.
- [x] Rename CLDM classes/imports so LDM and CLDM can coexist.
- [x] Add `model_name=cldm` training path.
- [x] Add CLDM evaluation/generation path.
- [ ] Add a small dataset smoke test.
- [ ] Add a model forward/loss smoke test with one batch.
- [ ] Verify validation visualization still works.

## Progress Notes

- Implemented CLDM train/eval wiring through `model_name=cldm`, reusing the existing `ldm` config namespace.
- Implemented train-metadata condition stats, `num_junctions` clipping at 5, and normalization.
- Added continuous condition dropout with a learned null condition embedding.
- Added random train-condition sampling for generation/inpainting/lane-conditioned generation.
- Added condition text overlays to generated/validation visualizations.

## Main Risks

- File-name mismatch between metadata rows and latent `.pkl` files.
- Metadata derived from AE preprocess files may not match latent-cache filenames if latent generation changed names or split paths.
- `num_junctions` scale can dominate the condition embedding if left unnormalized.
- Generation needs an explicit condition source; otherwise CLDM cannot know what map statistics to condition on.
- If continuous conditions are not dropped during classifier-free guidance, `guidance_scale` will only guide existing discrete labels, not the new metadata.

## Suggested First Milestone

Implement only the training-time condition path first:

1. Parse metadata in `WaymoDatasetCLDM`.
2. Attach normalized `data.condition`.
3. Add a condition embedder to CLDM DiT.
4. Run one dataloader batch and one `diff_model.loss(data)` smoke test.

Defer controlled generation until training data flow is verified.
