# TODO

## DDPO puffer reward backend

- Align native puffer `ego_min_ttc` with the numpy backend's ego-swept box TTC.
  The current C binding uses point-mass radial closing time with pair relative
  velocity, so it can score another actor moving toward the ego or lateral
  closing that would not become an ego box collision.
- Audit native puffer reward metrics against numpy for:
  - controlled vs static agent inclusion in TTC and collision checks
  - goal-reached / respawn masking
  - `init_invalid` consistency
  - scene-index ordering for eval visualizations
- Until this is done, prefer `ddpo.reward_backend=numpy` for DDPO reward
  experiments that depend on TTC semantics.
