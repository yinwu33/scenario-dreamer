"""SMART-style learned traffic model: a behavior model, kept out of the main repo.

The rest of this repository is a diffusion pipeline (autoencoder -> latent
diffusion -> DDPO). This package is a different kind of thing -- an imitation
behavior model -- so everything specific to it lives here: the network, its
observation spec, the rollout planner that wraps it, data processing over the
Waymo records, and the training entry point. Nothing in ``models/``, ``nets/``,
``datasets/`` or ``datamodules/`` belongs to it.

The seam with the main repo is deliberately two lines plus one config, and
nothing else:

  * ``sim/planners/__init__.py`` imports ``smart.planner.SMARTPlanner`` and maps
    one name to it in ``PLANNER_REGISTRY``. That registry is how any role's
    planner is resolved, so a planner is only reachable through it.
  * ``cfgs/planner/<name>.yaml`` must live in the main config tree, because the
    rollout composes its roles from the ``planner`` hydra group
    (``planner@ddpo.planner.env=smart_probe``).

The dependency runs one way: this package imports from ``sim`` -- the planner
contract, the world's scaling constants, and (in ``actions``) the shared
accel/steer table and its integrator. That last one is deliberate and load
bearing: action labels are produced by inverting exactly the dynamics
``SimScene.step_dynamics`` will run, so a training target can never drift from
what the simulator does. ``sim`` imports nothing from here except the one
registry line.

See ``README.md`` in this directory for the workflow.
"""
