import os
import hydra
from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from models.scenario_dreamer_dm import ScenarioDreamerDM
from models.scenario_dreamer_dm_goal import ScenarioDreamerDMGoal
from models.scenario_dreamer_dm_fixed_map_agent_goal import ScenarioDreamerDMFixedMapAgentGoal
from models.scenario_dreamer_cldm import ScenarioDreamerCLDM
from model_registry import collapse_cfg
from metrics import Metrics

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from cfgs.config import CONFIG_PATH
from utils.train_helpers import set_latent_stats
import utils.sim_env_helpers as _sim_env_helpers


def generate_simulation_environments(cfg, cfg_ae, save_dir=None, model_cls=ScenarioDreamerLDM):
    """ Generate simulation environments using the Scenario Dreamer Latent Diffusion Model.
    
    This involves 1 step of initial scene generation followed by multiple steps of
    inpainting to extend the scenario until the desired route length is reached.
    Additional rule-based heuristics are applied to ensure scenario validity.
    """
    cfg = set_latent_stats(cfg)

    # load last ckpt for inference
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    model = model_cls.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    _sim_env_helpers.generate_simulation_environments(model, cfg, save_dir)


def eval_ldm(cfg, cfg_ae, save_dir=None, model_cls=ScenarioDreamerLDM):
    """ Evaluate the Scenario Dreamer Latent Diffusion Model.
    
    - mode: metrics
    use the generated scene to calculate metrics
    
    - mode: initial_scene
    generate scene
    
    """
    if cfg.eval.mode == 'metrics':
        metric_evaluator = Metrics(cfg)
        metric_evaluator.compute_metrics()
        return
    
    cfg = set_latent_stats(cfg)
    
    # load last ckpt for inference
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."
    
    # generate samples
    model = model_cls.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    # models that generate the same mode under several protocols (e.g. ldm_adv's
    # prior vs dataset sampling) set eval.cache_dir explicitly so their sample sets
    # do not overwrite each other; everything else keeps the default layout.
    cache_dir = cfg.eval.get('cache_dir') or os.path.join(save_dir, f'{cfg.eval.mode}_samples')
    model.generate(
        mode = cfg.eval.mode, # Scenario Dreamer supports multiple generation modes: initial_scene, lane_conditioned, and inpainting
        num_samples = cfg.eval.num_samples,
        batch_size = cfg.eval.batch_size,
        cache_samples = cfg.eval.cache_samples,
        visualize = cfg.eval.visualize,
        conditioning_path = cfg.eval.conditioning_path,
        cache_dir = cache_dir,
        viz_dir = cfg.eval.viz_dir,
        save_wandb = False,
        return_samples=False,
    )


def eval_dm(cfg, save_dir=None, model_cls=ScenarioDreamerDM):
    """Evaluate the direct vectorized diffusion model."""
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break

    assert ckpt_path is not None, "No checkpoint found in the save directory."

    model = model_cls.load_from_checkpoint(ckpt_path, cfg=cfg).to('cuda')
    model.generate(
        mode=cfg.eval.mode,
        num_samples=cfg.eval.num_samples,
        batch_size=cfg.eval.batch_size,
        cache_samples=cfg.eval.cache_samples,
        visualize=cfg.eval.visualize,
        conditioning_path=cfg.eval.conditioning_path,
        cache_dir=os.path.join(save_dir, f'{cfg.eval.mode}_samples'),
        viz_dir=cfg.eval.viz_dir,
        save_wandb=False,
        return_samples=False,
    )



def eval_autoencoder(cfg, save_dir=None):
    """ Evaluate the Scenario Dreamer AutoEncoder model."""
    model = ScenarioDreamerAutoEncoder(cfg)
    model_summary = ModelSummary(max_depth=-1)
    
    # load checkpoint
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
    
    assert ckpt_path is not None, "No checkpoint found in the save directory."
    
    tester = pl.Trainer(accelerator='auto',
                         devices=1,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary],
                         precision='32-true'
                        )
    
    tester.test(model, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    # need to track whether we are evaluating a nuplan or waymo model as
    # nuplan predicts lane types (lane/green light/red light) and waymo does not.
    # collapse_cfg picks the right root-config child node, injects dataset_name,
    # and returns the (frozen) autoencoder cfg for the latent-diffusion family.
    model_name = cfg.model_name
    # eval.py supports a subset of trained models (no ctrl_sim, no autoencoder_bezier).
    if model_name in ('ctrl_sim', 'autoencoder_bezier'):
        raise ValueError(f"Unsupported evaluation model_name: {model_name}")
    spec, cfg, cfg_ae = collapse_cfg(cfg, model_name)

    pl.seed_everything(cfg.eval.seed, workers=True)

    # checkpoints loaded from here
    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating Scenario Dreamer {model_name} trained on {cfg.dataset_name} dataset.")

    if spec.kind == 'autoencoder':
        eval_autoencoder(cfg, save_dir)
    elif spec.kind == 'ldm':
        if cfg.eval.mode == 'simulation_environments':
            generate_simulation_environments(cfg, cfg_ae, save_dir, model_cls=spec.model_cls)
        else:
            eval_ldm(cfg, cfg_ae, save_dir, model_cls=spec.model_cls)
    elif spec.kind == 'dm':
        eval_dm(cfg, save_dir, model_cls=spec.model_cls)


if __name__ == '__main__':
    main()
