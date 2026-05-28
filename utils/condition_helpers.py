import os

import torch


def resolve_condition_metadata_path(cfg, split_name):
    metadata_dir = cfg.condition_metadata_dir
    template = cfg.get("condition_metadata_filename_template", "waymo_{split}.txt")
    filename = template.format(split=split_name)
    return os.path.join(metadata_dir, filename)


def load_condition_metadata(cfg, split_name):
    metadata_path = resolve_condition_metadata_path(cfg, split_name)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing condition metadata file: {metadata_path}")

    metadata = {}
    with open(metadata_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Expected 3 fields in {metadata_path}:{line_idx}, got {len(fields)}: {stripped}"
                )

            file_name, num_junctions, curvature_frac = fields
            key = os.path.splitext(os.path.basename(file_name))[0]
            if key in metadata:
                raise ValueError(f"Duplicate condition metadata key in {metadata_path}: {key}")

            metadata[key] = (
                float(num_junctions),
                float(curvature_frac),
            )

    if len(metadata) == 0:
        raise ValueError(f"Condition metadata file is empty: {metadata_path}")

    return metadata


def clip_condition_values(values, num_junctions_clip):
    condition = torch.as_tensor(values, dtype=torch.float32)
    clipped = condition.clone()
    clipped[..., 0] = torch.clamp(clipped[..., 0], max=float(num_junctions_clip))
    return clipped


def condition_stats_from_metadata(metadata, num_junctions_clip, eps=1e-6):
    values = torch.tensor(list(metadata.values()), dtype=torch.float32)
    clipped = clip_condition_values(values, num_junctions_clip)
    mean = clipped.mean(dim=0)
    std = clipped.std(dim=0, unbiased=False)
    std = torch.clamp(std, min=eps)
    return mean, std


def get_condition_stats(cfg):
    if cfg.condition_mean is not None and cfg.condition_std is not None:
        mean = torch.as_tensor(cfg.condition_mean, dtype=torch.float32)
        std = torch.as_tensor(cfg.condition_std, dtype=torch.float32)
        std = torch.clamp(std, min=float(cfg.get("condition_normalization_eps", 1e-6)))
        return mean, std

    stats_split = cfg.get("condition_stats_split", "train")
    train_metadata = load_condition_metadata(cfg, stats_split)
    return condition_stats_from_metadata(
        train_metadata,
        cfg.condition_num_junctions_clip,
        eps=float(cfg.get("condition_normalization_eps", 1e-6)),
    )


def normalize_condition_values(values, mean, std, num_junctions_clip):
    raw = torch.as_tensor(values, dtype=torch.float32)
    clipped = clip_condition_values(raw, num_junctions_clip)
    normalized = (clipped - mean) / std
    return raw, clipped, normalized
