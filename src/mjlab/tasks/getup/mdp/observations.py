"""Getup task observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = 0.10,
  max_height: float = 0.25,
) -> torch.Tensor:
  """Get normalized robot body height from ground.

  Normalizes height to [0, 1] range based on full possible height range.
  Range covers seated (0.12) to standing (0.15-0.19) to overshoot (0.25).

  Args:
      env: The environment.
      asset_cfg: Asset configuration.
      min_height: Minimum height for normalization (default: 0.10).
      max_height: Maximum height for normalization (default: 0.25).

  Returns:
      Normalized height tensor of shape (num_envs, 1), clipped to [-0.5, 1.5].
  """
  asset = env.scene[asset_cfg.name]
  raw_height = asset.data.root_link_pos_w[:, 2:3]
  # Normalize to [0, 1] based on full range
  normalized = (raw_height - min_height) / (max_height - min_height)
  # Clip to handle extreme cases (fallen/jumped)
  normalized = torch.clamp(normalized, -0.5, 1.5)
  return normalized


def target_joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Get target standing joint positions (zeros in relative space).

  For getup task, the target is the default pose, which is zeros
  in relative joint position space.
  """
  asset = env.scene[asset_cfg.name]
  return torch.zeros(
    (env.num_envs, asset.num_joints),
    device=env.device,
  )


def target_height(
  env: ManagerBasedRlEnv,
  min_height: float = 0.10,
  max_height: float = 0.25,
) -> torch.Tensor:
  """Get normalized target body height for each environment.

  Normalizes height to [0, 1] range based on full possible height range.
  Uses same normalization as body_height for consistency.

  Args:
      env: The environment.
      min_height: Minimum height for normalization (default: 0.10).
      max_height: Maximum height for normalization (default: 0.25).

  Returns:
      Normalized target height tensor of shape (num_envs, 1).
  """
  if "target_height" in env.extras:
    raw_height = env.extras["target_height"]
    # Normalize to [0, 1] based on full range
    # Target is always in [0.15, 0.19], so normalized ~[0.33, 0.60]
    normalized = (raw_height - min_height) / (max_height - min_height)
    return normalized.unsqueeze(-1)
  else:
    # Fallback: return middle of standing range (~0.17 normalized)
    return torch.full((env.num_envs, 1), 0.47, device=env.device, dtype=torch.float32)
