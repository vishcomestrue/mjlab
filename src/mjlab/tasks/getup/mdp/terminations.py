"""Getup task terminations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fell_over(
  env: ManagerBasedRlEnv,
  max_tilt_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Check if robot has fallen over beyond maximum tilt angle.

  Args:
      env: The environment.
      max_tilt_angle: Maximum allowed tilt angle in degrees.
      asset_cfg: Asset configuration.

  Returns:
      Boolean tensor of shape (num_envs,) indicating termination.
  """
  asset = env.scene[asset_cfg.name]
  # projected_gravity_b[:, 2] is cos(tilt angle) when robot is upright
  # When upright: projected_gravity_b[:, 2] = -1
  # When tilted 90°: projected_gravity_b[:, 2] = 0
  max_tilt_cos = math.cos(math.radians(max_tilt_angle))
  # Terminate when |projected_gravity_b[:, 2]| < max_tilt_cos
  return torch.abs(asset.data.projected_gravity_b[:, 2]) < max_tilt_cos
