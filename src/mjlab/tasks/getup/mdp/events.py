"""Getup task events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_to_seated_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  seated_joint_pos: dict[str, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robot to initial seated/crouched position.

  Args:
      env: The environment.
      env_ids: Environment indices to reset.
      seated_joint_pos: Optional dictionary mapping joint names to angles.
                        If None, uses default pose with lowered base.
      asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset = env.scene[asset_cfg.name]

  # Get default state
  default_root_state = asset.data.default_root_state[env_ids].clone()
  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
  default_joint_vel = asset.data.default_joint_vel[env_ids].clone()

  # Modify root state to lower the robot (seated position)
  root_state = default_root_state.clone()
  root_state[:, 7:] = 0.0  # Zero velocities

  # Add env_origins offset
  root_state[:, 0:3] += env.scene.env_origins[env_ids]

  # Write root state
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)

  # Set joint positions
  joint_pos = default_joint_pos.clone()

  # Apply seated joint positions if provided
  if seated_joint_pos is not None:
    # Get all joint names from the asset
    joint_names = asset.joint_names

    # For each pattern in seated_joint_pos, find matching joints and set values
    for name_pattern, angle_value in seated_joint_pos.items():
      # Resolve which joints match this pattern
      matching_indices, _ = resolve_matching_names(name_pattern, joint_names)

      # Set the joint angles for all matching joints
      if len(matching_indices) > 0:
        for joint_idx in matching_indices:
          joint_pos[:, joint_idx] = angle_value

  joint_vel = torch.zeros_like(default_joint_vel)

  # Write joint state
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_target_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  min_height: float = 0.15,
  max_height: float = 0.19,
) -> None:
  """Randomize target height for each environment.

  Samples target heights uniformly from [min_height, max_height] and
  stores them in env.extras["target_height"].

  If curriculum is enabled, uses curriculum_height_min and curriculum_height_max
  from env.extras instead of the default parameters.

  Args:
      env: The environment.
      env_ids: Environment indices to randomize.
      min_height: Minimum target height in meters (used if no curriculum).
      max_height: Maximum target height in meters (used if no curriculum).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  # Initialize target_height buffer if it doesn't exist
  if "target_height" not in env.extras:
    env.extras["target_height"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )

  # Use curriculum values if available, otherwise use default parameters
  curr_min = env.extras.get("curriculum_height_min", min_height)
  curr_max = env.extras.get("curriculum_height_max", max_height)

  # Sample random heights for the specified environments
  num_resets = len(env_ids)
  random_heights = (
    torch.rand(num_resets, device=env.device) * (curr_max - curr_min) + curr_min
  )

  # Store in env.extras
  env.extras["target_height"][env_ids] = random_heights
