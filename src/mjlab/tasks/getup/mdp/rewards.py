"""Getup task rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def height_reward(
  env: ManagerBasedRlEnv,
  target_height: float | None = None,
  std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for achieving target body height.

  Args:
      env: The environment.
      target_height: Target body height in meters. If None, reads from
          env.extras["target_height"] (per-environment targets).
      std: Standard deviation for Gaussian reward.
      asset_cfg: Asset configuration.

  Returns:
      Reward tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  current_height = asset.data.root_link_pos_w[:, 2]

  # Get target height (either fixed or per-env)
  if target_height is not None:
    # Fixed target for all environments
    target = target_height
  elif "target_height" in env.extras:
    # Per-environment targets
    target = env.extras["target_height"]
  else:
    # Fallback default
    target = 0.17

  height_error = torch.abs(current_height - target)
  return torch.exp(-height_error / std)


def upright_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for maintaining upright orientation.

  Uses exponential of squared error between gravity vector and ideal
  upright vector. When upright, projected_gravity_b = [0, 0, -1].

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Reward tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  # Ideal upright gravity vector in body frame
  up_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32)
  # Sum of squared differences
  error = torch.sum(torch.square(up_vec - asset.data.projected_gravity_b), dim=1)
  # Exponential reward
  return torch.exp(-2.0 * error)


def joint_symmetry_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for symmetric joint angles between left and right legs.

  Encourages FL ↔ FR and BL ↔ BR symmetry. For quadrupeds:
  - Hip joints: should be opposite (±hip for abduction/adduction)
  - Thigh/Calf joints: should be same (symmetric flexion)

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Reward tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos
  joint_names = asset.joint_names

  # Find joint indices
  def get_joint_idx(pattern: str) -> int:
    for i, name in enumerate(joint_names):
      if pattern in name:
        return i
    return -1

  # Get indices for each joint
  fl_hip = get_joint_idx("FL_hip")
  fr_hip = get_joint_idx("FR_hip")
  bl_hip = get_joint_idx("BL_hip")
  br_hip = get_joint_idx("BR_hip")

  fl_thigh = get_joint_idx("FL_thigh")
  fr_thigh = get_joint_idx("FR_thigh")
  bl_thigh = get_joint_idx("BL_thigh")
  br_thigh = get_joint_idx("BR_thigh")

  fl_calf = get_joint_idx("FL_calf")
  fr_calf = get_joint_idx("FR_calf")
  bl_calf = get_joint_idx("BL_calf")
  br_calf = get_joint_idx("BR_calf")

  # Calculate symmetry errors
  # Hip: same signs (FL_hip ≈ FR_hip) - abduction/adduction
  # Thigh/Calf: opposite signs due to mirrored joint mounting

  # Hip: FL_hip ≈ FR_hip (same sign)
  hip_error_front = torch.square(joint_pos[:, fl_hip] - joint_pos[:, fr_hip])
  hip_error_back = torch.square(joint_pos[:, bl_hip] - joint_pos[:, br_hip])

  # Thigh: FL_thigh ≈ -FR_thigh (opposite sign)
  thigh_error_front = torch.square(joint_pos[:, fl_thigh] + joint_pos[:, fr_thigh])
  thigh_error_back = torch.square(joint_pos[:, bl_thigh] + joint_pos[:, br_thigh])

  # Calf: FL_calf ≈ -FR_calf (opposite sign)
  calf_error_front = torch.square(joint_pos[:, fl_calf] + joint_pos[:, fr_calf])
  calf_error_back = torch.square(joint_pos[:, bl_calf] + joint_pos[:, br_calf])

  total_error = (
    hip_error_front
    + hip_error_back
    + thigh_error_front
    + thigh_error_back
    + calf_error_front
    + calf_error_back
  ) / 6.0  # Average error

  return torch.exp(-total_error)


def hip_stability_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping hip joints close to neutral (zero) position.

  Penalizes hip abduction/adduction, encouraging legs to stay vertical.

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Reward tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos
  joint_names = asset.joint_names

  # Find all hip joint indices
  hip_indices = [i for i, name in enumerate(joint_names) if "_hip_joint" in name]

  if not hip_indices:
    return torch.zeros(env.num_envs, device=env.device)

  # Sum of squared hip angles
  hip_positions = joint_pos[:, hip_indices]
  hip_error = torch.sum(torch.square(hip_positions), dim=1)

  return torch.exp(-hip_error)


def illegal_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize non-foot body parts contacting the ground.

  Returns a negative reward proportional to the number of non-foot
  geoms in contact with the terrain.

  Args:
      env: The environment.
      sensor_name: Name of the contact sensor tracking non-foot
          contacts.

  Returns:
      Penalty tensor of shape (num_envs,).
  """
  from mjlab.sensor import ContactSensor

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Count how many non-foot geoms are in contact
  return -torch.sum(sensor.data.found.float(), dim=-1)


def action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize rapid action changes.

  Args:
      env: The environment.

  Returns:
      Penalty tensor of shape (num_envs,).
  """
  return -torch.sum(
    (env.action_manager.action - env.action_manager.prev_action) ** 2,
    dim=1,
  )


def joint_limits_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Soft penalty for approaching joint limits.

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Penalty tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  soft_limits = asset.data.soft_joint_pos_limits
  if soft_limits is None:
    return torch.zeros(env.num_envs, device=env.device)

  joint_pos = asset.data.joint_pos
  # Penalty for exceeding soft limits
  out_of_limits = -(joint_pos - soft_limits[:, :, 0]).clip(max=0.0)
  out_of_limits += (joint_pos - soft_limits[:, :, 1]).clip(min=0.0)
  return -torch.sum(out_of_limits, dim=1)


def dof_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize large joint velocities for smooth motion.

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Penalty tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel

  # Squared joint velocities
  return -torch.sum(torch.square(joint_vel), dim=1)


def torques_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize large joint torques for energy efficiency.

  Uses L2 + L1 norm similar to Go1 implementation.

  Args:
      env: The environment.
      asset_cfg: Asset configuration.

  Returns:
      Penalty tensor of shape (num_envs,).
  """
  asset = env.scene[asset_cfg.name]

  # Get actuator forces (torques)
  if hasattr(asset.data, "applied_torque"):
    torques = asset.data.applied_torque
  else:
    # Fallback: estimate from action * stiffness
    return torch.zeros(env.num_envs, device=env.device)

  # L2 + L1 norm (similar to Go1)
  l2_norm = torch.sqrt(torch.sum(torch.square(torques), dim=1))
  l1_norm = torch.sum(torch.abs(torques), dim=1)

  return -(l2_norm + l1_norm)
