"""Curriculum learning functions for getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HeightStage(TypedDict):
  """Stage definition for target height curriculum."""

  step: int
  min_height: float
  max_height: float


def target_height_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  height_stages: list[HeightStage],
) -> dict[str, torch.Tensor]:
  """Curriculum for gradually increasing target standing height.

  Starts with easier (lower) target heights and progressively increases
  to harder (higher) target heights as training progresses.

  Args:
      env: The environment instance.
      env_ids: Environment indices (unused, kept for API compatibility).
      height_stages: List of stages with step thresholds and height ranges.
          Each stage should have:
          - step: Training step when this stage activates
          - min_height: Minimum target height for this stage (meters)
          - max_height: Maximum target height for this stage (meters)

  Returns:
      Dictionary with current min and max target heights for logging.

  Example:
      height_stages = [
          {"step": 0, "min_height": 0.15, "max_height": 0.16},
          {"step": 5000 * 48, "min_height": 0.16, "max_height": 0.17},
          {"step": 10000 * 48, "min_height": 0.17, "max_height": 0.18},
      ]
  """
  del env_ids  # Unused, kept for API compatibility

  # Initialize curriculum state in env.extras if not present
  if "curriculum_height_min" not in env.extras:
    env.extras["curriculum_height_min"] = height_stages[0]["min_height"]
    env.extras["curriculum_height_max"] = height_stages[0]["max_height"]

  # Update height range based on current training step
  current_step = env.common_step_counter
  for stage in height_stages:
    if current_step >= stage["step"]:
      env.extras["curriculum_height_min"] = stage["min_height"]
      env.extras["curriculum_height_max"] = stage["max_height"]

  # Return current values for logging
  return {
    "min_height": torch.tensor([env.extras["curriculum_height_min"]]),
    "max_height": torch.tensor([env.extras["curriculum_height_max"]]),
  }
