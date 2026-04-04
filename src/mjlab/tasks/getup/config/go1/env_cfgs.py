"""Unitree GO1 specific configuration for getup task."""

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.tasks.getup.getup_env_cfg import (
  GetupTaskCfg,
  make_getup_env_cfg,
)


def unitree_go1_getup_env_cfg(play: bool = False):
  """Create getup environment configuration for Unitree GO1.

  Args:
      play: If True, use play mode settings.

  Returns:
      Complete environment configuration for GO1 getup task.
  """
  # Get GO1 robot configuration
  robot_cfg = get_go1_robot_cfg()

  # Define task-specific parameters
  task_cfg = GetupTaskCfg(
    # Target standing pose (use default GO1 standing configuration)
    target_joint_pos=None,  # None means use robot defaults
    # Initial seated pose (crouched position)
    # GO1 joint order: FR_hip, FR_thigh, FR_calf, FL_hip, ...
    seated_joint_pos={
      # Front Right leg (more bent)
      "FR_hip": 0.15,  # More abducted
      "FR_thigh": 1.2,  # More bent forward
      "FR_calf": -2.2,  # More bent back
      # Front Left leg
      "FL_hip": -0.15,
      "FL_thigh": 1.2,
      "FL_calf": -2.2,
      # Rear Right leg
      "RR_hip": 0.15,
      "RR_thigh": 1.2,
      "RR_calf": -2.2,
      # Rear Left leg
      "RL_hip": -0.15,
      "RL_thigh": 1.2,
      "RL_calf": -2.2,
    },
    # GO1 body height when standing is approximately 0.278m
    target_body_height=0.278,
    min_body_height=0.25,
    # GO1 is stable up to ~45 degree tilt
    max_tilt_angle=45.0,
    # Episode timeout
    episode_timeout=20.0,
    # Action scale
    action_scale=0.25,
  )

  # GO1 specific settings
  body_name = "trunk"
  foot_site_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

  return make_getup_env_cfg(
    robot_cfg=robot_cfg,
    robot_name="unitree_go1",
    body_name=body_name,
    foot_site_names=foot_site_names,
    task_cfg=task_cfg,
    play=play,
  )
