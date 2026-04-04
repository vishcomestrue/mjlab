"""Getup task environment configuration.

This task trains quadrupeds to stand up from a seated position to a target
standing pose.
"""

from dataclasses import dataclass

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.getup import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


@dataclass
class GetupTaskCfg:
  """Configuration for getup task parameters."""

  # Target standing pose (joint angles relative to defaults)
  target_joint_pos: dict[str, float] | None = None
  # Initial seated pose (joint angles relative to defaults)
  seated_joint_pos: dict[str, float] | None = None
  # Target body height from ground (meters)
  target_body_height: float = 0.20
  # Minimum body height threshold for success
  min_body_height: float = 0.17
  # Maximum body tilt angle for stability (degrees)
  max_tilt_angle: float = 20.0
  # Episode timeout (seconds)
  episode_timeout: float = 20.0
  # Action scale
  action_scale: float = 0.25


def make_getup_env_cfg(
  robot_cfg: EntityCfg,
  robot_name: str,
  body_name: str,
  foot_site_names: list[str],
  task_cfg: GetupTaskCfg | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create getup environment configuration.

  Args:
      robot_cfg: Robot entity configuration.
      robot_name: Name identifier for the robot.
      body_name: Name of the main body/torso for height measurements.
      foot_site_names: List of foot site names for contact detection.
      task_cfg: Task-specific configuration parameters.
      play: If True, use play mode settings (longer episodes, no
          randomization).

  Returns:
      Complete environment configuration for the getup task.
  """
  if task_cfg is None:
    task_cfg = GetupTaskCfg()

  # ==================================================================
  # Scene Configuration
  # ==================================================================

  scene_cfg = SceneCfg(
    num_envs=1,
    extent=2.0,
    entities={"robot": robot_cfg},
    terrain=TerrainImporterCfg(
      terrain_type="plane",
    ),
  )

  viewer_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=body_name,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )

  sim_cfg = SimulationCfg(
    nconmax=35,
    njmax=1500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )

  # ==================================================================
  # Actions
  # ==================================================================

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=task_cfg.action_scale,
      use_default_offset=True,
    ),
  }

  # ==================================================================
  # Observations
  # ==================================================================

  actor_terms = {
    "body_height": ObservationTermCfg(func=mdp.body_height),
    "target_height": ObservationTermCfg(func=mdp.target_height),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if not play else None,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.5, n_max=0.5) if not play else None,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2) if not play else None,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5) if not play else None,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    # "target_joint_pos": ObservationTermCfg(func=mdp.target_joint_pos),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ==================================================================
  # Rewards
  # ==================================================================

  rewards = {
    # Primary rewards
    "height": RewardTermCfg(
      func=mdp.height_reward,
      weight=1.0,
      params={"target_height": None, "std": 0.1},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_reward,
      weight=3.0,
    ),
    # Regularization rewards
    "joint_symmetry": RewardTermCfg(
      func=mdp.joint_symmetry_reward,
      weight=1.0,
    ),
    "hip_stability": RewardTermCfg(
      func=mdp.hip_stability_reward,
      weight=0.5,
    ),
    # Penalties
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_penalty,
      weight=0.01,
    ),
    "dof_vel": RewardTermCfg(
      func=mdp.dof_vel_penalty,
      weight=0.001,
    ),
    "torques": RewardTermCfg(
      func=mdp.torques_penalty,
      weight=0.0001,
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_limits_penalty,
      weight=1.0,
    ),
  }

  # ==================================================================
  # Events
  # ==================================================================

  events = {
    "reset_to_seated": EventTermCfg(
      func=mdp.reset_to_seated_pose,
      mode="reset",
      params={"seated_joint_pos": task_cfg.seated_joint_pos},
    ),
    "randomize_target_height": EventTermCfg(
      func=mdp.randomize_target_height,
      mode="reset",
      params={"min_height": 0.15, "max_height": 0.23},
    ),
  }

  # Add random perturbations in training mode
  if not play:
    events["add_noise_to_pose"] = EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.05, 0.05),
      },
    )

  # ==================================================================
  # Terminations
  # ==================================================================

  terminations = {
    "timeout": TerminationTermCfg(
      func=envs_mdp.time_out,
      time_out=True,
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      time_out=False,
      params={"max_tilt_angle": task_cfg.max_tilt_angle},
    ),
  }

  # ==================================================================
  # Curriculum (optional - can be overridden by robot-specific configs)
  # ==================================================================

  curriculum = {
    "target_height": CurriculumTermCfg(
      func=mdp.target_height_curriculum,
      params={
        "height_stages": [
          {"step": 0, "min_height": 0.17, "max_height": 0.19},
          {"step": 1000 * 48, "min_height": 0.15, "max_height": 0.21},
          {"step": 2000 * 48, "min_height": 0.14, "max_height": 0.23},
        ],
      },
    ),
  }

  # ==================================================================
  # Environment Configuration
  # ==================================================================

  return ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    curriculum=curriculum,
    sim=sim_cfg,
    viewer=viewer_cfg,
    decimation=4,
    episode_length_s=(int(1e9) if play else task_cfg.episode_timeout),
  )
