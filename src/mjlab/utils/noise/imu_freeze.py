"""Stateful IMU freeze/dropout/spike noise for sim2real IMU failure robustness."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class ImuFreezeGroupNoiseCfg:
  """Configuration for coordinated IMU freeze noise across multiple obs terms.

  Simulates BNO080 hard-reset: gravity, gyro, accel all freeze together for
  1-150 steps. Also models single-frame dropout and rare spike readings.
  All default parameters match observed real-robot failure statistics at 50Hz.
  """

  gravity_term: str = "projected_gravity"
  """Obs term name for projected gravity."""
  gyro_term: str = "base_ang_vel"
  """Obs term name for gyroscope (angular velocity)."""
  accel_term: str = "imu_lin_acc"
  """Obs term name for accelerometer (linear acceleration)."""

  # Long freeze: ~once per 7s at 50Hz (prob=0.003 → mean interval=333 steps=6.7s).
  freeze_prob: float = 0.003
  """Per-step probability of triggering a new freeze for a non-frozen env."""
  freeze_min_steps: int = 1
  """Minimum freeze duration in steps."""
  freeze_max_steps: int = 150
  """Maximum freeze duration in steps (uniform sampling)."""

  # Single-frame dropout: independent of long freeze.
  dropout_prob: float = 0.05
  """Per-step probability of a 1-frame dropout for a non-freeze env."""

  # Rare large spike on projected_gravity only (fresh frames only).
  spike_prob: float = 0.02
  """Per-step probability of a spike perturbation on projected_gravity."""
  spike_magnitude: float = 0.4
  """Half-range of uniform spike noise: uniform(-spike_magnitude, spike_magnitude)."""


class ImuFreezeGroupNoiseModel:
  """Stateful noise model applied to the full group_obs dict each step.

  Maintains per-env freeze counters and last-good buffers for all three
  IMU terms. Called once per step from compute_group() after per-term noise
  has already been applied to each term individually.

  Freeze/dropout replaces the fresh (already noisy) value with the held
  last-good value — the fresh noise is discarded on stale frames.
  Spike noise is added on top of fresh frames only.
  """

  def __init__(
    self,
    cfg: ImuFreezeGroupNoiseCfg,
    num_envs: int,
    device: str,
  ) -> None:
    self._cfg = cfg
    self._num_envs = num_envs
    self._device = device
    self._freeze_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._last_good: dict[str, torch.Tensor] = {}

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset freeze state for specified envs (called on episode reset)."""
    idx: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
    self._freeze_counter[idx] = 0
    # last_good refreshes automatically on the next non-frozen step.

  def __call__(
    self, group_obs: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    cfg = self._cfg
    n = self._num_envs
    dev = self._device

    imu_keys = [
      k for k in (cfg.gravity_term, cfg.gyro_term, cfg.accel_term)
      if k in group_obs
    ]
    if not imu_keys:
      return group_obs

    # Lazy-initialize last_good on first call (shape unknown until runtime).
    for k in imu_keys:
      if k not in self._last_good:
        self._last_good[k] = group_obs[k].clone()

    # --- Long freeze ---
    currently_frozen = self._freeze_counter > 0

    # Trigger new freezes for non-frozen envs only.
    trigger = (~currently_frozen) & (torch.rand(n, device=dev) < cfg.freeze_prob)
    new_duration = torch.randint(
      cfg.freeze_min_steps,
      cfg.freeze_max_steps + 1,
      (n,),
      device=dev,
      dtype=torch.long,
    )
    self._freeze_counter = torch.where(trigger, new_duration, self._freeze_counter)
    long_stale = currently_frozen | trigger

    # --- Single-frame dropout (independent of long freeze) ---
    dropout_stale = (~long_stale) & (torch.rand(n, device=dev) < cfg.dropout_prob)

    stale = long_stale | dropout_stale  # (num_envs,)

    # Update last_good for fresh envs BEFORE applying override.
    for k in imu_keys:
      obs = group_obs[k]
      fresh_2d = (~stale).unsqueeze(-1).expand_as(obs)
      self._last_good[k] = torch.where(fresh_2d, obs, self._last_good[k])

    # Override stale envs: replace fresh noisy value with held last-good value.
    for k in imu_keys:
      obs = group_obs[k]
      stale_2d = stale.unsqueeze(-1).expand_as(obs)
      group_obs[k] = torch.where(stale_2d, self._last_good[k], obs)

    # --- Spike on projected_gravity (fresh frames only, stacks on normal noise) ---
    if cfg.gravity_term in group_obs:
      g_obs = group_obs[cfg.gravity_term]
      spike_mask = (~stale) & (torch.rand(n, device=dev) < cfg.spike_prob)
      spike_noise = (
        torch.rand_like(g_obs) * (2.0 * cfg.spike_magnitude) - cfg.spike_magnitude
      )
      spike_2d = spike_mask.unsqueeze(-1).expand_as(g_obs)
      group_obs[cfg.gravity_term] = torch.where(spike_2d, g_obs + spike_noise, g_obs)

    # Decrement freeze counter (newly triggered envs go duration→duration-1).
    self._freeze_counter = torch.clamp(self._freeze_counter - 1, min=0)

    return group_obs
