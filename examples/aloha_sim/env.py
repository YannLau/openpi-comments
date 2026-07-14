import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from gym_aloha.constants import normalize_puppet_gripper_position
from gym_aloha.tasks.sim import BOX_POSE
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        box_pose: list[float] | None = None,
    ) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        self._box_pose = box_pose  # [x, y, z, qw, qx, qy, qz], None=random

        self._gym = gymnasium.make(task, obs_type=obs_type)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))

        # Override cube position for testing (if specified).
        # Default random range: x=[0, 0.2], y=[0.4, 0.6], z=[0.05].
        # Set box_pose=[x, y, z, qw, qx, qy, qz] to test far positions.
        if self._box_pose is not None:
            BOX_POSE[0] = np.asarray(self._box_pose, dtype=np.float64)
            physics = self._gym.unwrapped._env.physics
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            physics.forward()

            # Rebuild gym observation after moving the box
            qpos = physics.data.qpos.copy()
            agent_pos = np.concatenate([
                qpos[0:6],
                [normalize_puppet_gripper_position(qpos[6])],
                qpos[8:14],
                [normalize_puppet_gripper_position(qpos[14])],
            ])
            top_img = physics.render(height=480, width=640, camera_id="top")
            gym_obs = {"agent_pos": agent_pos, "pixels": {"top": top_img}}

        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }
