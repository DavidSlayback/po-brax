"""Some example environments to help get started quickly with brax."""

import functools
from typing import Callable, Optional, Union, overload

from brax.envs import ant
from brax.envs import fast
from brax.envs import fetch
from brax.envs import grasp
from brax.envs import halfcheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import humanoid_standup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import reacher
from brax.envs import reacherangle
from brax.envs import ur5e
from brax.envs import walker2d
from brax.envs import wrappers as bwrappers
from .ant_tag import AntTagEnv
from .ant_heavenhell import AntHeavenHellEnv
from .ant_gather import AntGatherEnv
from .wrappers import VmapGymWrapper, AutoresetVmapGymWrapper, AutoresetGymWrapper
from brax.envs.env import Env
import gym

HAI_ACTION_REPEAT = 6
_envs = {
    'ant': ant.Ant,
    'ant_tag': AntTagEnv,
    'ant_heavenhell': AntHeavenHellEnv,
    'ant_gather': AntGatherEnv,
    'fast': fast.Fast,
    'fetch': fetch.Fetch,
    'grasp': grasp.Grasp,
    'halfcheetah': halfcheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoidstandup': humanoid_standup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'reacher': reacher.Reacher,
    'reacherangle': reacherangle.ReacherAngle,
    'ur5e': ur5e.Ur5e,
    'walker2d': walker2d.Walker2d,
}


def create(env_name: str,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           **kwargs) -> Env:
    """Creates an Env with a specified brax system."""
    env = _envs[env_name](**kwargs)
    if episode_length is not None:
        env = bwrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = bwrappers.VmapWrapper(env)
        # env = bwrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = bwrappers.AutoResetWrapper(env)

    return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env."""
    return functools.partial(create, env_name, **kwargs)


@overload
def create_gym_env(env_name: str,
                   batch_size: None = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.Env:
    ...


@overload
def create_gym_env(env_name: str,
                   batch_size: int,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.vector.VectorEnv:
    ...


def create_gym_env(env_name: str,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
    """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
    kwargs['auto_reset'] = False  # Use gym wrappers for autoreset
    environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
    if batch_size is None:
        return AutoresetGymWrapper(environment, seed=seed, backend=backend)
    if batch_size <= 0:
        raise ValueError(
            '`batch_size` should either be None or a positive integer.')
    return AutoresetVmapGymWrapper(environment, batch_size, seed=seed, backend=backend)
    # return VmapGymWrapper(environment, batch_size, seed=seed, backend=backend)
