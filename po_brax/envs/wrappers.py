from typing import ClassVar, Optional

import brax.envs.wrappers
import brax.jumpy as jp
import gym
from brax.envs.env import Wrapper, Env, State
from gym import spaces
from gym.vector import utils
import jax

from po_brax.more_jp import cond

VectorWrapper = brax.envs.wrappers.VmapWrapper  # Need this version to do rng properly


class ActionRepeatWrapper(Wrapper):
    """Just change action duration (episode wrapper on top will ignore change in length of steps)"""

    def __init__(self, env: Env, action_repeat: int):
        super().__init__(env)
        if hasattr(self.unwrapped, 'sys'):
            self.unwrapped.sys.config.dt *= action_repeat
            self.unwrapped.sys.config.substeps *= action_repeat
        self.action_repeat = action_repeat


AutoResetWrapper = brax.envs.wrappers.AutoResetWrapper


class RandomizedAutoResetWrapperNaive(Wrapper):
    """Automatically resets Brax envs that are done.

    Force resample every step. Inefficient"""

    def step(self, state: State, action: jp.ndarray) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        maybe_reset = self.reset(state.info['rng'])

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        qp = jp.tree_map(where_done, maybe_reset.qp, state.qp)
        obs = where_done(maybe_reset.obs, state.obs)
        return state.replace(qp=qp, obs=obs)


class RandomizedAutoResetWrapperOnTerminal(Wrapper):
    """Automatically reset Brax envs that are done.

    Resample only when >=1 environment is actually done. Still resamples for all
    """

    def step(self, state: State, action: jp.ndarray) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        maybe_reset = cond(state.done.any(), self.reset, lambda rng: state, state.info['rng'])

        # maybe_reset = self.reset(state.info['rng'])

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        qp = jp.tree_map(where_done, maybe_reset.qp, state.qp)
        obs = where_done(maybe_reset.obs, state.obs)
        return state.replace(qp=qp, obs=obs)


class RandomizedAutoResetWrapperCached(Wrapper):
    """Automatically reset Brax envs that are done.

    Updates "first_qp" and "first_obs" every N environment steps
    """

    def __init__(self, env, n_steps_between_updates=200):
        super().__init__(env)
        self.n_steps_between_updates = n_steps_between_updates
        self.steps = 0

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info['first_qp'] = state.qp
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        self.steps += 1
        if (self.steps % self.n_steps_between_updates) == 0:  # Update our reset state
            rng, rng1 = jp.random_split(state.info['rng'], 2)
            s = self.env.reset(rng1)
            state.info['first_qp'] = s.qp
            state.info['first_obs'] = s.obs
            state.info['rng'] = rng
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        qp = jp.tree_map(where_done, state.info['first_qp'], state.qp)
        obs = where_done(state.info['first_obs'], state.obs)
        return state.replace(qp=qp, obs=obs)


class VmapGymWrapper(brax.envs.wrappers.VectorGymWrapper):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API.

    Modified to use VmapWrapper instead of VectorWrapper (needed for environments that
    use alternative autoreset wrappers)
    """
    def __init__(self,
                 env: Env,
                 batch_size: int,
                 seed: int = 0,
                 backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1 / self._env.sys.config.dt
        }

        self.num_envs = batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None

        obs_high = jp.inf * jp.ones(self._env.observation_size, dtype='float32')
        self.single_observation_space = spaces.Box(
            -obs_high, obs_high, dtype='float32')
        self.observation_space = utils.batch_space(self.single_observation_space,
                                                   self.num_envs)

        action_high = jp.ones(self._env.action_size, dtype='float32')
        self.single_action_space = spaces.Box(
            -action_high, action_high, dtype='float32')
        self.action_space = utils.batch_space(self.single_action_space,
                                              self.num_envs)

        def reset(key):
            keys = jp.random_split(key, self.num_envs + 1)
            key1, keys = keys[0], keys[1:]
            state = self._env.reset(keys)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            return state, state.obs, state.reward, state.done, state.metrics

        self._step = jax.jit(step, backend=self.backend)


class AutoresetGymWrapper(brax.envs.wrappers.GymWrapper):
    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        if done: self._state, obs, self._key = self._reset(self._key)
        return obs, reward, done, info


class AutoresetVmapGymWrapper(VmapGymWrapper):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API.

    Use stored gym key to reset environments where done. Call reset only when an environment is actually done
    """
    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        if done.any():
            new_state, new_obs, self._key = self._reset(self._key)  # Get new state (for each environment).
            # Update state's qp, obs, and info. Leave reward, done, metrics
            # qp = jp.where(done, new_state.qp, self._state.qp)
            # obs = jp.where(done, new_obs, obs)

            def where_done(x, y):
                done = self._state.done
                if done.shape:
                    done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
                return jp.where(done, x, y)
            qp = jp.tree_map(where_done, new_state.qp, self._state.qp)
            obs = where_done(new_obs, obs)
            self._state = self._state.replace(qp=qp, obs=obs)
        return obs, reward, done, info
