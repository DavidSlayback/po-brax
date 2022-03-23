import brax.envs.wrappers
import brax.jumpy as jp
from brax.envs.env import Wrapper, Env, State
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
