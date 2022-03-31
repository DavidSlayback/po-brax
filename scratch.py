import brax.jumpy as jp
import torch
# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
from brax.envs.wrappers import VmapWrapper
from po_brax.envs.wrappers import RandomizedAutoResetWrapperNaive
from po_brax.envs import AntGatherEnv
from po_brax.envs import create_gym_env
import jax

ENV_NAME = 'ant'
B = 16
T = 1000
if __name__ == "__main__":
    test = create_gym_env(ENV_NAME, B, 0, episode_length=20, eval_metrics=True, discount=0.99)
    o = test.reset()
    o2 = test.step(test.action_space.sample())
    for t in range(T):
        o2 = test.step(test.action_space.sample())
    print(test.get_stats())
    key = jax.random.PRNGKey(0)
    # e = envs.ant_heavenhell.AntHeavenHellEnv()
    # o = e.reset(key)
    # multi_key = jp.random_split(key, B)
    # # e = VmapWrapper(envs.ant_heavenhell.AntHeavenHellEnv())
    # e = VmapWrapper(AntGatherEnv())
    # e = RandomizedAutoResetWrapperNaive(e)
    # reset = jax.jit(e.reset)
    # step = jax.jit(e.step)
    # # o = e.reset(multi_key)
    # # key = jp.random_prngkey(0)
    # # e = envs.create(ENV_NAME, None, 1, False, batch_size=B)
    # # e = RandomizedAutoResetWrapper(e)
    # o = reset(multi_key)
    # for t in range(T):
    #     o = step(o, jp.zeros((B, 8)))
