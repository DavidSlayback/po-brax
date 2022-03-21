import brax
import brax.jumpy as jp
import torch
# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
import torch.nn as nn
import torch.nn.functional as F
import po_brax as envs
from brax.envs.wrappers import VmapWrapper
from po_brax.wrappers import RandomizedAutoResetWrapper, ActionRepeatWrapper
import jax

ENV_NAME = 'ant_heavenhell'
B = 16
T = 200
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # e = envs.ant_heavenhell.AntHeavenHellEnv()
    # o = e.reset(key)
    multi_key = jp.random_split(key, B)
    e = VmapWrapper(envs.ant_heavenhell.AntHeavenHellEnv())
    e = RandomizedAutoResetWrapper(e)
    reset = jax.jit(e.reset)
    step = jax.jit(e.step)
    # o = e.reset(multi_key)
    # key = jp.random_prngkey(0)
    # e = envs.create(ENV_NAME, None, 1, False, batch_size=B)
    # e = RandomizedAutoResetWrapper(e)
    o = reset(multi_key)
    for t in range(T):
        o = step(o, jp.zeros((B, 8)))
