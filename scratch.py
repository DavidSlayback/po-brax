import brax
from brax import envs
import brax.jumpy as jp
# from IPython.display import HTML, Image  # Ipynb or Image render
from brax.envs.to_torch import JaxToTorchWrapper
# from brax.io import html, image  # Ipynb or image render
import torch
# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
import torch.nn as nn
import torch.nn.functional as F

ENV_NAME = 'ant'
B = 16
if __name__ == "__main__":
    base_ant = envs.create(ENV_NAME)
    state = base_ant.reset(rng=jp.random_prngkey(seed=0))
    # HTML(html.render(base_ant.sys, [state.qp]))
    # Image(image.render(base_ant.sys, [state.qp], 320, 240))
    e = envs.create_gym_env(ENV_NAME, B, 0, 'gpu', action_repeat=1)
    o = e.reset()