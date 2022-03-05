From [here](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb)

qp_{t+1} = step(system, qp_t, act)

1. System is a static description of all the environment
2. qp is dynamic state at a timestep
3. act is the action

# brax.Config

- Protobuf thing, it's a [mess](https://github.com/google/brax/blob/main/brax/physics/config.proto)
  - Add things programmatically to bodies, then set their colliders, etc
  - Create joints with "parent" and "child", apply angle limit, child offset.z, rotation.z
  - can CopyFrom other config
  - Add to actuators, link to joint
  - 
# brax.QP

- pos (3d each body)
- vel (3d each body)
- rot (4d quat rotation about center of each body)
- ang (3d angular velocity about center of body)

# brax.System(Config)

- Mainly just connects functionality to the config object
- default_angle and default_qp
  - Look in config defaults for overrides, otherwise jitter angles within range
  - Same for positions

# ant_mountain
  - Made ant environment, grabbed config
  - Copied just the ant components (skipped ground), made "repeat" of these
  - added a default, added qps for each ant torso for all
  - Deleted collide_include so default behavior (all collisons) would apply

