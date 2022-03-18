"""Trains an ant to "tag" a moving ball"""
from typing import Tuple
import brax
import jax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
from .more_jp import while_loop, meshgrid, index_add
from .utils import draw_arena
from google.protobuf import text_format


def extend_ant_cfg(cfg: str = brax.envs.ant._SYSTEM_CONFIG, cage_max_xy: jp.ndarray = jp.array([4.5, 4.5]), offset: float = 1) -> brax.Config:
    cfg = text_format.Parse(cfg, brax.Config())  # Get ant config
    ant_body_names = [b.name for b in cfg.bodies if b.name != 'Ground']
    # Add target
    target = cfg.bodies.add(name='Target', mass=1.)
    target.frozen.all = True
    sph = target.colliders.add().sphere
    sph.radius = 0.5
    # Add arena
    draw_arena(cfg, cage_max_xy[0] + offset, cage_max_xy[1] + offset, 0.5)
    for b in ant_body_names:
        cfg.collide_include.add(first=b, second='Arena')
    # print(cfg)
    return cfg


class AntTagEnv(env.Env):
    def __init__(self, **kwargs):
        # Preliminaries
        self.tag_radius = kwargs.get('tag_radius', 1.5)
        self.visible_radius = kwargs.get('visible_radius', 3.)
        self.target_step = kwargs.get('target_step', 0.5)
        self.min_spawn_distance = kwargs.get('min_spawn_distance', 5.)
        self.cage_x, self.cage_y = kwargs.get('cage_xy', (4.5, 4.5))
        self.cage_xy = jp.array((self.cage_x, self.cage_y))
        # See https://github.com/google/brax/issues/161
        # cfg = add_walls()
        cfg = extend_ant_cfg(cage_max_xy=self.cage_xy, offset=1.)
        self.sys = brax.System(cfg)
        # super().__init__(_SYSTEM_CONFIG)
        # Ant and target indexes
        self.target_idx = self.sys.body.index['Target']
        self.torso_idx = self.sys.body.index['$ Torso']
        self.ant_indices = jp.arange(self.torso_idx, self.target_idx)  # All parts of ant
        self.ant_l = self.ant_indices.shape[0]  # Number of parts that belong to ant
        self.ant_mg = tuple(meshgrid(self.ant_indices, jp.arange(0, 2)))  # Indices that correspond to x,y positions of all parts of ant

    def reset(self, rng: jp.ndarray) -> env.State:
        rng, rng1, rng2, rng3, rng4 = jp.random_split(rng, 5)
        # Initial joint and velocity positions
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        # initial ant position
        ant_pos = jp.random_uniform(rng3, (2,), -self.cage_xy, self.cage_xy)
        # Set default qp with the sampled joints
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        # Add ant xy to all ant part positions (otherwise they spring back hilariously)
        pos = index_add(qp.pos, self.ant_mg, ant_pos[..., None])
        # Sample random target position based on ant
        _, tgt = self._random_target(rng4, ant_pos)
        # Update ant position with this
        pos = jp.index_update(pos, self.target_idx, tgt)
        # Actually update qpos
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'hits': zero,
        }
        info = {'rng': rng}  # Save rng
        return env.State(qp, obs, reward, done, metrics, info)

    def _random_target(self, rng: jp.ndarray, ant_xy: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location at least min_spawn_location away from ant"""
        xy = jp.random_uniform(rng, (2,), -self.cage_xy, self.cage_xy)
        minus_ant = lambda xy: xy - ant_xy
        def resample(rngxy: Tuple[jp.ndarray, jp.ndarray]) -> Tuple[jp.ndarray, jp.ndarray]:
            rng, xy = rngxy
            _, rng1 = jp.random_split(rng, 2)
            xy = jp.random_uniform(rng1, (2,), -self.cage_xy, self.cage_xy)
            return rng1, xy

        _, xy = while_loop(lambda rngxy: jp.norm(minus_ant(rngxy[1])) <= self.min_spawn_distance,
                              resample,
                              (rng, xy))
        target_z = 0.5
        target = jp.array([*xy, target_z]).transpose()
        return rng, target

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        # Move target
        rng, tgt_pos = self._step_target(state.info['rng'], qp.pos[self.torso_idx, :2], qp.pos[self.target_idx, :2])
        pos = jp.index_update(qp.pos, self.target_idx, tgt_pos)
        qp = qp.replace(pos=pos)
        # Update rng
        state.info.update(rng=rng)
        # Get observation
        obs = self._get_obs(qp, info)
        # Done if we "tag"
        done = jp.where(jp.norm(qp.pos[self.torso_idx, :2] - qp.pos[self.target_idx, :2]) <= self.tag_radius, jp.float32(1), jp.float32(0))
        state.metrics.update(hits=done)
        # Reward is 1 for tag, 0 otherwise
        reward = jp.where(done > 0, jp.float32(1), jp.float32(0))
        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _step_target(self, rng: jp.ndarray, ant_xy: jp.ndarray, tgt_xy: jnp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Move target in 1 of 4 directions based on ant"""
        rng, rng1 = jp.random_split(rng, 2)
        choice = jax.random.randint(rng1, (), 0, 4)
        # Unit vector of target -> ant
        target2ant_vec = ant_xy - tgt_xy
        target2ant_vec = target2ant_vec / jp.norm(target2ant_vec)

        # Orthogonal vectors
        per_vec_1 = jp.multiply(target2ant_vec[::-1], jp.array([1., -1.]))
        per_vec_2 = jp.multiply(target2ant_vec[::-1], jp.array([-1., 1.]))
        # per_vec_2 = jp.array([-target2ant_vec[1], target2ant_vec[0]])
        opposite_vec = -target2ant_vec  # run away!

        vec_list = jp.stack([per_vec_1, per_vec_2, opposite_vec, jp.zeros(2)], 0)
        new_tgt_xy = vec_list[choice] * self.target_step + tgt_xy
        new_tgt_xy = jp.where((jp.abs(new_tgt_xy) > self.cage_xy).any(), tgt_xy, new_tgt_xy)
        return rng, jp.concatenate((new_tgt_xy, jp.ones(1)), 0)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe ant body position and velocities."""
        # Check if we can observe target. Otherwise just 0s
        target_xy = qp.pos[self.target_idx, :2]  # xy of target
        ant_xy = qp.pos[self.torso_idx, :2]  # xy of ant
        target_xy = jp.where(jp.norm(target_xy - ant_xy) <= self.visible_radius, target_xy, jp.zeros(2))

        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # XYZ of the torso (3,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        # target xy (2,)
        qpos = [qp.pos[0], qp.rot[0], joint_angle, target_xy]

        # qvel:
        # velcotiy of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities (8,)
        qvel = [qp.vel[0], qp.ang[0], joint_vel]

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * 10 bodies in the system
        # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
        # ignores
        cfrc = [
            jp.clip(info.contact.vel, -1, 1),
            jp.clip(info.contact.ang, -1, 1)
        ]
        # flatten bottom dimension
        cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
        # Target xy (if in range)

        return jp.concatenate(qpos + qvel + cfrc)


if __name__ == "__main__":
    e = AntTagEnv()
    from brax.envs.wrappers import EpisodeWrapper, VectorWrapper, AutoResetWrapper, VectorGymWrapper, GymWrapper
    from brax.io import html
    # e = AutoResetWrapper(VectorWrapper(EpisodeWrapper(e, 1000, 1), 16))
    e = AutoResetWrapper(EpisodeWrapper(e, 1000, 1))
    egym = GymWrapper(e, seed=0, backend='cpu')
    # egym = VectorGymWrapper(e, seed=0, backend='cpu')
    # egym = gym.wrappers.record_video.RecordVideo(egym, 'videos/', video_length=2)
    ogym = egym.reset()
    o = e.reset(jp.random_prngkey(0))
    # o2 = jax.jit(e.step)(o, jp.zeros((16, 8)))
    # for t in range(200):
    #     o2 = e.step(o2, jp.zeros((16, 8)))
    # for t in range(200):
    #     ogym2 = egym.step(jp.zeros((16,8)))
    for t in range(200):
        ogym2 = egym.step(egym.action_space.sample())
    print(3)
