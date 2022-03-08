"""Trains an ant to go to heaven by following the advice of a priest"""
import brax
from brax import jumpy as jp
from brax.envs import env
from more_jp import meshgrid, index_add, choice, atleast_1d
from utils import draw_t_maze
from google.protobuf import text_format


def extend_ant_cfg(cfg: str = brax.envs.ant._SYSTEM_CONFIG, hhp: jp.ndarray = jp.array([[-6.25, 6.0], [6.25, 6.0], [0., 6.]]), hallway_width: float = 2) -> brax.Config:
    cfg = text_format.Parse(cfg, brax.Config())  # Get ant config
    ant_body_names = [b.name for b in cfg.bodies if b.name != 'Ground']
    # Add priest
    priest = cfg.bodies.add(name='Priest', mass=1.)
    priest.frozen.all = True
    sph = priest.colliders.add().sphere
    sph.radius = 0.5
    aqp = cfg.defaults.add().qps.add(name='Priest')  # Default priest position, never changes
    aqp.pos.x, aqp.pos.y, aqp.pos.z = hhp[-1, 0], hhp[-1, 1], 1.
    heaven = cfg.bodies.add(name='Target', mass=1.)
    heaven.frozen.all = True
    sph = heaven.colliders.add().sphere
    sph.radius = 0.5
    hell = cfg.bodies.add(name='Hell', mass=1.)
    hell.frozen.all = True
    sph = hell.colliders.add().sphere
    sph.radius = 0.5
    # Add walls
    # Rotate x for x walls, y for y walls
    draw_t_maze(cfg, t_x=hhp[:,0].max() + hallway_width / 2, t_y=hhp[:,1].max() + hallway_width / 2, hallway_width=hallway_width)
    for b in ant_body_names:
        cfg.collide_include.add(first=b, second='Arena')
    return cfg


class AntHeavenHellEnv(env.Env):
    def __init__(self, **kwargs):
        # Preliminaries
        self.heaven_hell_xy = jp.array(kwargs.get('heaven_hell', [[-6.25, 6.0], [6.25, 6.0]]))
        self.priest_pos = jp.array(kwargs.get('priest_pos', [0., 6.]))  # Priest is at 0,6 xy
        self._hhp = jp.concatenate((jp.concatenate((self.heaven_hell_xy, self.priest_pos[None, ...]), axis=0), jp.ones((3, 1))), axis=1)
        self.visible_radius = kwargs.get('visible_radius', 2.)  # Where can I see priest
        # See https://github.com/google/brax/issues/161
        cfg = extend_ant_cfg(hhp=self._hhp, hallway_width=2.)
        self.sys = brax.System(cfg)
        # super().__init__(_SYSTEM_CONFIG)
        # Ant and target indexes
        self.target_idx = self.sys.body.index['Target']
        self.hell_idx = self.sys.body.index['Hell']
        self.priest_idx = self.sys.body.index['Hell']
        self.torso_idx = self.sys.body.index['$ Torso']
        self.ant_indices = jp.arange(self.torso_idx, self.target_idx)  # All parts of ant
        self.ant_l = self.ant_indices.shape[0]
        self.ant_mg = tuple(meshgrid(self.ant_indices, jp.arange(0,2)))
        self._init_ant_pos = jp.array([[-0.5, 0.5], [0.5, 1.5]])  # Low and high xy for ant position

    def reset(self, rng: jp.ndarray) -> env.State:
        rng, rng1, rng2, rng3 = jp.random_split(rng, 4)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        ant_pos = jp.random_uniform(rng1, (2,), *self._init_ant_pos)  # Sample ant torso position
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        pos = index_add(qp.pos, self.ant_mg, ant_pos[...,None])
        target_pos, hell_pos = choice(rng3, self._hhp[:2], (2,), False)
        pos = jp.index_update(pos, jp.stack([self.target_idx, self.hell_idx]), jp.stack([target_pos, hell_pos]))
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info, jp.float32(0))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'heavens': zero,
            'hells': zero
        }
        info = {'rng': rng}
        return env.State(qp, obs, reward, done, metrics, info)


    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        # Done if we go to heaven or hell
        heaven_hell_priest = jp.stack([qp.pos[self.target_idx], qp.pos[self.hell_idx], qp.pos[self.priest_idx]])
        in_range = (jp.norm(heaven_hell_priest[:, :2] - qp.pos[self.torso_idx, :2], axis=-1) <= self.visible_radius)  # In range of pos1, pos2, priest
        priest_in_range = in_range[-1]
        reward = jp.where(in_range[0], jp.float32(1), jp.float32(0)) # +1 for heaven
        reward = jp.where(in_range[1], jp.float32(-1), reward)  # -1 for hell
        done = jp.where(reward != 0, jp.float32(1), jp.float32(0))  # Done at either position
        # Get observation
        obs = self._get_obs(qp, info, priest_in_range)
        state.metrics.update(hits=done)
        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_obs(self, qp: brax.QP, info: brax.Info, priest_in_range: jp.float32) -> jp.ndarray:
        """Observe ant body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
        tgt_x = atleast_1d(qp.pos[self.target_idx][0])
        heaven_direction = jp.where(priest_in_range > 0, jp.sign(tgt_x), jp.zeros_like(tgt_x))

        # qpos:
        # XYZ of the torso (3,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        # heaven direction (1,)
        qpos = [qp.pos[0], qp.rot[0], joint_angle, heaven_direction]

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
    # test = extend_ant_cfg()
    e = AntHeavenHellEnv()
    from brax.envs.wrappers import EpisodeWrapper, VectorWrapper, AutoResetWrapper, VectorGymWrapper, GymWrapper
    # e = AutoResetWrapper(VectorWrapper(EpisodeWrapper(e, 1000, 1), 16))
    e = AutoResetWrapper(EpisodeWrapper(e, 1000, 1))
    egym = GymWrapper(e, seed=0, backend='cpu')
    # egym = VectorGymWrapper(e, seed=0, backend='cpu')
    # egym = gym.wrappers.record_video.RecordVideo(egym, 'videos/', video_length=2)
    ogym = egym.reset()
    # o = e.reset(jp.random_prngkey(0))
    # o2 = jax.jit(e.step)(o, jp.zeros((16, 8)))
    # for t in range(200):
    #     o2 = e.step(o2, jp.zeros((16, 8)))
    # for t in range(200):
    #     ogym2 = egym.step(jp.zeros((16,8)))
    for t in range(200):
        ogym2 = egym.step(egym.action_space.sample())
    print(3)
