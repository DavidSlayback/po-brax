"""Trains an ant to gather apples and avoid bombs

See: https://github.com/rll/rllab/blob/master/rllab/envs/mujoco/gather/gather_env.py
"""
from typing import Tuple, Sequence
import brax
import jax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
from .more_jp import while_loop, meshgrid, index_add
from .utils import draw_arena
from google.protobuf import text_format

"""
ORI_IDX = 6 for ant

    def reset(self, also_wrapped=True):
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        if also_wrapped:
            self.wrapped_env.reset()
        return self.get_current_obs()
        
    def step(self, action):
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew
        info['outer_rew'] = 0
        if done:
            return Step(self.get_current_obs(), self.dying_cost, done, **info)  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                    info['outer_rew'] = 1
                else:
                    reward = reward - 1
                    info['outer_rew'] = -1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return Step(self.get_current_obs(), reward, done, **info)

    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()  # overwrite this for Ant!

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb; ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])
"""

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


class AntGatherEnv(env.Env):
    """
    Args:
        n_apples: Number of apples in environment (+1 reward each)
        n_bombs: Number of bombs in environment  (-1 reward each)
        cage_xy: Max x and y values of arena (box from (-x,-y) to (x,y))
        robot_object_spacing: Minimum spawn distance of objects from ant initial position
        catch_range: Distance at which robot "catches" apple or bomb
        n_bins: Resolution of ant sensor. If multiple objects are in same bin span, only closest is seen
        sensor_range: Range of ant sensors
        sensor_span: Arc (in degrees) of ant sensors
        dying_cost: Cost for death (undoable locomotion error)

    Apples and bombs spawn at any integer grid location within cage_xy, except those too close to origin
    Ant gets its standard observations, plus:
      n_bins apple readings and n_bins bomb readings
    """
    def __init__(self,
                 n_apples: int = 8,
                 n_bombs: int = 8,
                 cage_xy: Sequence[float, float] = (6, 6),
                 robot_object_spacing: float = 2.,
                 catch_range: float = 1.,
                 n_bins: int = 10,
                 sensor_range: float = 6.,
                 sensor_span: float = 180,
                 dying_cost: float = -10.,
                 **kwargs
                 ):
        self.cage_xy = jp.array(cage_xy)
        cfg = extend_ant_cfg(cage_max_xy=self.cage_xy, offset=1.)  # Add walls
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
