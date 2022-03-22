"""Trains an ant to gather apples and avoid bombs

See: https://github.com/rll/rllab/blob/master/rllab/envs/mujoco/gather/gather_env.py
"""
from typing import Tuple, Sequence
import brax
from brax import math as math
import jax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
from .more_jp import meshgrid, choice
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

def extend_ant_cfg(cfg: str = brax.envs.ant._SYSTEM_CONFIG,
                   cage_max_xy: jp.ndarray = jp.array([4.5, 4.5]),
                   offset: float = 1,
                   n_apples: int = 8,
                   n_bombs: int = 8) -> brax.Config:
    cfg = text_format.Parse(cfg, brax.Config())  # Get ant config
    ant_body_names = [b.name for b in cfg.bodies if b.name != 'Ground']  # Find ant components
    # Add arena
    draw_arena(cfg, cage_max_xy[0] + offset, cage_max_xy[1] + offset, 0.5)
    for b in ant_body_names:
        cfg.collide_include.add(first=b, second='Arena')
    # Add apples and bombs. All frozen, non-collidable objects, all starting in same spot (actual spot determined on reset)
    for i in range(n_apples):
        apple = cfg.bodies.add(name=f'Target_{i+1}', mass=1.)
        apple.frozen.all = True
        sph = apple.colliders.add().sphere
        sph.radius = 0.25
    for i in range(n_bombs):
        bomb = cfg.bodies.add(name=f'Bomb_{i+1}', mass=1.)
        bomb.frozen.all = True
        sph = bomb.colliders.add().sphere
        sph.radius = 0.25
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
        sensor_span: Arc (in radians) of ant sensors
        dying_cost: Cost for death (undoable locomotion error)

    Apples and bombs spawn at any integer grid location within cage_xy, except those too close to origin
    Ant gets its standard observations, plus:
      n_bins apple readings and n_bins bomb readings
    """
    def __init__(self,
                 n_apples: int = 8,
                 n_bombs: int = 8,
                 cage_xy: Sequence[float] = (6, 6),
                 robot_object_spacing: float = 2.,
                 catch_range: float = 1.,
                 n_bins: int = 10,
                 sensor_range: float = 6.,
                 sensor_span: float = 2 * jp.pi,
                 dying_cost: float = -10.,
                 **kwargs
                 ):
        self.cage_xy = jp.array(cage_xy)
        cfg = extend_ant_cfg(cage_max_xy=self.cage_xy, offset=1., n_apples=n_apples, n_bombs=n_bombs)  # Add walls, apples, and bombs
        self.sys = brax.System(cfg)
        # super().__init__(_SYSTEM_CONFIG)
        # Ant and target indexes
        self.torso_idx = self.sys.body.index['$ Torso']  # Ant always starting in small jitter range at 0
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.n_objects = n_apples + n_bombs
        self.n_bins = n_bins
        self.dying_cost = dying_cost
        self.sensor_range = sensor_range
        self.half_span = sensor_span / 2
        self.catch_range = catch_range
        last_ind = self.sys.num_bodies; first_ind = last_ind - (self.n_objects)
        self.object_indices = jp.arange(first_ind, last_ind)  # Indices for apples and bombs
        # Find all integer locations at least robot_object_spacing away from ant spawn position
        possible_grid_positions = jp.stack([g.ravel() for g in meshgrid(jp.arange(-self.cage_xy[0], self.cage_xy[0]+1), jp.arange(-self.cage_xy[1], self.cage_xy[1]+1))], axis=1)
        self.possible_grid_positions = jp.stack([g for g in possible_grid_positions if jp.norm(g) > robot_object_spacing], axis=0)
        self.possible_grid_positions = jp.concatenate([self.possible_grid_positions, jp.zeros((self.possible_grid_positions.shape[0], 1))], axis=1)
        self.waiting_area = self.possible_grid_positions[-1] + self.sensor_range * 2  # Stick captured objects somewhere else

    def reset(self, rng: jp.ndarray) -> env.State:
        qp = self.sample_init_qp(rng)
        info = self.sys.info(qp)
        distances = jp.norm(qp.pos[self.torso_idx][:2] - qp.pos[self.object_indices][..., :2],
                            axis=1)  # Distances to all objects
        obs = self._get_obs(qp, info, distances)
        reward, done, zero = jp.zeros(3)
        # Use metrics to track apples and bombs, determine termination
        metrics = {
            'apples': zero,
            'bombs': zero,
            'objects': zero,
        }
        info = {'rng': rng}  # Save rng
        return env.State(qp, obs, reward, done, metrics, info)

    def sample_init_qp(self, rng: jp.ndarray) -> brax.QP:
        rng, rng1, rng2, rng3 = jp.random_split(rng, 4)
        # Initial joint and velocity positions
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        # Set default qp with the sampled joints
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        # Sample object positions
        object_pos = choice(rng3, self.possible_grid_positions, (self.n_objects,), replace=False)
        # apple_pos, bomb_pos = object_pos[:self.n_apples], object_pos[self.n_apples:]
        # Update object positions
        pos = jp.index_update(qp.pos, self.object_indices, object_pos)
        return qp.replace(pos=pos)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        distances = jp.norm(qp.pos[self.torso_idx][:2] - qp.pos[self.object_indices][..., :2],
                            axis=1)  # Distances to all objects
        # Get observation
        obs = self._get_obs(qp, info, distances)
        # "Death" and associated rewards
        done = jp.where(qp.pos[self.torso_idx, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
        done = jp.where(qp.pos[self.torso_idx, 2] > 1.0, x=jp.float32(1), y=done)
        reward = jp.where(done > 0, jp.float32(self.dying_cost), jp.float32(0))
        # Rewards for apples and bombs
        in_range = distances <= self.catch_range
        # Move objects we hit to the waiting area
        tgt_pos = jp.where(in_range[:, None], self.waiting_area, qp.pos[self.object_indices])
        qp = qp.replace(pos=jp.index_update(qp.pos, self.object_indices, tgt_pos))

        in_range_apple, in_range_bomb = in_range[:self.n_apples], in_range[self.n_apples:]
        reward = jp.where(in_range_apple.any() & (done == 0), jp.float32(1), reward)
        reward = jp.where(in_range_bomb.any() & (done == 0), jp.float32(-1), reward)
        # Done if we hit all objects
        done = jp.where((qp.pos[self.object_indices] == self.waiting_area).all(), jp.float32(1),done)
        apples_hit, bombs_hit = in_range_apple.sum(), in_range_bomb.sum()
        state.metrics.update(apples=apples_hit, bombs=bombs_hit)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_readings(self, qp: brax.QP, distances: jp.ndarray) -> jp.ndarray:
        """Get sensor readings for ant

        Get ant
          ori = [0, 1, 0, 0]
          rot = ant_quat
          ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]
          ori = atan2(ori[1], ori[0])
        Split ant sensor span into n_bins
        For each bin, get only closest of each object type (apple or bomb)
        """
        readings = jnp.zeros(self.n_bins * 2)
        bin_res = (2 * self.half_span) / self.n_bins  # FOV for each bin
        ant_orientation = qp.rot[self.torso_idx]  # Quaternion orientation
        ori = jp.array([0,1,0,0])
        ori = math.quat_mul(math.quat_mul(ant_orientation, ori), math.quat_inv(ant_orientation))[1:3]
        ori = jp.arctan2(ori[1], ori[0])  # Projected into x-y plane
        object_xy = qp.pos[self.object_indices][..., :2]
        angles = jp.arctan2(object_xy[...,0], object_xy[...,1]) - ori  # Angle from ant face to all objects (-pi to pi)
        in_range = distances <= self.sensor_range
        # Sensor bin for each object (apples then bombs) (-1 of out of range/span) (nobjects,)
        object_bins = jp.where(jp.logical_and(jp.abs(angles) <= self.half_span, in_range)
                               , ((angles + self.half_span) / bin_res).astype(int), jp.int32(-1))
        bomb_bins = jp.where(object_bins[self.n_apples:] >= 0, object_bins[self.n_apples:] + self.n_apples, -1)
        object_bins = jp.index_update(object_bins, jp.arange(self.n_apples, self.n_objects), bomb_bins)
        object_intensities = jp.where(object_bins >= 0, 1. - (distances / self.sensor_range), jp.float32(0))
        readings = jp.index_update(readings, object_bins, object_intensities)
        # sorted_indices = object_bins.argsort()  # Sort so that -1 is all at beginning
        # TODO: Not quite right. This doesn't guarantee the closest reading, it just guarantees *a* reading
        return readings

    def _get_obs(self, qp: brax.QP, info: brax, distances: jp.ndarray) -> jp.ndarray:
        """Observe ant body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # XYZ of the torso (3,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        qpos = [qp.pos[0], qp.rot[0], joint_angle]

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
        # Sensor readings
        readings = [self._get_readings(qp, distances)]

        return jp.concatenate(qpos + qvel + cfrc + readings)


if __name__ == "__main__":
    e = AntGatherEnv()
    from brax.envs.wrappers import EpisodeWrapper, VectorWrapper, AutoResetWrapper, VectorGymWrapper, GymWrapper
    from brax.io import html
    # e = AutoResetWrapper(VectorWrapper(EpisodeWrapper(e, 1000, 1), 16))
    e = AutoResetWrapper(EpisodeWrapper(e, 1000, 1))
    egym = GymWrapper(e, seed=0, backend='gpu')
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
