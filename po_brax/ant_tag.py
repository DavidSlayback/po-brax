"""Trains an ant to "tag" a moving ball"""
from typing import Tuple
import brax
import jax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp

class AntTagEnv(env.Env):
    def __init__(self, **kwargs):
        # Preliminaries
        self.tag_radius = kwargs.get('tag_radius', 1.5)
        self.visible_radius = kwargs.get('visible_radius', 3.)
        self.target_step = kwargs.get('target_step', 0.5)
        self.min_spawn_distance = kwargs.get('min_spawn_distance', 5.)
        self.cage_x, self.cage_y = kwargs.get('cage_xy', (4.5, 4.5))
        self.cage_xy = jp.array((self.cage_x, self.cage_y))
        # Modify base any environment config
        cfg = brax.envs.create('ant').sys.config
        # Target sphere
        target = cfg.bodies.add(name='Target', mass=1.)
        target.frozen.all = True
        sph = target.colliders.add().sphere
        sph.radius = 0.5
        super().__init__(_SYSTEM_CONFIG)
        # Ant and target indexes
        self.target_idx = self.sys.body.index['Target']
        self.torso_idx = self.sys.body.index['$ Torso']

    def reset(self, rng: jp.ndarray) -> env.State:
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -.1, .1)
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        ant = jp.index_update(qp.pos[self.torso_idx], jp.arange(0,2), jp.random_uniform(rng1, (2,), -self.cage_xy, self.cage_xy))
        rng, tgt = self._random_target(rng, ant[:2])
        pos = jp.index_update(qp.pos, jp.array((self.target_idx, self.torso_idx), dtype=jp.int32), jp.stack([tgt, ant], 0))
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'hits': zero,
        }
        info = {'rng': rng}
        return env.State(qp, obs, reward, done, metrics, info)

    def _random_target(self, rng: jp.ndarray, ant_xy: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location at least min_spawn_location away from ant"""
        rng, rng1 = jp.random_split(rng, 2)
        xy = jp.random_uniform(rng1, (2,), -self.cage_xy, self.cage_xy)
        # def sample(rng: jp.ndarray, xy: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        #     rng, rng1 = jp.random_split(rng, 2)
        #     xy = jp.random_uniform(rng1, (2,), -self.cage_xy, self.cage_xy)
        #     return rng, xy
        # jax.lax.while_loop(jp.norm(xy - ant_xy) <= self.min_spawn_distance, sample, (rng, xy))
        while jp.norm(xy - ant_xy) <= self.min_spawn_distance:
            rng, rng1 = jp.random_split(rng, 2)
            xy = jp.random_uniform(rng1, (2,), -self.cage_xy, self.cage_xy)
        target_z = 1.0
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
        """Move target in 1/4 directions based on ant"""
        rng, rng1 = jp.random_split(rng, 2)
        choice = jax.random.randint(rng1, (), 0, 4)
        target2ant_vec = ant_xy - tgt_xy
        target2ant_vec = target2ant_vec / jp.norm(target2ant_vec)

        per_vec_1 = jp.array([target2ant_vec[1], -target2ant_vec[0]])
        per_vec_2 = jp.array([-target2ant_vec[1], target2ant_vec[0]])
        opposite_vec = -target2ant_vec

        vec_list = [per_vec_1, per_vec_2, opposite_vec, jp.zeros(2)]
        chosen_vec = vec_list[choice] * self.target_step + tgt_xy
        chosen_vec = jp.where((jp.abs(chosen_vec) > self.cage_xy).any(), tgt_xy, chosen_vec)
        return rng, jp.concatenate((chosen_vec, jp.ones(1)), 0)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe ant body position and velocities."""
        # Check if we can observe target. Otherwise just 0s
        target_xy = qp.pos[self.target_idx, :2]  # xy of target
        ant_xy = qp.pos[self.torso_idx, :2] # xy of ant
        if jp.norm(target_xy - ant_xy) <= self.visible_radius: target_xy[:] = jp.zeros(2)

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

_SYSTEM_CONFIG = """
bodies {
  name: "$ Torso"
  colliders {
    capsule {
      radius: 0.25
      length: 0.5
      end: 1
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "Aux 1"
  colliders {
    rotation {
      x: 90.0
      y: -45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.4428427219390869
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "$ Body 4"
  colliders {
    rotation {
      x: 90.0
      y: -45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.7256854176521301
      end: -1
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "Aux 2"
  colliders {
    rotation {
      x: 90.0
      y: 45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.4428427219390869
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "$ Body 7"
  colliders {
    rotation {
      x: 90.0
      y: 45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.7256854176521301
      end: -1
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "Aux 3"
  colliders {
    rotation {
      x: -90.0
      y: 45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.4428427219390869
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "$ Body 10"
  colliders {
    rotation {
      x: -90.0
      y: 45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.7256854176521301
      end: -1
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "Aux 4"
  colliders {
    rotation {
      x: -90.0
      y: -45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.4428427219390869
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "$ Body 13"
  colliders {
    rotation {
      x: -90.0
      y: -45.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.7256854176521301
      end: -1
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "Ground"
  colliders {
    plane {
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
bodies {
  name: "Target"
  colliders {
    sphere {
      radius: 0.5
    }
    material {
      friction: 1.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
joints {
  name: "$ Torso_Aux 1"
  stiffness: 18000.0
  parent: "$ Torso"
  child: "Aux 1"
  parent_offset {
    x: 0.20000000298023224
    y: 0.20000000298023224
  }
  child_offset {
    x: -0.10000000149011612
    y: -0.10000000149011612
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -30.0
    max: 30.0
  }
  spring_damping: 80.0
}
joints {
  name: "Aux 1_$ Body 4"
  stiffness: 18000.0
  parent: "Aux 1"
  child: "$ Body 4"
  parent_offset {
    x: 0.10000000149011612
    y: 0.10000000149011612
  }
  child_offset {
    x: -0.20000000298023224
    y: -0.20000000298023224
  }
  rotation {
    z: 135.0
  }
  angular_damping: 20.0
  angle_limit {
    min: 30.0
    max: 70.0
  }
  spring_damping: 80.0
}
joints {
  name: "$ Torso_Aux 2"
  stiffness: 18000.0
  parent: "$ Torso"
  child: "Aux 2"
  parent_offset {
    x: -0.20000000298023224
    y: 0.20000000298023224
  }
  child_offset {
    x: 0.10000000149011612
    y: -0.10000000149011612
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -30.0
    max: 30.0
  }
  spring_damping: 80.0
}
joints {
  name: "Aux 2_$ Body 7"
  stiffness: 18000.0
  parent: "Aux 2"
  child: "$ Body 7"
  parent_offset {
    x: -0.10000000149011612
    y: 0.10000000149011612
  }
  child_offset {
    x: 0.20000000298023224
    y: -0.20000000298023224
  }
  rotation {
    z: 45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -70.0
    max: -30.0
  }
  spring_damping: 80.0
}
joints {
  name: "$ Torso_Aux 3"
  stiffness: 18000.0
  parent: "$ Torso"
  child: "Aux 3"
  parent_offset {
    x: -0.20000000298023224
    y: -0.20000000298023224
  }
  child_offset {
    x: 0.10000000149011612
    y: 0.10000000149011612
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -30.0
    max: 30.0
  }
  spring_damping: 80.0
}
joints {
  name: "Aux 3_$ Body 10"
  stiffness: 18000.0
  parent: "Aux 3"
  child: "$ Body 10"
  parent_offset {
    x: -0.10000000149011612
    y: -0.10000000149011612
  }
  child_offset {
    x: 0.20000000298023224
    y: 0.20000000298023224
  }
  rotation {
    z: 135.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -70.0
    max: -30.0
  }
  spring_damping: 80.0
}
joints {
  name: "$ Torso_Aux 4"
  stiffness: 18000.0
  parent: "$ Torso"
  child: "Aux 4"
  parent_offset {
    x: 0.20000000298023224
    y: -0.20000000298023224
  }
  child_offset {
    x: -0.10000000149011612
    y: 0.10000000149011612
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -30.0
    max: 30.0
  }
  spring_damping: 80.0
}
joints {
  name: "Aux 4_$ Body 13"
  stiffness: 18000.0
  parent: "Aux 4"
  child: "$ Body 13"
  parent_offset {
    x: 0.10000000149011612
    y: -0.10000000149011612
  }
  child_offset {
    x: -0.20000000298023224
    y: 0.20000000298023224
  }
  rotation {
    z: 45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: 30.0
    max: 70.0
  }
  spring_damping: 80.0
}
actuators {
  name: "$ Torso_Aux 1"
  joint: "$ Torso_Aux 1"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "Aux 1_$ Body 4"
  joint: "Aux 1_$ Body 4"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "$ Torso_Aux 2"
  joint: "$ Torso_Aux 2"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "Aux 2_$ Body 7"
  joint: "Aux 2_$ Body 7"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "$ Torso_Aux 3"
  joint: "$ Torso_Aux 3"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "Aux 3_$ Body 10"
  joint: "Aux 3_$ Body 10"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "$ Torso_Aux 4"
  joint: "$ Torso_Aux 4"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "Aux 4_$ Body 13"
  joint: "Aux 4_$ Body 13"
  strength: 350.0
  torque {
  }
}
friction: 1.0
gravity {
  z: -9.800000190734863
}
angular_damping: -0.05000000074505806
baumgarte_erp: 0.10000000149011612
collide_include {
  first: "$ Torso"
  second: "Ground"
}
collide_include {
  first: "$ Body 4"
  second: "Ground"
}
collide_include {
  first: "$ Body 7"
  second: "Ground"
}
collide_include {
  first: "$ Body 10"
  second: "Ground"
}
collide_include {
  first: "$ Body 13"
  second: "Ground"
}
dt: 0.05000000074505806
substeps: 10
"""
if __name__ == "__main__":
    e = AntTagEnv()
    from brax.envs.wrappers import EpisodeWrapper, VectorWrapper, AutoResetWrapper, VectorGymWrapper
    e = AutoResetWrapper(VectorWrapper(EpisodeWrapper(e, 1000, 1), 16))
    e2 = VectorGymWrapper(e, seed=0, backend='gpu')
    o = e2.reset()
    # o = e.reset(jp.random_prngkey(0))
    o2 = e.step(o, jp.zeros((16, 8)))
    o2 = e2.step(o, jp.zeros((16,8)))
    print(3)
