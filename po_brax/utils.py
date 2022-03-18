import brax
import brax.jumpy as jp
from brax.physics.config_pb2 import Body


def add_box_wall_to_body(body: Body, from_xy: jp.ndarray, to_xy: jp.ndarray, half_height: float = 0.5, wall_width: float = 0.25) -> None:
    """Add a box wall collider to a body

    Args:
        body: Body contained from a config_pb2 object. Assume body handles z-axis
        from_xy: xy coordinates of start of capsule (relative to body)
        to_xy: xy coordinates of end of capsule (relative to body)
        half_height: Half height of box

    Returns:
        Nothing
    """
    unit_vector = jp.zeros_like(from_xy); unit_vector = jp.index_update(unit_vector, jp.arange(0, 1), jp.ones(1))  # x unit vector
    vector = to_xy - from_xy  # Used for angle and length
    length = jp.norm(vector)
    midpoint = (from_xy + to_xy) / 2  # xy midpoint (position)
    # Dot product for vector rotation (a dot b = |a| * |b| * cos theta), convert to degrees
    z_rotation = (jp.arccos(jp.dot(unit_vector, vector) / length) * 180 / jp.pi)
    coll = body.colliders.add()  # Add collider (for position and rotation)
    coll.position.x, coll.position.y = midpoint
    coll.rotation.z = z_rotation  # W.r.t. unit x-vector
    box = coll.box  # Actual box object
    box.halfsize.x, box.halfsize.y, box.halfsize.z = length, wall_width, half_height


def add_capsule_wall_to_body(body: Body, from_xy: jp.ndarray, to_xy: jp.ndarray, radius: float = 0.5, include_radius: bool = False) -> None:
    """Add a capsule wall collider to a body

    Note: currently only support horizontal and vertical

    Args:
        body: Body contained from a config_pb2 object
        from_xy: xy coordinates of start of capsule (relative to body)
        to_xy: xy coordinates of end of capsule (relative to body)
        radius: Radius of capsule
        include_radius: If true, include capsule radius in capsule length calculation (shrink by radius * 2)
    Returns:
        Nothing. Body is modified in place anyway
    """
    length = jp.norm(from_xy - to_xy) - (include_radius * radius)  # Capsule length
    cap_xy = (from_xy + to_xy) / 2  # Capsule xy position
    # Need capsule rotation
    # A vertical (same x) capsule has rotation.x = 90. Goes from up in z to up in x
    # A horizontal (same y) capsule has rotation.y = 90. Goes from up in z to up in y
    assert (from_xy[0] == to_xy[0]) or (from_xy[1] == to_xy[1])
    coll = body.colliders.add()  # Add collider (for position and rotation)
    vertical = (from_xy[0] == to_xy[0])  # Vertical walls (y to y), otherwise horizontal (x to x)
    coll.position.x = cap_xy[0]; coll.position.y = cap_xy[1]
    if vertical: coll.rotation.x = 90
    else: coll.rotation.y = 90
    cap = coll.capsule  # Actual capsule object
    cap.radius = radius; cap.length = length


def draw_arena(cfg: brax.Config, cage_x: float, cage_y: float, capsule_radius_or_box_half_height: float = 0.5, arena_name: str = "Arena", use_boxes: bool = True) -> None:
    """Add frozen 4-sided arena using capsule walls to enforce bounds of play

    Arranged such that cage_x and cage_y are the bounds of the inner area (i.e., at radius edge of capsule facing inward)
    Defines a rectangle from [-cage_x - rad, -cage_y - rad] to [cage_x + rad, cage_y + rad]. Determine additional space needs elsewhere!
    Args:
        cfg: brax Config object
        cage_x: Max x size
        cage_y: Max y size
        capsule_radius_or_box_half_height: thickness of wall (capsule) or half height of wall (box, is 2x thickness). >= 0.5 recommended
        arena_name: Name given to arena (used to include collide pairs later)
        use_boxes:
    Returns:
        Nothing, in-place
    """
    x, y, r = cage_x, cage_y, capsule_radius_or_box_half_height
    arena = cfg.bodies.add(name=arena_name, mass=1.)  # 1 frozen body, many colliders
    arena.frozen.all = True
    aqp = cfg.defaults.add().qps.add(name=arena_name)  # Default height such that walls just touch the ground
    aqp.pos.z = capsule_radius_or_box_half_height
    if use_boxes: r /= 2  # Wall halfsize, expand coordinates so that we *enclose* this space
    xy_positions = jp.array([[x + r, y + r], [x + r, -y - r], [-x - r, -y - r], [-x - r, y + r]])
    for i in range(len(xy_positions)):
        add_capsule_wall_to_body(arena, xy_positions[i], xy_positions[int((i+1) % 4)], r, True) if not use_boxes else add_box_wall_to_body(arena, xy_positions[i], xy_positions[int((i+1) % 4)], capsule_radius_or_box_half_height, r)



def draw_t_maze(cfg: brax.Config, t_x: float, t_y: float, hallway_width: float = 2., capsule_radius: float = 0.5, arena_name: str = "Arena") -> None:
    """Draw a T (like in TMaze or heaven hell)

    Arranged such that cage_x and cage_y are the bounds of the inner area (i.e., at radius edge of capsule facing inward)
    Defines a rectangle from [-cage_x - rad, -cage_y - rad] to [cage_x + rad, cage_y + rad]. Determine additional space needs elsewhere!
    Args:
        cfg: brax Config object
        t_x: Rightmost x coordinate of top of T
        t_y: Top of T y coordinate
        hallway_width: Uniform width within T
        capsule_radius: thickness of wall. >=0.5 recommended
        arena_name: Name given to arena (used to include collide pairs later
    Returns:
        Nothing, in-place
    """
    r = capsule_radius
    arena = cfg.bodies.add(name=arena_name, mass=1.)  # 1 frozen body, many colliders
    arena.frozen.all = True
    aqp = cfg.defaults.add().qps.add(name=arena_name)  # Default height such that walls just touch the ground
    aqp.pos.z = capsule_radius
    # Top-left point, clockwise around T
    xy_positions = jp.array([
        [-t_x - r, t_y + r],
        [t_x + r, t_y + r],
        [t_x + r, t_y - hallway_width - r],
        [hallway_width + r, t_y - hallway_width - r],
        [hallway_width + r, -r],
        [-hallway_width - r, -r],
        [-hallway_width - r, t_y - hallway_width - r],
        [-t_x - r, t_y - hallway_width - r]
    ])
    for i in range(len(xy_positions)):
        add_capsule_wall_to_body(arena, xy_positions[i], xy_positions[int((i+1) % xy_positions.shape[0])], r, True)
