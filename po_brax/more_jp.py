from typing import Sequence, Union

import jax
from brax.jumpy import _in_jit, X, Optional, Tuple, Callable, onp, Any, _which_np, jnp, ndarray


def while_loop(cond_fun: Callable[[X], Any],
               body_fun: Callable[[X], X],
               init_val: X) -> X:
  """Call body function while conditional function is true, starting with state"""
  if _in_jit():
    return jax.lax.while_loop(cond_fun, body_fun, init_val)
  else:
    val = init_val
    while cond_fun(val):
      val = body_fun(val)
    return val


def fori_loop(lower: int, upper: int,
               body_fun: Callable[[X], X],
               init_val: X) -> X:
  """Call body function while conditional function is true, starting with state"""
  if _in_jit():
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)
  else:
    val = init_val
    for i in range(lower, upper):
      val = body_fun(val)
    return val


def index_add(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
    """Pure equivalent of x[idx] += y."""
    if _which_np(x) is jnp:
        return x.at[idx].add(y)
    x = onp.copy(x)
    x[idx] += y
    return x


def index_update(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
  """Pure equivalent of x[idx] += y."""
  if _which_np(x) is jnp:
    return x.at[idx].set(y, mode='drop')
  x = onp.copy(x)
  x[idx] = y
  return x


def meshgrid(*xi, copy: bool = True, sparse: bool = False, indexing: str = 'xy') -> ndarray:
    if _which_np(xi[0]) is jnp:
        return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
    return onp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)


def randint(rng: ndarray, shape: Tuple[int, ...] = (),
            low: Optional[int] = 0, high: Optional[int] = 1) -> ndarray:
  """Sample integers in [low, high) with given shape."""
  if _which_np(rng) is jnp:
    return jax.random.randint(rng, shape=shape, minval=low, maxval=high)
  else:
    return onp.random.default_rng(rng).integers(low=low, high=high, size=shape)


def maximum(x1: ndarray, x2: ndarray) -> ndarray:
  """Element-wise maximum of array elements."""
  return _which_np(x1, x2).maximum(x1, x2)


def choice(rng: ndarray, a: Union[int, Any], shape: Tuple[int, ...] = (),
           replace: bool = True, p: Optional[Any] = None, axis: int = 0) -> ndarray:
  """Generate sample(s) from given array"""
  if _which_np(rng) is jnp:
    return jax.random.choice(rng, a, shape=shape, replace=replace, p=p, axis=axis)
  else:
    return onp.random.default_rng(rng).choice(a, size=shape, replace=replace, p=p, axis=axis)


def atleast_1d(*arys) -> ndarray:
  """Ensure arrays are all at least 1d (dimensions added to beginning)"""
  return _which_np(*arys).atleast_1d(*arys)


def atleast_2d(*arys) -> ndarray:
  """Ensure arrays are all at least 2d (dimensions added to beginning)"""
  return _which_np(*arys).atleast_2d(*arys)


def atleast_3d(*arys) -> ndarray:
  """Ensure arrays are all at least 3d (dimensions added to beginning)"""
  return _which_np(*arys).atleast_3d(*arys)


def cond(pred, true_fun: Callable, false_fun: Callable, *operands: Any):
  """Conditionally apply true_fun or false_fun to operands"""
  if _in_jit():
    return jax.lax.cond(pred, true_fun, false_fun, *operands)
  else:
    if pred:
      return true_fun(operands)
    else:
      return false_fun(operands)

