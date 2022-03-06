from typing import Sequence

import jax
from brax.jumpy import _in_jit, Carry, X, Y, Optional, Tuple, Callable, onp, Any


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state."""
  if _in_jit():
      return jax.lax.scan(f, init, xs, length, reverse, unroll)
  else:
      xs_flat, xs_tree = jax.tree_flatten(xs)
      carry = init
      ys = []
      maybe_reversed = reversed if reverse else lambda x: x
      for i in maybe_reversed(range(length)):
        xs_slice = [x[i] for x in xs_flat]
        carry, y = f(carry, jax.tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
      stacked_y = jax.tree_map(lambda *y: onp.vstack(y), *maybe_reversed(ys))
      return carry, stacked_y

def while_loop(cond_fun: Callable[[X], Any],
               body_fun: Callable[[X], X],
               init_val: X) -> X:
    """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

    The type signature in brief is

    .. code-block:: haskell

      while_loop :: (a -> Bool) -> (a -> a) -> a -> a

    The semantics of ``while_loop`` are given by this Python implementation::

      def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
          val = body_fun(val)
        return val
    """
    if _in_jit():
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    else:
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val