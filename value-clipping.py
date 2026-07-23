import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, lo: float, hi: float, N: int) -> jax.Array:
    return jnp.clip(input, lo, hi)
