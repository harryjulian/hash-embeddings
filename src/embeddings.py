from functools import partial
import jax
from jax import Array
import jax.numpy as jnp


def encode_text(text: str) -> Array:
  """Return string as unicode representation of chars."""
  return jnp.array([ord(i) for i in text])


@jax.jit
def rolling_hash(encoded: Array, p: int = 31, m: int = 10**9+7) -> Array:
  """Given a piece of text, compute it's rolling hash.
  
  Args:
    encoded: Array
    p: int
    m: int
  
  Returns:
    hash values: Array
  """

  def body_fn(result, elem):
    value, power = result
    value = (value + (elem - 96) * power) % m
    power = (power * p) % m
    return (value, power), elem

  # Compute rolling hash
  ((hash_value, _), _) = jax.lax.scan(body_fn, (encoded[0], encoded.shape[0]), encoded[1:])

  return hash_value


@partial(jax.jit, static_argnums=(1,))
def sliding_window(a: Array, size: int) -> Array:
  """Get all sliding windows of size over a."""
  starts = jnp.arange(len(a) - size + 1)
  return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)


def hash_embedding(encoded: Array, seed: int = 42, b: int = 10**9+7, n: int = 3, d: int = 768) -> Array:
    """Compute hash embedding of a given piece of text.
    
    Args:
        encoded: Array -  encoded text we're embedding
        seed: int - random seed - MUST be held constant across embeddings.
        b: int - scalar bucket size.
        n: int - maximum size of an i-gram.
        d: int - the dimension of the embedding.
    
    Returns:
        embedding: jnp.array[d,]
    """

    # Initialize h and partitions
    partitions = jnp.sum(jnp.arange(1, n+1))
    h = jax.random.split(jax.random.PRNGKey(seed), d)[:, 0].reshape((d / partitions).astype(int), partitions) # reduce to 1d

    # Initialize loop variables
    embedding = jnp.zeros((d,))
    partition_idx = jnp.arange(0, d+1, int(d / partitions))
    run = 0

    # TODO: It'd be nice to move this to use jax.lax.scan
    for i in range(1, n+1):

        # Compute rolling hash
        igrams = sliding_window(encoded, i)
        s = rolling_hash(igrams.T)

        # Compute projection matrix
        p = jnp.outer(s, h[:, run:run+i]) # select the partition which is equal to run: run + i

        # Normalize
        p = p % b
        p = p - jnp.greater(p, b / 2) * b
        p = p / (b / 2)

        # Average
        igram_embedding = jnp.mean(p, axis = 0)

        # Concat to final embedding
        embedding = embedding.at[partition_idx[run]: partition_idx[run + i]].set(igram_embedding)
        run += i

    return embedding