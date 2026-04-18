import os, jax, jax.numpy as jnp
print(f"PROCESS INDEX: {os.environ.get('JAX_PROCESS_INDEX')}", flush=True)
jax.distributed.initialize(
    coordinator_address=os.environ['JAX_COORDINATOR_ADDRESS'],
    num_processes=int(os.environ['JAX_NUM_PROCESSES']),
    process_id=int(os.environ['JAX_PROCESS_INDEX']),
)
print(f'[rank{jax.process_index()}] init OK, total devices={jax.device_count()}', flush=True)
import jax.numpy as jnp
mesh = jax.make_mesh((jax.device_count(),), ('x',))
with mesh:
    x = jnp.ones(()) 
    result = jax.lax.psum(x, axis_name='x')
print(f'[rank{jax.process_index()}] psum OK: {result.item()}', flush=True)
