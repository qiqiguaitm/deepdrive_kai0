import dataclasses
import faulthandler
import functools
import json
import logging
import os
import platform
import subprocess
import sys
from typing import Any

# On sim01, training sometimes silently SIGBUSes on GPU 1/2 (0-MB NUMA nodes)
# when JAX/XLA does a pinned-memory alloc after inline-eval. faulthandler
# intercepts SIGSEGV/SIGBUS/SIGFPE/SIGILL/SIGABRT and dumps the Python C stack
# trace to stderr before the process dies, so we can see exactly which call
# triggers it instead of silent death. No overhead when no fault happens.
faulthandler.enable(file=sys.stderr, all_threads=True)
# Also register SIGUSR1 so we can pkill -USR1 <pid> to dump stacks on demand.
import signal as _signal
try:
    faulthandler.register(_signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except Exception:
    pass

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import shutil
from pathlib import Path


# ----- Inline eval (in-process, shares train_state.params — no GPU double-allocation) -----
# Enable via TrainConfig fields:
#   inline_eval_val_root : str path to val dataset root (required; None = disabled)
#   inline_eval_n_frames : int  query frames per val episode (default 20)
#   inline_eval_every    : int  run eval every Nth save_interval boundary (default 1)
# Memory: re-uses train_state params (~0 GB extra) + inference activations (~3 GB).
_VAL_CACHE = {"root": None, "samples": None}

def _load_val_data(val_root: str, n_frames_per_ep: int):
    """Load val parquets + decoded mp4 frames at q_idx only (sparse cache).

    Per-process RAM: ~500 MB (9 ep × 3 cam × 20 frames × 480×640×3 B) vs. ~18.5 GB
    for full-episode cache. Critical when 4 train procs each hold their own copy.
    """
    global _VAL_CACHE
    if _VAL_CACHE["root"] == val_root and _VAL_CACHE["samples"] is not None:
        return _VAL_CACHE["samples"]
    import pyarrow.parquet as pq
    import av as _av
    val_root_p = Path(val_root)
    samples = []
    for ep_path in sorted((val_root_p / "data" / "chunk-000").glob("episode_*.parquet")):
        df = pq.read_table(ep_path).to_pandas()
        ep_idx = int(ep_path.stem.split("_")[1])
        state = np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
        action = np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]])
        L = len(state)
        q_idx = np.linspace(0, max(L - 51, 0), n_frames_per_ep).astype(int)
        q_set = set(int(k) for k in q_idx)
        vids = {}  # cam -> {k: frame}
        for cam in ("top_head", "hand_left", "hand_right"):
            vp = val_root_p / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            container = _av.open(str(vp))
            container.streams.video[0].thread_type = "AUTO"
            picked = {}
            for i, f in enumerate(container.decode(video=0)):
                if i in q_set:
                    picked[i] = f.to_ndarray(format="rgb24")
                    if len(picked) == len(q_set):
                        break
            container.close()
            # last-frame fallback for short episodes
            last_decoded = next(iter(sorted(picked.keys(), reverse=True)), None)
            if last_decoded is not None:
                for k in q_set - set(picked):
                    picked[k] = picked[last_decoded]
            vids[cam] = picked
        samples.append({"ep_idx": ep_idx, "length": L, "state": state, "action": action,
                        "images": vids, "q_idx": q_idx})
    _VAL_CACHE = {"root": val_root, "samples": samples}
    return samples


def _build_eval_policy(train_state, config, data_config):
    """Policy sharing train_state params (no copy). ema_params preferred if present."""
    from openpi.policies import policy as _policy
    import openpi.transforms as _transforms_mod
    params = train_state.ema_params if train_state.ema_params is not None else train_state.params
    model = nnx.merge(train_state.model_def, params)
    model.eval()
    default_prompt = getattr(config.data, "default_prompt", None)
    norm_stats = data_config.norm_stats
    use_q = data_config.use_quantile_norm
    # Mirror policy_config.create_trained_policy: skip data_config.repack_transforms (they're
    # for dataset→internal key remapping during training; our inline-eval obs is already in
    # the internal format {images, state, prompt}).
    transforms_in = [
        _transforms_mod.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        _transforms_mod.Normalize(norm_stats, use_quantiles=use_q),
        *data_config.model_transforms.inputs,
    ]
    transforms_out = [
        *data_config.model_transforms.outputs,
        _transforms_mod.Unnormalize(norm_stats, use_quantiles=use_q),
        *data_config.data_transforms.outputs,
    ]
    return _policy.Policy(model, transforms=transforms_in, output_transforms=transforms_out)


def _run_inline_eval(train_state, config, data_config, step, mesh):
    val_root = config.inline_eval_val_root
    if not val_root:
        return
    n_frames = config.inline_eval_n_frames
    every = config.inline_eval_every
    if (step // config.save_interval) % every != 0:
        return
    import time
    t0 = time.time()
    try:
        samples = _load_val_data(val_root, n_frames)
        with sharding.set_mesh(mesh):
            policy = _build_eval_policy(train_state, config, data_config)
            HORIZONS = (1, 10, 25, 50)
            acc = {h: [] for h in HORIZONS}
            for s in samples:
                for k in s["q_idx"]:
                    obs = {
                        "images": {c: s["images"][c][k] for c in s["images"]},
                        "state": s["state"][k],
                        "prompt": getattr(config.data, "default_prompt", "stand up the fallen box"),
                    }
                    res = policy.infer(obs)
                    pred = np.asarray(res["actions"])
                    for h in HORIZONS:
                        if k + 1 + h <= s["length"] and h <= pred.shape[0]:
                            gt = s["action"][k + 1:k + 1 + h]
                            acc[h].append(float(np.mean(np.abs(gt - pred[:h]))))
        mae = {h: float(np.mean(acc[h])) for h in HORIZONS if acc[h]}
        wandb.log({f"val/mae_{h}": v for h, v in mae.items()}, step=step)
        logging.info(
            f"[inline-eval] step={step}  MAE@1={mae.get(1,0):.4f}  @10={mae.get(10,0):.4f}  "
            f"@25={mae.get(25,0):.4f}  @50={mae.get(50,0):.4f}  ({time.time()-t0:.1f}s)"
        )
    except Exception as e:
        logging.warning(f"[inline-eval] failed at step={step}: {e}", exc_info=True)


import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def eval_step(
    config: _config.TrainConfig,
    norm_q01: jnp.ndarray,
    norm_q99: jnp.ndarray,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Run sample_actions on eval batch and compute per-horizon MAE in raw units (rad / m).

    Applies inverse quantile normalization to both pred and gt before MAE, so output
    is in dataset-native units (radian for joints, meter for gripper).

    Returns: mae_joint_{1,10,50} (rad), mae_grip_{1,10,50} (m).
      Joint dims: 0-5 (L arm joints), 7-12 (R arm joints). Gripper dims: 6, 13.
      assumes use_delta_joint_actions=False (direct abs-action prediction).
    """
    observation, actions_gt = batch
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    eval_rng = jax.random.fold_in(rng, state.step)
    pred = model.sample_actions(eval_rng, observation, num_steps=config.eval_num_diffusion_steps)
    # pred: [B, H, 32] in normalized [-1, 1] quantile space
    # actions_gt: [B, H, 32] same space

    # Inverse quantile normalize to raw (rad/meter) space:
    #   x_raw = (x_norm + 1) / 2 * (q99 - q01) + q01
    q01_14 = norm_q01[:14]
    q99_14 = norm_q99[:14]
    scale = (q99_14 - q01_14 + 1e-6) / 2.0
    bias = q01_14
    pred_raw = (pred[..., :14] + 1.0) * scale + bias
    gt_raw = (actions_gt[..., :14] + 1.0) * scale + bias
    err = jnp.abs(pred_raw - gt_raw)  # [B, H, 14] in rad/meter
    H = err.shape[1]

    joint_idx = jnp.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    grip_idx = jnp.array([6, 13])

    idx_1 = 0
    idx_10 = jnp.minimum(9, H - 1)
    idx_50 = jnp.minimum(49, H - 1)

    return {
        "mae_joint_1":  err[:, idx_1, joint_idx].mean(),
        "mae_joint_10": err[:, idx_10, joint_idx].mean(),
        "mae_joint_50": err[:, idx_50, joint_idx].mean(),
        "mae_grip_1":   err[:, idx_1, grip_idx].mean(),
        "mae_grip_10":  err[:, idx_10, grip_idx].mean(),
        "mae_grip_50":  err[:, idx_50, grip_idx].mean(),
    }


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        result = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(result, dict):
            main = jnp.mean(result["main_loss"])
            total = main
            aux_info = {"main_loss": main}
            for name in ("cl_loss", "dct_loss"):
                if name in result:
                    weight_key = name.replace("_loss", "_weight")
                    contrib = result[weight_key] * result[name]
                    total = total + contrib
                    aux_info[name] = result[name]
                    aux_info[name + "_weighted"] = contrib
            return total, aux_info
        return jnp.mean(result), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **aux_info,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    # Multi-node init (env var driven — set by run_multinode.sh or similar launcher).
    coord_addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coord_addr:
        jax.distributed.initialize(
            coordinator_address=coord_addr,
            num_processes=int(os.environ.get("JAX_NUM_PROCESSES", 1)),
            process_id=int(os.environ.get("JAX_PROCESS_INDEX", 0)),
        )

    init_logging()
    logging.info(f"Running on: {platform.node()} process {jax.process_index()}/{jax.process_count()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    if jax.process_index() == 0:
        dst_dir = config.checkpoint_dir
        src_file = Path(config.data.repo_id) / 'norm_stats.json'
        shutil.copy(src_file, dst_dir)

    init_wandb(config, resuming=resuming, enabled=(config.wandb_enabled and jax.process_index() == 0))

    # 90/10 episode split (if val_ratio > 0)
    # NOTE: Task_A/advantage has sparse episode_index (values > 3055 exist while meta has
    # only 3055 entries) — LeRobotDataset's `episodes=` filter breaks when val subset
    # lands entirely in sparse region. Falling back to train-distribution eval.
    train_episodes = None
    val_episodes = None

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        episodes=train_episodes,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Tensor-based eval loader (triggered by config.eval_interval_early > 0).
    # Uses a different seed to sample a different subset — approximation of held-out
    # eval until the sparse episode_index bug is fixed.
    val_loader = None
    val_iter = None
    if config.eval_interval_early > 0:
        val_config = dataclasses.replace(config, seed=config.seed + 10000)
        val_loader = _data_loader.create_data_loader(
            val_config,
            sharding=data_sharding,
            shuffle=True,
            episodes=None,  # full dataset, bypass sparse-index bug
        )
        val_iter = iter(val_loader)
        logging.info("Initialized eval loader (full dataset, different seed)")

    # Log images from first batch to sanity check (primary process only).
    if config.wandb_enabled and jax.process_index() == 0:
        host_images = {k: np.asarray(v) for k, v in batch[0].images.items()}
        images_to_log = [
            wandb.Image(np.concatenate([host_images[k][i] for k in host_images], axis=1))
            for i in range(min(5, len(next(iter(host_images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Jit tensor-based eval step (if enabled).
    peval_step = None
    if val_iter is not None:
        dc = data_loader.data_config()
        action_norm = dc.norm_stats.get("actions") if dc.norm_stats else None
        if action_norm is None or action_norm.q01 is None:
            logging.warning("No action quantile norm stats; eval MAE will be in normalized space, not rad/meter.")
            norm_q01 = jnp.zeros(32) - 1.0   # so: (x+1)*1 - 1 = x
            norm_q99 = jnp.ones(32)
        else:
            norm_q01 = jnp.asarray(action_norm.q01)
            norm_q99 = jnp.asarray(action_norm.q99)
        peval_step = jax.jit(
            functools.partial(eval_step, config, norm_q01, norm_q99),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)

    def should_eval_at(step: int) -> bool:
        """Eval schedule: first eval_early_count times at eval_interval_early (counted from start_step),
        then eval_interval_late (absolute step multiples). Works on both fresh and resumed training.
        """
        if peval_step is None or config.eval_interval_early <= 0:
            return False
        if step <= start_step:
            return False
        steps_since_start = step - start_step
        n_early_steps = config.eval_interval_early * config.eval_early_count
        if steps_since_start <= n_early_steps:
            return steps_since_start % config.eval_interval_early == 0
        return step % config.eval_interval_late == 0

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)
            infos = []

        if should_eval_at(step):
            eval_accum = []
            for _ in range(config.eval_batches):
                try:
                    eval_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    eval_batch = next(val_iter)
                with sharding.set_mesh(mesh):
                    em = peval_step(train_rng, train_state, eval_batch)
                eval_accum.append(em)
            eval_reduced = jax.device_get(jax.tree.map(
                lambda *xs: jnp.mean(jnp.stack(xs)), *eval_accum
            ))
            eval_str = ", ".join(f"{k}={v:.4f}" for k, v in eval_reduced.items())
            pbar.write(f"  Eval@{step}: {eval_str}")
            if jax.process_index() == 0:
                wandb.log({f"eval/{k}": v for k, v in eval_reduced.items()}, step=step)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if config.inline_eval_val_root:
                _run_inline_eval(train_state, config, data_loader.data_config(), step, mesh)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
