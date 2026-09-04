import copy

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import (
    EncDecCTCModelBPE,
    EncDecHybridRNNTCTCBPEModel,
    EncDecRNNTBPEModel,
)
from nemo.utils.exp_manager import exp_manager

_MODEL_CLASSES = {
    "hybrid": EncDecHybridRNNTCTCBPEModel,
    "ctc": EncDecCTCModelBPE,
    "rnnt": EncDecRNNTBPEModel,
}


class UnfreezeEncoderCallback(pl.Callback):
    def __init__(self, unfreeze_at_step: int):
        self.unfreeze_at_step = unfreeze_at_step
        self.done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.done and trainer.global_step >= self.unfreeze_at_step:
            pl_module.encoder.unfreeze()
            print(f"Encoder unfrozen at step {trainer.global_step}")
            self.done = True


def _num_devices(devices) -> int:
    """Resolves how many GPUs `devices` (an int count, -1 for "all", or a
    list of GPU indices) actually maps to, so the right strategy can be
    picked before Lightning builds the accelerator."""
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if devices == -1:
        return torch.cuda.device_count()
    return int(devices)


def build_trainer(args, callbacks) -> pl.Trainer:
    return pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        strategy="ddp" if _num_devices(args.devices) != 1 else "auto",
        use_distributed_sampler=False,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
    )


def build_exp_manager_cfg(args) -> dict:
    cfg = {
        "exp_dir": args.exp_dir,
        "name": args.exp_name,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 5,
        },
    }
    if args.wandb_project:
        cfg["create_wandb_logger"] = True
        cfg["wandb_logger_kwargs"] = {
            "name": args.wandb_run_name or args.exp_name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
        }
    if getattr(args, "resume", False):
        cfg["resume_if_exists"] = True
        cfg["resume_ignore_no_checkpoint"] = True
    return cfg


def load_pretrained(model_class, pretrained_model: str, trainer):
    """Load a base checkpoint by NGC name, HuggingFace repo id, or local path.

    `from_pretrained` only resolves NVIDIA's own NGC/HF model names, so anything
    else - a community repo like `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large`,
    or a `.nemo` sitting on disk - is fetched first and restored from the file.
    """
    if pretrained_model.endswith(".nemo"):
        print(f"Restoring from local checkpoint {pretrained_model}")
        return model_class.restore_from(pretrained_model, trainer=trainer)

    if "/" in pretrained_model and not pretrained_model.startswith("nvidia/"):
        from huggingface_hub import list_repo_files, hf_hub_download

        nemo_files = [f for f in list_repo_files(pretrained_model) if f.endswith(".nemo")]
        if not nemo_files:
            raise SystemExit(
                f"{pretrained_model!r} has no .nemo file to restore from"
            )
        print(f"Downloading {nemo_files[0]} from {pretrained_model}")
        local_path = hf_hub_download(pretrained_model, nemo_files[0])
        return model_class.restore_from(local_path, trainer=trainer)

    return model_class.from_pretrained(model_name=pretrained_model, trainer=trainer)


def run_training(args):
    torch.set_float32_matmul_precision("medium")

    callbacks = []
    if args.freeze_encoder_steps > 0:
        callbacks.append(UnfreezeEncoderCallback(args.freeze_encoder_steps))

    trainer = build_trainer(args, callbacks)
    exp_manager(trainer, build_exp_manager_cfg(args))

    model_type = getattr(args, "model_type", None) or "hybrid"
    if model_type not in _MODEL_CLASSES:
        raise SystemExit(
            f"Invalid train.model_type {model_type!r}, expected one of {list(_MODEL_CLASSES)}"
        )
    model = load_pretrained(_MODEL_CLASSES[model_type], args.pretrained_model, trainer)

    # A checkpoint already trained on the target language ships a matching
    # tokenizer, and swapping it out would discard the pretrained decoder along
    # with it. Setting tokenizer_dir to null keeps the checkpoint's own.
    if args.tokenizer_dir:
        model.change_vocabulary(
            new_tokenizer_dir=args.tokenizer_dir, new_tokenizer_type="bpe"
        )
        print(f"Swapped in tokenizer from {args.tokenizer_dir}")
    else:
        print("tokenizer_dir is unset - keeping the checkpoint's own tokenizer")
    print(f"Tokenizer vocab size: {model.tokenizer.vocab_size}")

    train_ds_cfg = copy.deepcopy(model.cfg.train_ds)
    train_ds_cfg.manifest_filepath = args.train_manifest
    train_ds_cfg.batch_size = args.batch_size
    train_ds_cfg.num_workers = args.num_workers
    train_ds_cfg.is_tarred = False
    model.setup_training_data(train_ds_cfg)

    val_ds_cfg = copy.deepcopy(model.cfg.train_ds)
    val_ds_cfg.manifest_filepath = args.val_manifest
    val_ds_cfg.batch_size = args.val_batch_size
    val_ds_cfg.num_workers = args.num_workers
    val_ds_cfg.is_tarred = False
    val_ds_cfg.shuffle = False
    model.setup_validation_data(val_ds_cfg)

    model.cfg.optim.lr = args.lr
    if "sched" in model.cfg.optim and model.cfg.optim.sched.get("min_lr") is not None:
        model.cfg.optim.sched.min_lr = min(model.cfg.optim.sched.min_lr, args.lr / 10)
    model.setup_optimization(model.cfg.optim)

    if args.freeze_encoder_steps > 0:
        model.encoder.freeze()
        print(f"Encoder frozen for the first {args.freeze_encoder_steps} steps")

    trainer.fit(model)

    final_path = f"{args.exp_dir}/{args.exp_name}/final.nemo"
    model.save_to(final_path)
    print(f"Saved final checkpoint to {final_path}")
    return final_path
