import copy

import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.exp_manager import exp_manager


class UnfreezeEncoderCallback(pl.Callback):
    def __init__(self, unfreeze_at_step: int):
        self.unfreeze_at_step = unfreeze_at_step
        self.done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.done and trainer.global_step >= self.unfreeze_at_step:
            pl_module.encoder.unfreeze()
            print(f"Encoder unfrozen at step {trainer.global_step}")
            self.done = True


def build_trainer(args, callbacks) -> pl.Trainer:
    return pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=25,
        check_val_every_n_epoch=1,
        strategy="ddp" if args.devices != 1 else "auto",
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
    return cfg


def run_training(args):
    callbacks = []
    if args.freeze_encoder_steps > 0:
        callbacks.append(UnfreezeEncoderCallback(args.freeze_encoder_steps))

    trainer = build_trainer(args, callbacks)
    exp_manager(trainer, build_exp_manager_cfg(args))

    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=args.pretrained_model, trainer=trainer
    )

    model.change_vocabulary(
        new_tokenizer_dir=args.tokenizer_dir, new_tokenizer_type="bpe"
    )
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
