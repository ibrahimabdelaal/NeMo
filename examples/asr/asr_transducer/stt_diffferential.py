import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler

# --- Step 1: Create a Custom Model Class ---
# We inherit from the standard model and override its optimization setup.
class CustomRNNTBPEModel(EncDecRNNTBPEModel):
    def setup_optimization(self, optim_config):
        """
        This method is overridden to set up differential learning rates.
        It looks for `encoder_optim` and `decoder_optim` in the config.
        """
        # Check if our special config structure exists
        if "encoder_optim" in optim_config and "decoder_optim" in optim_config:
            logging.info("Setting up differential learning rates for encoder and decoder.")

            # --- Manually Create Parameter Groups ---
            # Group 1: The pre-trained encoder with a low learning rate
            encoder_params = {
                "params": self.encoder.parameters(),
                "lr": optim_config.encoder_optim.lr,
            }

            # Group 2: The new/fine-tuned decoder and joint with a high learning rate
            decoder_params = {
                "params": list(self.decoder.parameters()) + list(self.joint.parameters()),
                "lr": optim_config.decoder_optim.lr,
            }
            
            # Add other optimizer params from the decoder's config
            optimizer_kwargs = {
                "betas": optim_config.decoder_optim.betas,
                "weight_decay": optim_config.decoder_optim.weight_decay,
            }

            param_groups = [encoder_params, decoder_params]

            # --- Create Optimizer and Scheduler ---
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

            # Use NeMo's utility to create the scheduler
            # The scheduler will manage the learning rates for BOTH groups
            scheduler = prepare_lr_scheduler(
                optimizer=optimizer, optim_config=optim_config.decoder_optim.sched, trainer=self.trainer
            )

            self._optimizer = optimizer
            self._scheduler = scheduler

        else:
            # If our special config isn't found, fall back to the default NeMo behavior
            logging.info("Using default single learning rate setup.")
            super().setup_optimization(optim_config)


@hydra_runner(config_path="conf", config_name="parakeet_hybrid_finetune")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Step 2: Use our Custom Model Class ---
    # Instead of the default, we instantiate our new class.
    asr_model = CustomRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
