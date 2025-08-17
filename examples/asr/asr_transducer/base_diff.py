import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.core.optim.lr_scheduler import NoamAnnealing

# --- NO CHANGES TO THIS CLASS ---
# This class is still essential for applying your differential learning rates.
class CustomHybridModel(EncDecHybridRNNTCTCBPEModel):
    def configure_optimizers(self):
        optim_config = self._new_optim_config

        if "encoder_optim" in optim_config and "decoder_optim" in optim_config:
            logging.info("Setting up differential learning rates for encoder and decoders.")
            
            encoder_params = {
                "params": self.encoder.parameters(),
                "lr": optim_config.encoder_optim.lr,
            }
            decoder_params = {
                "params": list(self.decoder.parameters()) + list(self.joint.parameters()) + list(self.ctc_decoder.parameters()),
                "lr": optim_config.decoder_optim.lr,
            }
            
            optimizer_kwargs = {
                "betas": optim_config.decoder_optim.betas,
                "weight_decay": optim_config.decoder_optim.weight_decay,
            }
            
            param_groups = [encoder_params, decoder_params]
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

            self._optimizer = optimizer
            
            logging.info("Manually instantiating NoamAnnealing scheduler.")
            
            scheduler_config = optim_config.decoder_optim.sched
            max_steps = self.trainer.max_steps if self.trainer else -1
            if max_steps is None or max_steps == -1:
                 logging.warning(
                    "Trainer.max_steps is not set, so NoamAnnealing scheduler will run indefinitely."
                 )
                 max_steps = None

            scheduler_instance = NoamAnnealing(
                optimizer=optimizer,
                d_model=scheduler_config.d_model,
                warmup_steps=scheduler_config.warmup_steps,
                min_lr=scheduler_config.min_lr,
                max_steps=max_steps
            )
            
            scheduler = {
                'scheduler': scheduler_instance,
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
        else:
            logging.info("Using default single learning rate setup.")
            return super().configure_optimizers()

# --- Use this updated main function ---
# --- Use this updated main function ---
@hydra_runner(config_path="conf", config_name="para_hypird_resume")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Step 1: Restore the model from the .nemo file.
    logging.info(f"Restoring model from: {cfg.model.restore_from_path}")
    asr_model = CustomHybridModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer
    )
    logging.info("Model restored successfully.")

    # --- ✨ KEY FIX APPLIED HERE ---
    # Step 2: Manually resolve the sample_rate before updating the model's dataset config.
    # We create a new config object for the datasets and explicitly set the sample rate value.
    
    # Get the concrete sample_rate value from the main config
    sample_rate_val = cfg.model.sample_rate
    
    # Create a new dataset config with the resolved sample_rate
    new_data_cfg = OmegaConf.create({
        'train_ds': cfg.model.train_ds,
        'validation_ds': cfg.model.validation_ds,
        'test_ds': cfg.model.get('test_ds', None) # Use .get for optional test_ds
    })
    
    # Update the sample_rate in the new config to be the actual value, not the variable
    new_data_cfg.train_ds.sample_rate = sample_rate_val
    new_data_cfg.validation_ds.sample_rate = sample_rate_val
    if new_data_cfg.test_ds:
        new_data_cfg.test_ds.sample_rate = sample_rate_val

    # Step 3: Use the model's setup methods with the fully resolved data config
    logging.info("Explicitly setting up new training and validation datasets...")
    asr_model.setup_training_data(new_data_cfg.train_ds)
    asr_model.setup_validation_data(new_data_cfg.validation_ds)
    if new_data_cfg.test_ds:
        asr_model.setup_test_data(new_data_cfg.test_ds)

    # Step 4 (As before): Re-attach the optimizer config
    asr_model._new_optim_config = cfg.optim

    logging.info("--- ✅ Model setup complete. Starting Phase 2 training... ---")
    trainer.fit(asr_model, ckpt_path=None)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()

