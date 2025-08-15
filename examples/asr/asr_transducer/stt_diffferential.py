import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
import tarfile
import tempfile
import os

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
# Import the specific scheduler class instead of the general utility
from nemo.core.optim.lr_scheduler import NoamAnnealing

class CustomHybridModel(EncDecHybridRNNTCTCBPEModel):
    # --- This is the corrected approach: Override configure_optimizers ---
    def configure_optimizers(self):
        """
        Overrides the standard PyTorch Lightning method to set up
        differential learning rates for the encoder and decoders.
        """
        # Read the optimizer config from the attribute we manually attached.
        optim_config = self._new_optim_config

        if "encoder_optim" in optim_config and "decoder_optim" in optim_config:
            logging.info("Setting up differential learning rates for encoder and decoders.")
            
            encoder_params = {
                "params": self.encoder.parameters(),
                "lr": optim_config.encoder_optim.lr,
            }
            # The decoder group includes the RNN-T decoder, joint network, and the new CTC head
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

            # --- FINAL FIX APPLIED HERE ---
            # The base NeMo model expects the optimizer to be stored on the object.
            # We assign it here to make it accessible for the training_step.
            self._optimizer = optimizer
            
            # Manually instantiate the scheduler directly.
            logging.info("Manually instantiating NoamAnnealing scheduler.")
            
            scheduler_config = optim_config.decoder_optim.sched
            
            # The trainer computes max_steps at runtime. We access it here.
            max_steps = self.trainer.max_steps if self.trainer else -1
            if max_steps is None or max_steps == -1:
                 logging.warning(
                    "Trainer.max_steps is not set, so NoamAnnealing scheduler will run indefinitely."
                 )
                 # Pass None to the scheduler if max_steps is not available
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


@hydra_runner(config_path="conf", config_name="para_hypird")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Definitive Logic: Build from config, then manually load filtered weights ---
    
    # 1. Save the optimizer config BEFORE creating the model.
    optim_config = cfg.optim
    
    # 2. Build the model from the config file using our custom class.
    logging.info("Building model from config...")
    asr_model = CustomHybridModel(cfg=cfg.model, trainer=trainer)

    # 3. Manually attach the saved optimizer config to the model object.
    asr_model._new_optim_config = optim_config

    # 4. Manually load the pretrained model from Hugging Face to get its weights.
    pretrained_model_name = cfg.model.init_from_pretrained_model
    logging.info(f"Downloading and loading pretrained model: {pretrained_model_name}")
    
    # We load the full model into a temporary object just to get its state_dict
    pretrained_model = CustomHybridModel.from_pretrained(model_name=pretrained_model_name, map_location='cpu')
    pretrained_weights = pretrained_model.state_dict()
    del pretrained_model  # Free up memory
    
    # 5. Create a new state_dict, keeping only the encoder weights.
    new_state_dict = {}
    excluded_prefixes = ['decoder.', 'joint.', 'ctc_decoder.']
    
    for key, value in pretrained_weights.items():
        # Only include encoder weights and exclude all decoder-related weights
        if key.startswith('encoder.'):
            new_state_dict[key] = value
        elif not any(key.startswith(prefix) for prefix in excluded_prefixes):
            # For weights that don't match these patterns and have matching shapes,
            # we can try to load them
            if key in asr_model.state_dict() and asr_model.state_dict()[key].shape == value.shape:
                new_state_dict[key] = value

    # 6. Load the filtered (encoder-only) weights into our new model.
    asr_model.load_state_dict(new_state_dict, strict=False)
    
    logging.info(f"Successfully loaded {len(new_state_dict)} encoder weights.")
    logging.info("Decoder, Joint, and CTC Head are randomly initialized.")

    logging.info("--- ✅ Model setup complete. Starting training... ---")
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
