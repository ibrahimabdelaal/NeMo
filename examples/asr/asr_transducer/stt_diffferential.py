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
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler

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

            scheduler_config = prepare_lr_scheduler(
                optimizer=optimizer, optim_config=optim_config.decoder_optim.sched, trainer=self.trainer
            )
            
            scheduler = {
                'scheduler': scheduler_config['scheduler'],
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
        else:
            logging.info("Using default single learning rate setup.")
            return super().configure_optimizers()


@hydra_runner(config_path="conf", config_name="your_config_name_here")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Definitive Logic: Build from config, then manually load filtered weights ---
    
    # --- This is the critical fix ---
    # 1. Save the optimizer config BEFORE creating the model.
    optim_config = cfg.model.optim
    
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
    for key, value in pretrained_weights.items():
        if key.startswith('encoder.'):
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
