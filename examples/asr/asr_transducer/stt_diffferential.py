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
    def setup_optimization(self, optim_config):
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

            scheduler_config = prepare_lr_scheduler(
                optimizer=optimizer, optim_config=optim_config.decoder_optim.sched, trainer=self.trainer
            )
            
            scheduler = {
                'scheduler': scheduler_config['scheduler'],
                'interval': 'step',
                'frequency': 1,
            }
            self._optimizer = optimizer
            self._scheduler = scheduler
        else:
            logging.info("Using default single learning rate setup.")
            super().setup_optimization(optim_config)


@hydra_runner(config_path="conf", config_name="your_config_name_here")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Definitive Logic: Build from config, then manually load filtered weights ---
    
    # 1. Build the model from the config file using our custom class.
    logging.info("Building model from config...")
    asr_model = CustomHybridModel(cfg=cfg.model, trainer=trainer)

    # --- IMPORTANT: The following block REPLACES `maybe_init_from_pretrained_checkpoint` ---
    # 2. Manually load the pretrained weights and filter them.
    logging.info(f"Loading and filtering weights from {cfg.init_from_nemo_model}")
    
    with tempfile.TemporaryDirectory() as restore_dir:
        # Ensure the .nemo file exists before trying to open it
        if not os.path.exists(cfg.init_from_nemo_model):
            raise FileNotFoundError(f"The .nemo file was not found at path: {cfg.init_from_nemo_model}")
            
        with tarfile.open(cfg.init_from_nemo_model, "r:gz") as tar:
            tar.extractall(path=restore_dir)
        
        checkpoint_path = os.path.join(restore_dir, 'model_weights.ckpt')
        pretrained_weights = torch.load(checkpoint_path, map_location='cpu')

    # 3. Create a new state_dict, keeping only the encoder weights.
    #    This directly implements your request: "we do not want the weights from the decoder".
    new_state_dict = {}
    for key, value in pretrained_weights.items():
        if key.startswith('encoder.'):
            new_state_dict[key] = value

    # 4. Load the filtered (encoder-only) weights into our new model.
    #    `strict=False` is crucial because we are intentionally ignoring the decoder/joint weights
    #    and the new CTC head has no corresponding pre-trained weights.
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
