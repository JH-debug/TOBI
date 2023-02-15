import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from seq2seq_data import Seq2SeqDataModule
from seq2seq_model import Seq2SeqModelTransformer


@hydra.main(version_base=None, config_path='../conf', config_name='seq2seq')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    dm = Seq2SeqDataModule(batch_size=config.batch_size,
                           preprocessing_num_workers=config.num_workers,
                           data_type=config.data_type,
                           train_file=config.train_file,
                           validation_file=config.validation_file,
                           test_file=config.test_file,
                           tokenizer=tokenizer)

    model = Seq2SeqModelTransformer(pretrained_model_name_or_path=config.pretrained_model)

    trainer = pl.Trainer(accelerator=config.accelerator,
                         devices=config.devices,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         enable_checkpointing=True,
                         enable_progress_bar=True,
                         enable_model_summary=True
                         )
    trainer.test(model=model, ckpt_path=config.model_path, datamodule=dm)


if __name__ == "__main__":
    main()
