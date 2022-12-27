import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from language_modeling_data import LanguageModelingDataModule
from language_modeling_model import LanguageModelingTransformer


@hydra.main(version_base=None, config_path='conf', config_name='language_modeling')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    tokenizer.add_tokens(['break'])
    dm = LanguageModelingDataModule(batch_size=config.batch_size,
                                    preprocessing_num_workers=config.num_workers,
                                    data_type=config.data_type,
                                    train_file=config.train_file,
                                    validation_file=config.validation_file,
                                    test_file=config.test_file,
                                    tokenizer=tokenizer)

    model = LanguageModelingTransformer(pretrained_model_name_or_path=config.pretrained_model, tokenizer=tokenizer)
    model.on_fit_start()

    trainer = pl.Trainer(accelerator=config.accelerator,
                         devices=config.devices,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         enable_checkpointing=True,
                         enable_progress_bar=True,
                         enable_model_summary=True
                         )
    trainer.test(model=model, datamodule=dm, ckpt_path=config.model_path)

    print(trainer.model.inference("엄마, 떡국 먹으면 한 살 더 먹어요?"))
    print(trainer.model.inference('그리스 신화에 나오는 태양과 예언 및 광명 의술 궁술 음악 시를 주관하는 신이다.'))


if __name__ == "__main__":
    main()
