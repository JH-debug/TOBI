# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Type

import torch
import transformers
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.rouge import ROUGEScore
from transformers import MBartTokenizer
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from lightning_transformers.task.nlp.translation import TranslationDataModule


class Seq2SeqModelTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Translation Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        n_gram: Gram value ranged from 1 to 4.
        smooth: Whether or not to apply smoothing.
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSeq2SeqLM,
        n_gram: int = 1,
        smooth: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.bleu = None
        self.wer = None
        self.rouge = None
        self.n_gram = n_gram
        self.smooth = smooth

    def common_step(self, prefix: str, batch):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        if self.should_compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix)
        if prefix == 'val' or 'test':
            labels = batch['labels'][0]
            input_ids = batch['input_ids'][0]
            attention_mask = batch['attention_mask'][0]
            return {'loss': loss, 'labels': labels, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return loss

    def validation_epoch_end(self, outputs):
        print(self.tokenize_labels(outputs[0]['labels'].unsqueeze(0)))
        print(self.generate(outputs[0]['input_ids'].unsqueeze(0), outputs[0]['attention_mask'].unsqueeze(0)))

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        # wrap targets in list as score expects a list of potential references
        bleu_result = self.bleu(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_bleu_score", bleu_result, on_step=False, on_epoch=True, prog_bar=True)
        wer_result = self.wer(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_wer_score", wer_result, on_step=False, on_epoch=True, prog_bar=True)
        rouge_result = self.rouge(preds=pred_lns, target=tgt_lns)
        self.log_dict({f"{prefix}_{k}": v for k, v in rouge_result.items()}, on_step=False, on_epoch=True, prog_bar=True)

        correct = 0
        len_ = 0
        for tgt, pred in zip(tgt_lns, pred_lns):
            break_label = [i for i, j in enumerate(tgt.split()) if j == 'break']
            break_pred = [i for i, j in enumerate(pred.split()) if j == 'break']
            correct += len(set(break_label) & set(break_pred))
            len_ += len(break_label)
        accuracy = (correct / len_)
        self.log(f"{prefix}_break_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.bleu = BLEUScore(self.n_gram, self.smooth)
        self.wer = WordErrorRate()
        self.rouge = ROUGEScore()

    def initialize_model_specific_parameters(self):
        super().initialize_model_specific_parameters()
        if isinstance(self.tokenizer, MBartTokenizer):
            dm: TranslationDataModule = self.trainer.datamodule
            tgt_lang = dm.target_language
            # set decoder_start_token_id for MBart
            if self.model.config.decoder_start_token_id is None:
                assert tgt_lang is not None, "mBart requires --target_language"
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]

    @property
    def hf_pipeline_task(self) -> str:
        return "seq_to_seq"