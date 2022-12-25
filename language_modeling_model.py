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
from typing import Any, Type

import torch
import transformers
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.rouge import ROUGEScore
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.seq2seq.utils import _pad_tensors_to_max_len


class LanguageModelingTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self, *args, downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForCausalLM,
            val_target_max_length: int = 128,
            num_beams: int = 1,
            n_gram: int = 1,
            smooth: bool = True,
            **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.val_target_max_length = val_target_max_length
        self.num_beams = num_beams
        self.n_gram = n_gram
        self.smooth = smooth
        self.bleu = BLEUScore(self.n_gram, self.smooth)
        self.wer = WordErrorRate()
        self.rouge = ROUGEScore()

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, prefix: str, batch):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        self.compute_generate_metrics(batch, prefix)
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        return {'loss': loss, 'labels': labels, 'input_ids': input_ids}

    def training_step(self, batch):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, dataloader_idx=0):
        return self._step("val", batch)

    def test_step(self, batch, dataloader_idx=0):
        return self._step("test", batch)

    def validation_epoch_end(self, outputs):
        print(self.tokenize_labels(outputs[0]['labels'].unsqueeze(0)))
        print(self.generate(outputs[0]['input_ids'].unsqueeze(0)))

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"])
        bleu_result = self.bleu(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_bleu_score", bleu_result, on_step=False, on_epoch=True, prog_bar=True)
        wer_result = self.wer(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_wer_score", wer_result, on_step=False, on_epoch=True, prog_bar=True)
        rouge_result = self.rouge(preds=pred_lns, target=tgt_lns)
        self.log_dict({f"{prefix}_{k}": v for k, v in rouge_result.items()}, on_step=False, on_epoch=True,
                      prog_bar=True)

        correct = 0
        len_ = 0
        for tgt, pred in zip(tgt_lns, pred_lns):
            break_label = [i for i, j in enumerate(tgt.split()) if j == 'break']
            break_pred = [i for i, j in enumerate(pred.split()) if j == 'break']
            correct += len(set(break_label) & set(break_pred))
            len_ += len(break_label)
        accuracy = (correct / len_)
        self.log(f"{prefix}_break_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def generate(self, input_ids: torch.Tensor):
        max_length = self.val_target_max_length if self.val_target_max_length else self.model.config.max_length
        num_beams = self.num_beams if self.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=input_ids, max_length=max_length, num_beams=num_beams
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = _pad_tensors_to_max_len(
                model_cfg=self.model.config, tensor=generated_tokens, max_length=max_length
            )
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str

    def tokenize_labels(self, labels: torch.Tensor):
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return [str.strip(s) for s in label_str]

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"

    def inference(self, text: str, device: torch.device = torch.device("cpu"), **kwargs) -> Any:
        if self.tokenizer is None:
            raise MisconfigurationException(
                "A tokenizer is required to use the `generate` function. "
                "Please pass a tokenizer `LanguageModelingTransformer(tokenizer=...)`."
            )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        return self.model.generate(inputs["input_ids"], **kwargs)