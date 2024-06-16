from typing import Dict, Union
import os

import torch
from torch import nn

from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    AutoModel,
    AutoTokenizer,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from ..losses.ordinal_cross_entropy_loss import OrdinalCrossEntropyLoss


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        is_causal: bool,
        is_preprocessed: bool,
        custom_data_encoder_path: str,
        precision: Union[int, str],
        mode: str,
        quantization_type: str,
        quantization_config: BitsAndBytesConfig,
        peft_type: str,
        peft_config: LoraConfig,
        dropout_ratio: float,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.is_causal = is_causal
        self.is_preprocessed = is_preprocessed
        self.custom_data_encoder_path = custom_data_encoder_path

        self.attn_implementation = None
        if precision == 32 or precision == "32":
            self.precision = torch.float32
        elif precision == 16 or precision == "16":
            self.precision = torch.float16
            if self.is_causal:
                self.attn_implementation = "flash_attention_2"
        elif precision == "bf16":
            self.precision = torch.bfloat16
            if self.is_causal:
                self.attn_implementation = "flash_attention_2"
        else:
            self.precision = "auto"

        self.mode = mode
        self.quantization_type = quantization_type
        self.quantization_config = None
        self.device_map = None
        if self.quantization_type == "quantization":
            self.quantization_config = quantization_config
            if self.mode in ["test" "predict"]:
                self.quantization_config.load_in_4bit = False
            else:
                self.quantization_config.load_in_4bit = True
            self.quantization_config.bnb_4bit_compute_dtype = self.precision
            self.device_map = {
                "": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))
            }
        if self.quantization_type not in ["origin", "quantization"]:
            raise ValueError(f"Invalid quantization type: {self.quantization_type}.")

        self.peft_type = peft_type
        self.peft_config = peft_config
        if self.mode in ["test" "predict"]:
            self.peft_config.inference_mode = True

        self.model = self.get_model()
        self.dropout = nn.Dropout(
            p=dropout_ratio,
        )
        self.classifier = nn.Linear(
            self.model.config.hidden_size,
            num_labels,
        )
        self.criterion = OrdinalCrossEntropyLoss()

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        label = encoded["labels"]
        del encoded["labels"]
        output = self.model(**encoded)
        logit = self.classifier(
            self.dropout(
                output.last_hidden_state[:, 0, :],
            )
        )
        loss = OrdinalCrossEntropyLoss(
            logit,
            label,
        )
        return {
            "logit": logit,
            "loss": loss,
        }

    def get_model(self) -> PreTrainedModel:
        model = AutoModel.from_pretrained(
            self.pretrained_model_name,
            output_hidden_states=False,
            torch_dtype=self.precision,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
            device_map=self.device_map,
        )

        if self.is_preprocessed:
            data_encoder = AutoTokenizer.from_pretrained(
                f"{self.custom_data_encoder_path}/{self.pretrained_model_name}",
                use_fast=True,
            )
            if data_encoder.pad_token_id is None:
                data_encoder.pad_token_id = data_encoder.eos_token_id
            model.resize_token_embeddings(len(data_encoder))

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

        if self.quantization_type == "quantization" and self.mode not in [
            "test",
            "predict",
        ]:
            model = prepare_model_for_kbit_training(model)

        if self.peft_type == "lora":
            model.enable_input_require_grads()
            model = get_peft_model(model, self.peft_config)
        if self.peft_type not in ["origin", "lora"]:
            raise ValueError(f"Invalid PEFT type: {self.peft_type}.")
        return model
