from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer
import torch
from models.model import ImageTextRetrieval, ImageTextRetrievalConfig
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import TrainingArguments, Trainer, HfArgumentParser
import logging
import os
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass, field
import transformers
from typing import Union
from utils.data_collator import DataCollatorForImageTextRetrieval


@dataclass
class TrainArguments(TrainingArguments):
    output_dir: str = "runs/"
    do_train: bool = True
    do_eval: bool = False
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 5.0



if __name__ == "__main__":
    
    parser = HfArgumentParser((TrainArguments,))
    train_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("poloclub/diffusiondb", "2m_random_5k")

    model = ImageTextRetrieval.from_pretrained("ohgnues/ImageTextRetrieval")
    model.to("cuda")
    processor = AutoImageProcessor.from_pretrained("ohgnues/ImageTextRetrieval")
    tokenizer = AutoTokenizer.from_pretrained("ohgnues/ImageTextRetrieval")
    
    text_column_name = "prompt"
    image_column_name = "image"


    def example_function(examples):

        tokenized_text = tokenizer(
            examples[text_column_name],
            truncation=True,
            padding="max_length",
            max_length=100,
            return_tensors="pt"
        ).to("cuda")

        processed_image = processor(examples[image_column_name], return_tensors="pt").to("cuda")

        tokenized_text.update(processed_image)

        return tokenized_text

    dataset = dataset.map(example_function, batched=True, remove_columns=dataset["train"].column_names)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
    )

    trainer.train()
