# multi-modal-retrieval

This repository contains code for multi modal retrieval

# Data

## Sample Data

The pretraining was conducted using the dataset from Hugging Face's ["poloclub/diffusiondb"](https://huggingface.co/datasets/poloclub/diffusiondb) dataset.

I used 50k randomly sampled images and prompts for my project.

If you want to use a different dataset, follow the steps below

## Data Format

Only images and the corresponding text for those images are necessary, and other elements are irrelevant. In this case, the text can serve as prompts or captions for the images.

You specify the names of the columns for images and text in the training command.

```bash
python3 train.py --text_column_name text --image_column_name img
```

# Pretrained models

Pretrained models can be downloaded [huggingface](https://huggingface.co/ohgnues/ImageTextRetrieval) or Specify the model name "ohgnues/ImageTextRetrieval" in the training command.

```bash
python3 train.py --pretrained_model_name_or_path ohgnues/ImageTextRetrieval
```

The model "ohgnues/ImageTextRetrieval" was trained for 10 epochs using a Tesla P100 GPU.

# Usage

## Train

```bash
python3 train.py --name 2m_random_50k --cache_dir /data/.cache --max_length 100 --num_train_epochs 10
```
For detailed instructions, please refer to the official Hugging Face documentation or consult the dataclass within the "train.py" script.

## Encode
```python
    def encode(self, model_name: Literal["text", "image"],
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Tensor = None
            ):
        
        if model_name == "text":
            self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ).last_hidden_state[:, 0, :]
        
        elif model_name == "image":
            self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            ).pooler_output[:, :, 0, 0]
```