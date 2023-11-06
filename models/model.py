from models.config import ImageTextRetrievalConfig
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.models.bert import BertModel
from transformers.models.resnet import ResNetModel
from typing import Optional, Literal
from models.loss import loss
from transformers import BertConfig, ResNetConfig

class ImageTextRetrievalPreTrainedModel(PreTrainedModel):
    
    config_class = ImageTextRetrievalConfig
    base_model_prefix = "bert, resnet"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        factor = self.config.initializer_factor
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)  
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
class ImageTextRetrieval(ImageTextRetrievalPreTrainedModel):
    config_class = ImageTextRetrievalConfig
    
    def __init__(self, config: ImageTextRetrievalConfig):
        super().__init__(config)
        
        text_config = config.text_config
        image_config = config.image_config
        
        self.projection_dim = config.projection_dim
        
        self.text_embed_dim = text_config["hidden_size"]
        self.image_embed_dim = image_config["hidden_sizes"][-1]
        
        self.text_encoder = BertModel(BertConfig(**text_config), add_pooling_layer=False)
        self.image_encoder = ResNetModel(ResNetConfig(**image_config))
        
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        
        self.post_init()
        
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

         
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Tensor = None
        ):
        
        text_embs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        ).last_hidden_state[:, 0, :]
        
        image_embs = self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
        ).pooler_output[:, :, 0, 0]

        text_embs = self.text_projection(text_embs)
        image_embs = self.image_projection(image_embs)
        
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embs, image_embs.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        _loss = loss(logits_per_text)
        output = (logits_per_image, logits_per_text, text_embs, image_embs)
        return _loss, output