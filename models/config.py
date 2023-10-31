from transformers import ResNetConfig, BertConfig, PretrainedConfig

class ImageTextRetrievalConfig(PretrainedConfig):
    
    model_type = "bert, resnet"
    
    def __init__(
        self, 
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        image_hidden_act="relu",
        downsample_in_first_stage=False,
        out_features=None,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        text_hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        projection_dim=512, 
        logit_scale_init_value=2.6592, 
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        self.text_config = {
            'vocab_size':vocab_size,
            'hidden_size':hidden_size,
            'num_hidden_layers':num_hidden_layers,
            'num_attention_heads':num_attention_heads,
            'intermediate_size':intermediate_size,
            'text_hidden_act':text_hidden_act,
            'hidden_dropout_prob':hidden_dropout_prob,
            'attention_probs_dropout_prob':attention_probs_dropout_prob,
            'max_position_embeddings':max_position_embeddings,
            'type_vocab_size':type_vocab_size,
            'initializer_range':initializer_range,
            'layer_norm_eps':layer_norm_eps,
            'pad_token_id':pad_token_id,
            'position_embedding_type':position_embedding_type,
            'use_cache':use_cache,
            'classifier_dropout':classifier_dropout,
            }
        
        self.image_config = {
            'num_channels':num_channels,
            'embedding_size':embedding_size,
            'hidden_sizes':hidden_sizes,
            'depths':depths,
            'layer_type':layer_type,
            'image_hidden_act':image_hidden_act,
            'downsample_in_first_stage':downsample_in_first_stage,
            'out_features':out_features,
            }

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0