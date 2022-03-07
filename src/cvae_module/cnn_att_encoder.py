import os

import torch.nn as nn
import torchvision.models as models

from utils import *

os.environ['TORCH_HOME'] = MODELS_PATH


class CNNAttEncoder(nn.Module):
    def __init__(self):
        super(CNNAttEncoder, self).__init__()
        self.pooled_size = 7
        self.v_hidden_size = 1024
        self.hidden_size = HIDDEN_CVAE_SIZE
        self.num_attention_heads = VISUAL_ATT_HEADS
        self.dropout_rate = DROPOUT

        cnn = models.densenet121(pretrained=True)
        modules = list(cnn.children())[:-1]
        self.cnn_encoder = nn.Sequential(*modules)

        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(self.pooled_size, self.pooled_size))

        self.v_hidden_to_hidden = nn.Linear(in_features=self.v_hidden_size, out_features=self.hidden_size)
        self.v_hidden_to_hidden_2 = nn.Linear(in_features=self.v_hidden_size, out_features=self.hidden_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.LeakyReLU()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.v_hidden_size,
                                                                    nhead=self.num_attention_heads,
                                                                    dim_feedforward=self.hidden_size,
                                                                    dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=ENCODER_LAYERS_NUM)
        self.init_weights()
        self.freeze_layers(fine_tune=False)
        self.device = set_device()

    def freeze_layers(self, fine_tune):
        """Freeze the conv layers, because we only extract the learned features"""
        for param in self.cnn_encoder.parameters():
            param.requires_grad = False
        # If fine-tuning, only fine-tune conv blocks 2 through 4
        if fine_tune:
            for c in list(self.cnn_encoder.children())[5:]:
                for param in c.parameters():
                    param.requires_grad = True

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'downsample' not in name and 'bn' not in name \
                    and value.requires_grad:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def forward(self, images):
        batch_size = images.shape[0]
        num_sentences = MAX_SENT_NUM
        img_tokens = torch.empty(batch_size, num_sentences, self.hidden_size).to(self.device)
        self_attended_features = torch.empty(batch_size, num_sentences, 2*self.pooled_size**2,
                                             self.hidden_size).to(self.device)

        extracted_features = self.cnn_encoder(images[:, 0])
        pooled_features = self.adaptive_avg_pooling(extracted_features).permute(0, 2, 3, 1)
        pooled_features_dim = pooled_features.shape[-1]
        spatial_features_1 = pooled_features.reshape(batch_size, -1, pooled_features_dim)

        extracted_features = self.cnn_encoder(images[:, 1])
        pooled_features = self.adaptive_avg_pooling(extracted_features).permute(0, 2, 3, 1)
        pooled_features_dim = pooled_features.shape[-1]
        spatial_features_2 = pooled_features.reshape(batch_size, -1, pooled_features_dim)

        img_token = torch.cat((spatial_features_1, spatial_features_2), 1).mean(dim=1).unsqueeze(1)
        img_token_features = torch.cat((img_token, spatial_features_1, spatial_features_2), dim=1).to(self.device)

        for sent_id in range(num_sentences):
            transformer_output = self.transformer_encoder(img_token_features.permute(1, 0, 2))
            transformer_output = transformer_output.permute(1, 0, 2)

            pooled_img_token = transformer_output[:, 0, :]
            pooled_img_token = self.relu(self.dropout(self.v_hidden_to_hidden(pooled_img_token)))

            img_tokens[:, sent_id, :] = pooled_img_token
            self_attended_features[:, sent_id, :] = self.relu(self.dropout(self.v_hidden_to_hidden_2(
                transformer_output[:, 1:, :])))

        return img_tokens, self_attended_features







