import torch.nn as nn

from utils import *


class WordEncoder(nn.Module):
    def __init__(self,
                 embedding_model):
        super(WordEncoder, self).__init__()
        self.word_embed = WORD_EMB_SIZE
        self.hidden_size = HIDDEN_CVAE_SIZE
        self.embedding_model = embedding_model
        self.num_attention_heads = SEMANTIC_ATT_HEADS
        self.dropout_rate = DROPOUT

        self.position_embeddings_layer = nn.Linear(1, self.word_embed)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.linear_layer = nn.Linear(self.word_embed, self.hidden_size)
        self.relu = nn.ReLU()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                                    nhead=self.num_attention_heads,
                                                                    dim_feedforward=self.hidden_size,
                                                                    dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=ENCODER_LAYERS_NUM)

        self.device = set_device()
        self.init_weights()

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'batch' not in name and 'embedding_model' not in name\
                    and value.requires_grad and 'LayerNorm' not in name:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def forward(self, sentences):
        batch_size, num_sentences, num_words = sentences.size()
        cls_tokens = torch.empty(batch_size, num_sentences, self.hidden_size).to(self.device)

        for sent_id in range(num_sentences):
            sentence_embeddings = self.embedding_model(sentences[:, sent_id, :])

            position_ids = torch.arange(num_words).to(self.device)
            position_ids = position_ids.expand_as(sentences[:, sent_id, :]).unsqueeze(2)
            position_embeddings = self.position_embeddings_layer(position_ids.float())

            sentence_embeddings = sentence_embeddings + position_embeddings
            sentence_embeddings = self.relu(self.dropout(self.linear_layer(sentence_embeddings.float())))

            transformer_output = self.transformer_encoder(sentence_embeddings.permute(1, 0, 2))
            transformer_output = transformer_output.permute(1, 0, 2)

            cls_tokens[:, sent_id, :] = transformer_output[:, 0, :]  # pooled output

        return cls_tokens


