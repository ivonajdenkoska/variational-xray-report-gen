import torch.nn as nn
import torch.nn.functional as F

from utils import *


class WordDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_model):
        super(WordDecoder, self).__init__()
        self.embed_size = WORD_EMB_SIZE
        self.vocab_size = vocab_size
        self.latent_size = LATENT_SIZE
        self.hidden_size = HIDDEN_CVAE_SIZE
        self.dropout_rate = DROPOUT
        self.lstm_layers_num = LSTM_LAYERS_NUM

        self.embedding_model = embedding_model
        self.attention_lstm = AttentionLSTM(embed_size=self.embed_size, lang_lstm_size=self.latent_size,
                                            topic_size=self.latent_size, att_lstm_size=self.hidden_size)
        self.language_lstm = LanguageLSTM(att_lstm_size=self.hidden_size, visual_att_size=self.hidden_size,
                                          lang_lstm_size=self.latent_size)
        self.visual_attention = VisualAttention(visual_size=self.hidden_size, att_lstm_size=self.hidden_size,
                                                visual_att_size=self.hidden_size)

        # Parameter matrices
        self.latent_to_vocab = nn.Linear(self.latent_size, self.vocab_size)
        self.visual_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)

        self.hidden_to_latent = nn.Linear(self.hidden_size, self.latent_size)
        self.visual_to_latent = nn.Linear(self.hidden_size, self.latent_size)

        # for initializing hidden state and cell
        self.linear_hidden_state = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.linear_hidden_state_1 = nn.Linear(self.latent_size, self.latent_size, bias=False)

        self.dropout = nn.Dropout(p=DROPOUT)
        self.relu = nn.LeakyReLU()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.device = set_device()
        self.init_weights()

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'batch' not in name and 'embedding_model' not in name \
                    and value.requires_grad:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def init_hidden_states(self, topic):
        # for ablation study: use randomly initialized states
        # rand = torch.rand((topic.shape[0], self.latent_size)).to(self.device)

        c_init_a = self.linear_hidden_state(topic)
        h_init_a = torch.tanh(c_init_a)

        c_init_l = self.linear_hidden_state_1(topic)
        h_init_l = torch.tanh(c_init_l)

        return h_init_l, c_init_l, h_init_a, c_init_a

    # Defines the forward pass
    def forward(self, word_token, visual_features, hidden_states, topic_mean, first_time_step):
        if first_time_step:
            hidden_states = self.init_hidden_states(topic_mean)
        batch_size = word_token.shape[0]
        att_lstm_hidden, att_lstm_cell, lang_lstm_hidden, lang_lstm_cell = hidden_states
        word_embedding = self.embedding_model(word_token).view(batch_size, -1)

        visual_features_resized = self.relu(self.dropout(self.visual_to_hidden(visual_features)))
        att_lstm_hidden, att_lstm_cell = self.attention_lstm(att_lstm_hidden, att_lstm_cell, lang_lstm_hidden,
                                                             topic_mean, word_embedding)
        visual_context = self.visual_attention(visual_features_resized, att_lstm_hidden)

        lang_lstm_hidden, lang_lstm_cell = self.language_lstm(lang_lstm_hidden, lang_lstm_cell, att_lstm_hidden,
                                                              visual_context)

        outputs = self.latent_to_vocab(lang_lstm_hidden)
        hidden_states = [att_lstm_hidden, att_lstm_cell, lang_lstm_hidden, lang_lstm_cell]

        return outputs, hidden_states


class AttentionLSTM(nn.Module):
    def __init__(self, embed_size, lang_lstm_size, topic_size, att_lstm_size):
        super(AttentionLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size=lang_lstm_size + topic_size + embed_size,
                                     hidden_size=att_lstm_size,
                                     bias=True)

    def forward(self, att_lstm_hidden, att_lstm_cell, lang_lstm_hidden, topic_mean, word_embedding):
        input_feats = torch.cat((lang_lstm_hidden, topic_mean, word_embedding), dim=1)
        att_lstm_hidden, att_lstm_cell = self.lstm_cell(input_feats, (att_lstm_hidden, att_lstm_cell))

        return att_lstm_hidden, att_lstm_cell


class LanguageLSTM(nn.Module):
    def __init__(self, att_lstm_size, visual_att_size, lang_lstm_size):
        super(LanguageLSTM, self).__init__()
        self.latent_size = LATENT_SIZE
        self.lstm_cell = nn.LSTMCell(input_size=att_lstm_size + visual_att_size, hidden_size=lang_lstm_size,
                                     bias=True)

    def forward(self, lang_lstm_hidden, lang_lstm_cell, att_lstm_hidden, visual_context):
        input_feats = torch.cat((att_lstm_hidden, visual_context), dim=1)
        lang_lstm_hidden, lang_lstm_cell = self.lstm_cell(input_feats, (lang_lstm_hidden, lang_lstm_cell))

        return lang_lstm_hidden, lang_lstm_cell


class VisualAttention(nn.Module):
    def __init__(self, visual_size, att_lstm_size, visual_att_size):
        super(VisualAttention, self).__init__()
        self.visual_to_visual_att = nn.Linear(in_features=visual_size, out_features=visual_att_size, bias=False)
        self.att_lstm_to_visual_att = nn.Linear(in_features=att_lstm_size, out_features=visual_att_size, bias=False)
        self.attention_linear = nn.Linear(in_features=visual_att_size, out_features=1)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, visual_features, att_lstm_hidden):
        batch_size, num_local_features, _ = visual_features.size()

        att_lstm_emb = self.relu(self.dropout(self.att_lstm_to_visual_att(att_lstm_hidden))).unsqueeze(1)
        visual_feats_emb = self.relu(self.dropout(self.visual_to_visual_att(visual_features)))
        all_feats_emb = visual_feats_emb + att_lstm_emb.expand_as(visual_feats_emb)

        activate_feats = self.dropout(self.tanh(all_feats_emb))
        normed_attention = self.softmax(self.attention_linear(activate_feats))

        # weighted sum
        weighted_features = normed_attention * visual_features
        attended_visual_features = weighted_features.sum(dim=1)

        return attended_visual_features


# For ablation study
class WordDecoderWithoutAttention(nn.Module):
    def __init__(self, vocab_size, embedding_model):
        super(WordDecoderWithoutAttention, self).__init__()
        self.embed_size = WORD_EMB_SIZE
        self.vocab_size = vocab_size
        self.latent_size = LATENT_SIZE
        self.hidden_size = HIDDEN_CVAE_SIZE
        self.dropout_rate = DROPOUT
        self.lstm_layers_num = LSTM_LAYERS_NUM

        self.embedding_model = embedding_model
        self.lstm_cell = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.latent_size,
                                     bias=True)

        # Parameter matrices
        self.latent_to_vocab = nn.Linear(self.latent_size, self.vocab_size)

        # for initializing hidden state and cell
        self.linear_hidden_state = nn.Linear(self.latent_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(p=DROPOUT)
        self.relu = nn.LeakyReLU()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.device = set_device()
        self.init_weights()

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'batch' not in name and 'embedding_model' not in name \
                    and value.requires_grad:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def init_hidden_states(self, topic):
        batch_size = topic.shape[0]
        rand = torch.rand((batch_size, self.latent_size)).to(self.device)  # for ablation study
        c_init_a = self.linear_hidden_state(topic)
        h_init_a = torch.tanh(c_init_a)

        return h_init_a, c_init_a

    # Defines the forward pass
    def forward(self, word_token, hidden_states, topic_mean, first_time_step):
        if first_time_step:
            hidden_states = self.init_hidden_states(topic_mean)

        embeddings = self.embedding_model(word_token).squeeze(1)

        output_states, hidden_states = self.lstm_cell(embeddings, hidden_states)
        outputs = self.latent_to_vocab(output_states)

        return outputs, (output_states, hidden_states)


def sampling_with_t(logits, temperature=T):
    logits /= temperature
    probs = F.softmax(logits, dim=-1)

    predicted_word = torch.multinomial(probs, num_samples=1)

    return predicted_word


def top_k_sampling(logits, temperature=T, k=K, m=None):
    """
    :param logits: softmax probabilities
    :param temperature: temperature
    :param k: k for top-k sampling
    :param m: mass of original dist to interpolate
    :return: predicted_token
    """
    logits /= temperature
    probs = F.softmax(logits, dim=-1)

    indices_to_remove = probs < torch.topk(probs, k)[0][..., -1, None]
    top_k_probs = probs
    top_k_probs[indices_to_remove] = 0
    top_k_probs.div_(top_k_probs.sum())

    if m is not None:
        top_k_probs.mul_(1 - m)
        top_k_probs.add_(probs.mul(m))

    predicted_word = torch.multinomial(top_k_probs, num_samples=1)
    return predicted_word


def nucleus_sampling(logits, p=P, temperature=T):
    """
    :param logits: softmax probabilities
    :param p: p for Nucleus (top-p) sampling
    :param temperature:
    :return:
    """
    logits /= temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p

    sorted_samp_probs = sorted_probs.clone()
    sorted_samp_probs[sorted_indices_to_remove] = 0
    sorted_samp_probs.div_(sorted_samp_probs.sum())  # normalize

    sorted_next_indices = torch.multinomial(sorted_samp_probs, num_samples=1).view(-1, 1)
    predicted_word_id = sorted_indices.gather(1, sorted_next_indices)

    return predicted_word_id






