from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pack_padded_sequence

from cvae_module.cnn_att_encoder import *
from cvae_module.cvae_models import *
from cvae_module.word_decoder import *
from cvae_module.word_encoder import *
from utils import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        with open(CURR_DATA_PATH + "vocab.pkl", 'rb') as vocab_pickle_file:
            self.vocab = pickle.load(vocab_pickle_file)

        self.vocab_size = len(self.vocab)
        self.labels_vocab = get_labels_vocab()
        self.num_labels = len(self.labels_vocab)

        self.device = set_device()
        self.hidden_size = HIDDEN_CVAE_SIZE
        self.embed_size = WORD_EMB_SIZE
        self.latent_size = LATENT_SIZE
        self.epsilon = 1e-10

        self.prior = Prior()
        self.approx_posterior = ApproximatePosterior()
        self.cnn_att_encoder = CNNAttEncoder()
        self.embedding_model = self.load_embedding_layer()
        self.word_encoder = WordEncoder(embedding_model=self.embedding_model)
        self.word_decoder = WordDecoder(vocab_size=self.vocab_size, embedding_model=self.embedding_model)

        if USE_PRETRAINED_CHEXPERT:
            self.load_pre_trained_cnn()

        self.cross_entropy = nn.CrossEntropyLoss()

        # For multiple GPUs
        if torch.cuda.device_count() > 1:
            gpus_num = torch.cuda.device_count()
            print("Using {} GPUs!".format(gpus_num))
            self.prior = nn.DataParallel(self.prior)
            self.approx_posterior = nn.DataParallel(self.approx_posterior)
            self.cnn_att_encoder = nn.DataParallel(self.cnn_att_encoder)
            self.embedding_model = nn.DataParallel(self.embedding_model)
            self.word_encoder = nn.DataParallel(self.word_encoder)
            self.word_decoder = nn.DataParallel(self.word_decoder)
            print("The used GPUs are: {}".format(self.cnn_att_encoder.device_ids))

    def forward(self, image, paragraph, paragraph_lengths):
        batch_size, num_sentences, num_words = paragraph.shape
        rec_terms, kl_terms, labels_loss_terms = torch.empty(num_sentences), torch.empty(num_sentences), \
                                                 torch.empty(num_sentences)
        all_topics = torch.empty(batch_size, num_sentences, self.latent_size)

        img_tokens, visual_features = self.cnn_att_encoder(image)
        cls_tokens = self.word_encoder(paragraph)

        for sentence_id in range(num_sentences):
            mu_prior, log_var_prior = self.prior(img_tokens[:, sentence_id, :])
            p_normal = Normal(loc=mu_prior + self.epsilon, scale=log_var_prior + self.epsilon)
            mu_posterior, log_var_posterior = self.approx_posterior(cls_tokens[:, sentence_id, :])
            q_normal = Normal(loc=mu_posterior+self.epsilon, scale=log_var_posterior+self.epsilon)

            kl_terms[sentence_id] = torch.mean(kl_divergence(q_normal, p_normal))

            topics = torch.empty(batch_size, NUM_SAMPLES_INFERENCE, self.latent_size).to(self.device)
            for sample_id in range(NUM_SAMPLES):
                topic = reparametrize(mu_posterior, log_var_posterior)
                topics[:, sample_id, :] = topic

            topic_mean = torch.mean(topics, dim=1).to(self.device)
            all_topics[:, sentence_id, :] = topic_mean
            hidden_states = None
            words_dists = torch.empty(batch_size, num_words, self.vocab_size).to(self.device)

            for word_id in range(MAX_WORDS_IN_SENT):
                first_time_step = word_id == 0
                word_dist, hidden_states = self.word_decoder(paragraph[:, sentence_id, word_id],
                                                             visual_features[:, sentence_id, :],
                                                             hidden_states, topic_mean, first_time_step)
                words_dists[:, word_id, :] = word_dist

            paragraph_packed = pack_padded_sequence(paragraph[:, sentence_id],
                                                    paragraph_lengths[:, sentence_id], batch_first=True,
                                                    enforce_sorted=False)
            word_dist_packed = pack_padded_sequence(words_dists, paragraph_lengths[:, sentence_id],
                                                    batch_first=True, enforce_sorted=False)

            rec_term = self.cross_entropy(word_dist_packed.data, paragraph_packed.data)
            rec_terms[sentence_id] = rec_term

        reconstruction_term = torch.mean(rec_terms).to(self.device)
        avg_kl_term = torch.mean(kl_terms).to(self.device)

        return reconstruction_term, avg_kl_term

    def inference(self, image, start_tokens):
        num_sentences = MAX_SENT_NUM
        batch_size = image.shape[0]
        generated_paragraph_top_k = []
        img_tokens, visual_features = self.cnn_att_encoder(image)

        for sentence_id in range(num_sentences):
            predicted_word = start_tokens
            topics = torch.zeros(batch_size, NUM_SAMPLES_INFERENCE, self.latent_size)
            mu_prior, log_var_prior = self.prior(img_tokens[:, sentence_id, :])

            for sample_id in range(NUM_SAMPLES_INFERENCE):
                topic = reparametrize(mu_prior, log_var_prior)
                topics[:, sample_id, :] = topic

            topic = torch.mean(topics, dim=1).to(self.device)

            pred_sentence_top_k, att_maps = self.generate_sentence("top_k", start_tokens, predicted_word,
                                                                   visual_features[:, sentence_id, :], topic)
            generated_paragraph_top_k += [pred_sentence_top_k]

        return generated_paragraph_top_k

    def generate_sentence(self, sample_method, start_tokens, predicted_word, visual_features, topic):
        batch_size = visual_features.shape[0]
        max_generated_words = MAX_WORDS_INFERENCE
        pred_sentence = np.zeros(max_generated_words)
        attention_weights_sentence = np.zeros((batch_size, max_generated_words, 49))
        pred_sentence[0] = start_tokens.item()
        hidden_states = None

        for word_id in range(1, max_generated_words):
            first_time_step = word_id == 1
            word_dist, hidden_states, attention_weights = self.word_decoder(predicted_word, visual_features,
                                                                            hidden_states, topic, first_time_step)
            attention_weights_sentence[:, word_id, :] = attention_weights.cpu()
            # to ensure not to repeat same words
            word_dist = word_dist.squeeze(1)
            word_dist.squeeze()[predicted_word.item()] = torch.tensor(0).to(self.device)

            if sample_method == "temp":
                predicted_word = sampling_with_t(word_dist)
            elif sample_method == "top_k":
                predicted_word = top_k_sampling(word_dist)
            elif sample_method == "nucleus":
                predicted_word = nucleus_sampling(word_dist)

            pred_sentence[word_id] = predicted_word.item()

        return pred_sentence, attention_weights_sentence

    def beam_search(self, start_tokens, visual_features, hidden_states, topic_mean, k=BEAM_SIZE):
        """ Adapted from:
        Title: conditional-vae
        Availability: https://github.com/artidoro/conditional-vae """
        init_prob = 0
        best_options = [(init_prob, [start_tokens], hidden_states)]

        for word_id in range(1, MAX_WORDS_INFERENCE):  # maximum target length
            options = []  # candidates

            for word_prob, word_tokens, hidden_states in best_options:
                last_word_token = word_tokens[-1]

                if last_word_token.item() == self.vocab(STOP):
                    options.append((word_prob, word_tokens, hidden_states))
                else:
                    last_word_token = torch.tensor([last_word_token], requires_grad=False).to(self.device)
                    first_time_step = word_id == 1
                    logits, hidden_states = self.word_decoder(last_word_token, visual_features, hidden_states, topic_mean,
                                                              first_time_step)
                    words_probs = F.softmax(logits/T, dim=-1).squeeze()

                    # Add top k candidates to options list for next word
                    top_candidates = words_probs.multinomial(num_samples=k)
                    for index in top_candidates:
                        option = (words_probs[index].item() + word_prob,
                                  word_tokens + [index],
                                  hidden_states)
                        options.append(option)

            options.sort(key=lambda x: x[0], reverse=True)  # sort by word_prob
            best_options = options[:k]  # place top candidates in beam

        word_ids = [[word_id.item() for word_id in word_ids[1]] for word_ids in best_options]
        return word_ids

    def save_model(self):
        """
        Save all components of the model as dictionary
        """
        torch.save({'cnn_att_encoder': self.cnn_att_encoder.state_dict(),
                    'word_encoder': self.word_encoder.state_dict(),
                    'embedding_model': self.embedding_model.state_dict(),
                    'word_decoder': self.word_decoder.state_dict(),
                    'prior': self.prior.state_dict(),
                    'posterior': self.approx_posterior.state_dict(),
                    }, os.path.join(MODELS_PATH, "{}".format(MODEL_NAME)))
        print("Model saved on path {}".format(MODELS_PATH))

    def load_model(self):
        model_dict = torch.load(MODELS_PATH + MODEL_NAME, map_location=torch.device(self.device))
        if USE_PRETRAINED_CHEXPERT:
            self.load_pre_trained_cnn()
        else:
            self.cnn_att_encoder.load_state_dict(model_dict['cnn_att_encoder'])
        self.word_encoder.load_state_dict(model_dict['word_encoder'])
        self.embedding_model.load_state_dict(model_dict['embedding_model'])
        self.prior.load_state_dict(model_dict['prior'])
        self.word_decoder.load_state_dict(model_dict['word_decoder'])

    def load_embedding_layer(self, trainable=False):
        with open(CURR_DATA_PATH + "bio_word2vec_weights.pkl", 'rb') as bio_word2vec_weights:
            weights_matrix = pickle.load(bio_word2vec_weights)

        weights_matrix = torch.tensor(weights_matrix).float().to(self.device)
        embedding_layer = nn.Embedding.from_pretrained(weights_matrix)
        if trainable:
            embedding_layer.weight.requires_grad = True

        return embedding_layer

    def load_pre_trained_cnn(self):
        model_dict = torch.load(MODELS_PATH + PRETRAINED_MODEL_NAME, map_location=torch.device(self.device))
        model_dict = model_dict['cnn_att_encoder']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v

        self.cnn_att_encoder.load_state_dict(new_state_dict)








