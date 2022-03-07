import nltk

from utils import *


class Vocabulary:
    def __init__(self, dataset, vocab_name):
        """
        :param vocab_name: tags_dict or paragraphs_dict
        """
        self.vocabulary = generate_vocabulary(dataset, vocab_name)
        self.word2idx = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        self.idx2word = {idx: w for (idx, w) in enumerate(self.vocabulary)}

    def __call__(self, word):
        if word == "" or word == " ":
            return self.word2idx[PAD]
        elif word not in self.word2idx:
            return self.word2idx[UNK]
        return self.word2idx[word]  # Returns the ID of the word in he vocabulary

    def __len__(self):
        return len(self.vocabulary)

    def vec2sent(self, word_ids):
        sampled_caption = []
        for word_id in word_ids:
            word = self.idx2word[word_id]
            if word == START or word == PAD:
                continue
            else:
                sampled_caption.append(word)

        if sampled_caption and sampled_caption[-1] != ".":
            sampled_caption += "."

        return sampled_caption

    def inv_vec2tag(self, tags):
        """
        Returns tags, given their encodings
        """
        sampled_tags = []
        for tag_id in tags:
            if tag_id != 0:
                tag = self.idx2word[tag_id]
                sampled_tags.append(tag)

        return sampled_tags

    def vec2tag(self, tag_ids):
        """
        Returns the tags given the tag id in the vocab
        """
        sampled_tags = []
        for tag_id in tag_ids[0]:
            tag = self.idx2word[int(tag_id)]
            if tag != " ":
                sampled_tags.append(tag)

        return sampled_tags

    def get_tags_weights(self, train_dataset):
        tags_counter = Counter()
        tags_weights = [0] * len(self.vocabulary)
        total_tags = 0

        for _, _, tags, _ in train_dataset:
            tags_counter.update(tags)
            total_tags += len(tags)

        for i, tags in enumerate(self.vocabulary):
            cnt = tags_counter[tags]
            tags_weights[i] = cnt / total_tags

        return tags_weights


def generate_vocabulary(dataset, vocab_name):
    print('Generating paragraphs vocabulary')
    paragraph_tokens_counter = Counter()

    for _, _, paragraphs, _ in dataset:
        tokens = nltk.word_tokenize(paragraphs)
        paragraph_tokens_counter.update(tokens)

    vocabulary = [word for word, cnt in paragraph_tokens_counter.items()
                  if cnt > WORD_OCCURRENCE_THRESHOLD]

    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, PAD)
    vocabulary.insert(1, START)
    vocabulary.insert(2, UNK)
    vocabulary.insert(3, END)

    if "" in vocabulary:
        vocabulary.remove("")

    print("Saving vocabulary...\nTotal num. of words: {}".format(len(vocabulary)))
    save_data_to_csv(vocab_name + ".csv", vocabulary)
    return vocabulary





