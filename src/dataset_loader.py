import nltk
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop, \
    ColorJitter

from config import *
from utils import get_labels_vocab


class ChestXRaysDataSet(data.Dataset):
    def __init__(self, dataset, paragraphs_vocab, mode):
        self.dataset = dataset
        self.paragraphs_vocab = paragraphs_vocab

        image_transforms = {
            'train':  # Train uses data augmentation
            Compose([
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                Resize(size=256),
                CenterCrop(size=224),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val':  # Validation does not use augmentation
            Compose([
                Resize(size=256),
                CenterCrop(size=224),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test':  # Test does not use augmentation
            Compose([
                Resize(size=256),
                CenterCrop(size=224),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.img_transform = image_transforms[mode]

    def __getitem__(self, index):
        idx = self.dataset[index][0]
        image_dir = self.dataset[index][1]

        image_1 = Image.open(image_dir[0]).convert('RGB')
        transformed_image_1 = self.img_transform(image_1)

        image_2 = Image.open(image_dir[1]).convert('RGB')
        transformed_image_2 = self.img_transform(image_2)

        encoded_paragraph, labels = [], []

        if self.dataset[index][2] is not None:
            # Encoding paragraphs into 1-hot encoded vectors
            paragraph = self.dataset[index][2]
            sentences = paragraph.split(". ")

            for i, sentence in enumerate(sentences):
                words = nltk.word_tokenize(sentence)
                encoded_sentence = list()
                encoded_sentence.append(self.paragraphs_vocab(START))

                for j, word in enumerate(words):
                    encoded_sentence.append(self.paragraphs_vocab(word))
                encoded_sentence.append(self.paragraphs_vocab(END))
                encoded_paragraph.append(encoded_sentence)

        if self.dataset[index][3]:
            labels = self.dataset[index][3]

        return idx, transformed_image_1, transformed_image_2, encoded_paragraph, labels, image_dir

    def __len__(self):
        return len(self.dataset)


def collate_fn(data):
    """
    A custom collate_fn method
    :param data: A list of tuples
            - idx,
            - images,
            - tags_distribution,
            - paragraphs
    :return: Batch of examples
    """
    idx, transformed_image_1, transformed_image_2, paragraphs, labels, image_dir = zip(*data)

    transformed_image_1 = torch.stack(transformed_image_1, 0)
    transformed_image_2 = torch.stack(transformed_image_2, 0)
    images = torch.stack((transformed_image_1, transformed_image_2), 1)
    batch_size = images.shape[0]

    max_sentence_num = MAX_SENT_NUM
    max_word_num = MAX_WORDS_IN_SENT
    labels_len = len(get_labels_vocab())

    encoded_paragraphs = np.zeros((batch_size, max_sentence_num, max_word_num))
    sentences_lengths = np.ones((batch_size, max_sentence_num))
    encoded_labels = np.zeros((batch_size, labels_len))

    for i, paragraph in enumerate(paragraphs):
        for j, sentence in enumerate(paragraph):
            if j >= max_sentence_num:
                break
            if len(sentence) > max_word_num:
                sentence = np.concatenate((sentence[:max_word_num-1], [sentence[len(sentence)-1]]))
            encoded_paragraphs[i, j, :len(sentence)] = sentence
            sentences_lengths[i, j] = 1 if len(sentence) < 1 else len(sentence)

    if labels:
        for i, labels in enumerate(labels):
            if not isinstance(labels[0], str):
                encoded_labels[i, :len(labels)] = labels

    return idx, images, encoded_paragraphs, sentences_lengths, encoded_labels, image_dir
