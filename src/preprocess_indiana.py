"""
Preprocessing script
"""
import glob
import os
import xml.etree.ElementTree as ET

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

from utils import *
from vocabulary import Vocabulary


def preprocess_tokens(paragraph):
    """
    Lowercasing and removing non-alpha characters
    Returns the paragraph as list of tokens, including the dot as a separate token.
    :param paragraph: list of tokens
    :return: preprocessed_tokens
    """
    tokens = nltk.word_tokenize(paragraph)
    preprocessed_tokens = []
    for token in tokens:
        contains_letters = all(map(str.isalpha, token))
        if contains_letters:
            token = token.lower().strip()
            preprocessed_tokens.append(token)
        elif token == "." and preprocessed_tokens and preprocessed_tokens[-1] != ".":
            preprocessed_tokens.append(token)

    return preprocessed_tokens


def preprocess_tags(tags):
    """
    Lowercase tags and check for empty ones
    :param tags: list of tags
    :return: preprocessed_tags
    """
    preprocessed_tags = []
    for tag in tags:
        if tag != "":
            tag = tag.lower().strip()
            preprocessed_tags.append(tag)

    return preprocessed_tags


def parse_paragraphs():
    """
    Parsing of the XML reports.
    Extracts FINDINGS + IMPRESSIONS sections and the MeSH tags.
    :return: dictionaries paragraphs_dict, tags_dict
    """
    paragraphs = sorted(os.listdir(INDIANA_REPORTS_PATH))
    paragraphs_dict, labels_dict = {}, {}

    for paragraph_name in paragraphs:
        tree = ET.parse(os.path.join(INDIANA_REPORTS_PATH, paragraph_name))
        root = tree.getroot()
        paragraph = ""
        labels_mesh = []

        idx = root.find("uId").attrib['id']
        # XML parsing of the reports
        for plain_paragraph in root.find('MedlineCitation').find("Article").find("Abstract")\
                .findall("AbstractText"):
            label = plain_paragraph.attrib['Label']
            if label == "FINDINGS" and plain_paragraph.text is not None:
                paragraph = plain_paragraph.text  # findings
            elif label == "IMPRESSION" and plain_paragraph.text is not None:
                impression = plain_paragraph.text if plain_paragraph.text[-1] == STOP else plain_paragraph.text + STOP
                paragraph = impression + SPACE + paragraph

        if not paragraph:  # if there are no findings or impression, we don't consider the sample
            continue

        preprocessed_para = preprocess_tokens(paragraph)
        paragraph = ' '.join(preprocessed_para)
        paragraphs_dict[idx] = paragraph

        # XML parsing of the MeSH labels
        for labels in root.find("MeSH").findall("major"):
            tokenized_labels = labels.text.split("/")  # tags are separated by "/" and ","
            labels = preprocess_tags(tokenized_labels)
            labels_mesh += labels

        labels_dict[idx] = labels_mesh

    return paragraphs_dict, labels_dict


def create_complete_dataset():
    """
    Creates list of tuples
    :return: tuples of image ID and frontal + lateral image
    """
    tuples = []
    paragraphs_dict, labels_dict = parse_paragraphs()
    img_ids = paragraphs_dict.keys()

    for idx in img_ids:
        img_path = PROCESSED_IMAGES_PATH + idx
        images = glob.glob(img_path + '_*')  # frontal and lateral x-ray images
        if images:
            images = sorted(images)
            if len(images) != 2:
                continue
            sample_id = get_id_from_string(idx)
            tuples.append((sample_id, images, paragraphs_dict[idx], labels_dict[idx]))

    return tuples


def count_sentences_words():
    sentences_counter = Counter()
    words_counter = Counter()
    sentences_num = 0
    words_num = 0

    with open(INDIANA_PATH + "complete_dataset.pkl", 'rb') as data:
        dataset = pickle.load(data)

        for _, image, paragraph, _ in dataset:
            sentences = sent_tokenize(paragraph)
            sentences_num += len(sentences)
            for sentence in sentences:
                words = word_tokenize(sentence)
                words_num += len(words)
                sentences_counter.update([sentence])
                words_counter.update(words)

        sorted_sentences = sorted(sentences_counter.items(),
                                  key=lambda item: item[1], reverse=True)
        sorted_words = sorted(words_counter.items(),
                              key=lambda item: item[1], reverse=True)

    return sorted_sentences, sorted_words


def sentences_analysis(dataset):
    sorted_sentences, sorted_words = count_sentences_words()
    dataset_len = len(dataset)
    sentences_num = len(sorted_words)
    words_num = len(sorted_words)

    sentences_txt = "Dataset size: {}\n".format(dataset_len)
    sentences_txt += "Total number of sentences: {}\n".format(sentences_num)
    sentences_txt += "Total number of words: {}\n".format(words_num)
    sentences_txt += "Average number of sentences: {}\n".format(sentences_num//dataset_len)
    sentences_txt += "Average number of words in a sentence: {}\n\n".format(words_num // sentences_num)
    sentences_txt += "Sentence : Occurs\n"
    for sentence, counter in sorted_sentences:
        sentences_txt += "{} : {}\n".format(sentence, counter)

    sentences_txt += "\nWord : Occurs\n"
    for word, counter in sorted_words:
        sentences_txt += "{} : {}\n".format(word, counter)

    create_write_data(PATH + '/indiana_sorted_sentences.txt', sentences_txt)
    print(sentences_txt)

    return sorted_sentences


def create_sentences_dict():
    print("Creating sentences dictionary... ")
    sorted_sentences, _ = count_sentences_words()
    sentences_dict = dict()
    for sent_id, sentence in enumerate(sorted_sentences):
        sentences_dict[sentence[0]] = sent_id + 1

    print("Created the sentences dictionary. ")
    return sentences_dict


def split_dataset(dataset):
    print("Splitting up the dataset...")
    train_set, test_set = train_test_split(dataset, test_size=0.20, random_state=42)
    val_set, test_set = train_test_split(test_set, test_size=0.50, random_state=42)

    datasets_split_info = "Size of training dataset: {}\nSize of validation dataset: {}\nSize of test dataset: {}" \
        .format(len(train_set), len(val_set), len(test_set))
    print(datasets_split_info)

    pickle_data(INDIANA_SPLITS_PATH + "train_set.pkl", train_set)
    pickle_data(INDIANA_SPLITS_PATH + "val_set.pkl", val_set)
    pickle_data(INDIANA_SPLITS_PATH + "test_set.pkl", test_set)

    return train_set, val_set, test_set


complete_dataset = create_complete_dataset()
pickle_data(PATH + "/data/indiana_chest_xrays/" + "complete_dataset.pkl", complete_dataset)
with open(PATH + "/data/indiana_chest_xrays/" + "complete_dataset.pkl", 'rb') as data:
    dataset = pickle.load(data)
train_set, _, _ = split_dataset(dataset)

paragraphs_vocab = Vocabulary(train_set, INDIANA_VOCAB + "para_vocab")
pickle_data(PATH + "/data/indiana_chest_xrays/" + 'vocab.pkl', paragraphs_vocab)



