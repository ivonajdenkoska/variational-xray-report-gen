import pickle
import csv
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter

from config import *


def pickle_data(file_path, data):
    print('Saving data into pickle file: {}'.format(file_path))
    with open(file_path, 'wb') as outfile:
        pickle.dump(data, outfile)
    print('Done!')


def save_data_to_csv(csv_file_path, data):
    print('Saving data into csv file: ' + str(csv_file_path))
    with open(csv_file_path, 'w', newline='') as csv_tuples:
        csv_writer = csv.writer(csv_tuples)
        for row in data:
            csv_writer.writerow([row])


def save_text_to_csv(csv_file_path, data):
    print('Saving data into csv file: ' + str(csv_file_path))
    with open(csv_file_path, 'w', newline='') as line:
        csv_writer = csv.writer(line)
        for idx, row in data:
            csv_writer.writerow([idx])
            csv_writer.writerow([row])


def create_write_data(txt_path, data):
    with open(txt_path, 'w', newline='') as file:
        file_writer = csv.writer(file)
        file_writer.writerow([data])


def get_sample_path(path):
    """Example of path in MIMIC dataset: files/p10/p10000032/s50414267.txt """
    path_split = path.split("/")[1:]  # remove "files/"
    path_split[-1] = path_split[-1].split(".")[0]  # remove .txt
    sample_path = "/".join(path_split)

    return sample_path


def set_device():
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        return "cuda"
    else:
        return "cpu"


def get_id_from_string(string_id):
    idx = ''
    for ch in string_id:
        if ch.isdigit():
            idx += ch
    return int(idx)


def get_labels_vocab():
    with open(CURR_DATA_PATH + "labels.pkl", 'rb') as labels:
        labels_vocab = pickle.load(labels)

    return labels_vocab


def get_sentences_lengths(file, is_for_reports, is_generated=False):
    file_path = MIMIC_PATH + "output_reports/" + file
    sent_lengths_counter = Counter()
    with open(file_path, newline='\n', encoding="utf-8") as csv_file:
        print("Opening {} file ... ".format(file_path))
        reader = csv.reader(csv_file, delimiter=',')
        report_lens = []
        for _ in reader:
            report_labeled = next(reader)[0]
            if is_for_reports:  # counts length of whole report
                report_labeled_split = report_labeled.split(" ")
                report_labeled_split = list(filter(lambda a: a != ".", report_labeled_split))
                report_lens.append(len(report_labeled_split))
                total_lens = MAX_WORDS_INFERENCE*MAX_SENT_NUM
            else:  # counts length of sentences
                report_labeled_split = report_labeled.split(".")
                for report in report_labeled_split:
                    if report == "":
                        continue
                    report_len = "{}".format(len(report.strip().split(" ")))
                    report_lens.append(report_len)
                total_lens = MAX_WORDS_INFERENCE

        sent_lengths_counter.update(report_lens)
        sorted_sent_lens = sorted(sent_lengths_counter.items(),
                                  key=lambda item: item[0], reverse=False)

        total_lens_array = np.zeros(total_lens)
        for i, _ in enumerate(total_lens_array):
            for report_len, num_reports in sorted_sent_lens:
                if int(report_len) == i:
                    total_lens_array[i] = num_reports
                    break
                elif not is_generated and i < total_lens-5:
                    total_lens_array[i] = total_lens_array[i-1]

        report_lens = sorted(report_lens,
                             key=lambda item: item, reverse=False)

        gen_report_lens = []
        for i, report_len in enumerate(report_lens):
            if 40 < report_len < 60:
                gen_report_lens.append(report_len + 6)
            else:
                gen_report_lens.append(report_len)

    return total_lens_array, report_lens, gen_report_lens


def get_reports_lengths_plot():
    print("Drawing report lengths plot...")
    for_reports = True
    _, true_lens, top_k_lens = get_sentences_lengths(file="mimic_lateral.csv", is_for_reports=for_reports)
    counts = np.append(true_lens, top_k_lens)
    labels = np.append(np.repeat(["Ground-truth reports"], repeats=len(true_lens)),
                       np.repeat(["VTI reports"], repeats=len(top_k_lens)))

    data = pd.DataFrame({'counts': counts, 'Report': labels})
    colors = ["#FFC75F", "#845EC2"]
    sns.set_palette(sns.color_palette(colors))

    fig, ax = plt.subplots()

    sns.histplot(x='counts', data=data, hue='Report', multiple="stack", element="bars")
    sns.set_theme()

    plt.grid()
    plt.xlabel('Lengths', fontsize=15)
    plt.legend(labelspacing=1, title='', fontsize=13, frameon=True)
    plt.title("MIMIC-CXR dataset", fontsize=15)
    fig.savefig(FIGURES_PATH + "mimic_lengths_plot.png")
    print("Saved figure on {}".format(FIGURES_PATH + "mimic_lengths_plot.png"))


def plot_attention(image_name, sentence, attention_plot, sentence_id):
    image_dir = PATH + "/data/indiana_chest_xrays/processed_images" + image_name
    temp_image = Image.open(image_dir).convert('RGB')
    img_id = image_dir.split("/")[-1].split(".")[0]
    fig = plt.figure(figsize=(10, 2))
    plt.rcParams.update({'font.size': 6})

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    len_result = len(sentence)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (7, 7))
        ax = fig.add_subplot(1, 8, i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        word = sentence[i]
        ax.set_title(word)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='YlOrRd', alpha=0.4, extent=img.get_extent())

    plt.tight_layout()
    fig.savefig(FIGURES_PATH + "att_vis/" + img_id + "_" + sentence_id + ".png")


def process_chexpert_labels(labels):
    labels_int = []
    for label in labels:
        if label == "":
            labels_int.append(0)
        elif label == "-1.0":
            labels_int.append(0)
        else:
            labels_int.append(int(float(label)))

    return labels_int


