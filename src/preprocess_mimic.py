import glob
import logging

from collections import Counter

from utils import *

from preprocess_indiana import preprocess_tokens
from vocabulary import Vocabulary


def parse_mimic_cxr_dataset():
    logging.basicConfig(filename='../log/mimic_logger_file.log', level=logging.INFO)
    print("Start preprorcessing MIMIC-CXR dataset ...")
    image_views_dict = get_image_views_dict()
    labels_dict = get_chexpert_labels_dict()
    tuples = []

    with open(MIMIC_STUDIES_CSV, newline='', encoding="utf-8") as csv_file:
        print("Opening {} file ... ".format(MIMIC_STUDIES_CSV))
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for row in reader:
            print("Reading data sample {}".format(row[1]))
            logging.info("Reading data sample {}".format(row[1]))
            subject_id, study_id, path = row[0], row[1], row[2]
            sample_path = get_sample_path(path)
            report_path = MIMIC_REPORTS_PATH + sample_path + ".txt"
            image_path = MIMIC_PROCESSED_IMAGES + sample_path + "/"

            # ### Extracting the X-ray images ###
            xray_images = glob.glob(image_path + '*.jpg')
            if len(xray_images) < 2:
                continue
            else:
                xray_images = xray_images[:2]

            try:
                Image.open(xray_images[0])
            except FileNotFoundError:
                print("FileNotFoundError for {}".format(xray_images[0]))
                continue

            try:
                Image.open(xray_images[1])
            except FileNotFoundError:
                print("FileNotFoundError for {}".format(xray_images[1]))
                continue

            if not xray_images:
                print("No associated images for study: {}".format(study_id))
                logging.info("No associated images for study: {}".format(study_id))
                continue

            frontal_x_ray = None
            # Check for frontal image
            for x_ray_image in xray_images:
                x_ray_image_id = x_ray_image.split("/")[-1].split(".")[0]
                if image_views_dict[x_ray_image_id]:
                    x_ray_image_view = image_views_dict[x_ray_image_id]
                    if x_ray_image_view == "PA" or x_ray_image_view == "AP":  # postero-anterior
                        frontal_x_ray = x_ray_image
                        break

            if frontal_x_ray is None:
                print("No frontal image for the study: {}\\{}".format(subject_id, study_id))
                logging.info("No frontal image for the study: {}\\{}".format(subject_id, study_id))
                continue

            # ### Extracting and parsing the report ###
            try:
                report = open(report_path, "r").read()
            except FileNotFoundError:
                print("Report not found")
                logging.info("Report not found")
                continue

            parsed_report = parse_report(report)
            parsed_report = ' '.join(parsed_report)

            if len(parsed_report) < 2:  # if there are no findings or impression, we don't consider the sample
                print("No findings or impression for study: {}".format(study_id))
                logging.info("No findings or impression for study: {}".format(study_id))
                continue

            # ### Extracting the labels ###
            if study_id in labels_dict.keys():
                labels_row = labels_dict[study_id]
            else:
                continue

            labels = process_chexpert_labels(labels_row)
            if not labels:
                print("No associated labels for study: {}".format(study_id))
                logging.info("No associated labels for study: {}".format(study_id))
                continue

            tuples.append((study_id, xray_images, parsed_report, labels))
            print("Done processing study {} !!! ".format(study_id))

    return tuples


def get_image_views_dict():
    image_views_dict = {}
    with open(MIMIC_METADATA_PATH, newline='', encoding="utf-8") as metadata:
        metadata_reader = csv.reader(metadata, delimiter='\n')
        next(metadata_reader)

        for row in metadata_reader:
            row_split = row[0].split(",")
            image_id = row_split[0]
            image_view = row_split[4]  # PA, AP or LATERAL
            image_views_dict[image_id] = image_view

    return image_views_dict


def get_chexpert_labels_dict():
    labels_dict = {}
    with open(MIMIC_LABELS_PATH, newline='', encoding="utf-8") as labels:
        labels_reader = csv.reader(labels, delimiter='\n')
        next(labels_reader)

        for row in labels_reader:
            row_split = row[0].split(",")
            study_id = row_split[1]
            labels_dict[study_id] = row_split[2:]

    return labels_dict


def get_chexpert_labels(labels_row):
    """
    1.0 - The label was positively mentioned in the associated study, and is present in one or more of the images
    0.0 - The label was negatively mentioned in the associated study, and should not be present in any of the images
    -1.0 - The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present,
           or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
    :param labels_row:
    :return:
    """
    labels = []
    label_ids_ones = [i for i, x in enumerate(labels_row) if x == "1.0"]  # id of the label from the labels_list
    label_ids_zeros = [i for i, x in enumerate(labels_row) if x == "0.0"]

    if label_ids_ones:
        labels = [LABELS_LIST[label_id] for label_id in label_ids_ones]
    elif label_ids_zeros:
        labels = [LABELS_LIST[label_id] for label_id in label_ids_zeros]
        if "no finding" in labels:
            labels = []
        else:
            labels = ["no finding"]

    return labels


def parse_report(report):
    findings_start = report.find("FINDINGS:")
    impression_start = report.find("IMPRESSION:")
    report_start = report.find("REPORT:")
    additional_info_start = report.find("PA AND LATERAL CHEST RADIOGRAPHS:")
    frontal_info_start = report.find("AP FRONTAL CHEST RADIOGRAPH:")
    impression = ""

    if impression_start != -1:
        impression = get_section(report, "IMPRESSION:")
    if findings_start != -1:
        candidate_report = impression + get_section(report, "FINDINGS:")
    elif report_start != -1:
        candidate_report = impression + get_section(report, "REPORT:")
    elif additional_info_start != -1:
        candidate_report = impression + get_section(report, "PA AND LATERAL CHEST RADIOGRAPHS:")
    elif frontal_info_start != -1:
        candidate_report = impression + get_section(report, "AP FRONTAL CHEST RADIOGRAPH:")
    else:
        candidate_report = impression + report.split("\n \n")[-1]

    parsed_report = preprocess_tokens(candidate_report)

    return parsed_report


def get_section(report, section_name):
    # take everything that comes after the section title (which means other sections that come after this one)
    section = report.split(section_name)[1]
    section_letters = 0

    impression_start = section.find("IMPRESSION:")
    if impression_start != -1:  # check if there is an IMPRESSION section after
        section = section[:impression_start]
    else:  # find the first occurrence of a section title
        for i, ch in enumerate(section):
            if ch.isupper():
                section_letters += 1
            else:
                section_letters = 0

            if section_letters > 6:  # If upper letters occur in a row few time, it means it's a section title
                section = section[:(i - section_letters)]

    return section.strip()


def split_mimic_dataset(dataset):
    print("Splitting up the MIMIC CXR dataset...")
    train_set, val_set, test_set = [], [], []
    image_id_split_set = {}

    with open(MIMIC_SPLIT_SET, newline='', encoding="utf-8") as csv_file:
        print("Opening {} file ... ".format(MIMIC_SPLIT_SET))
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for row in reader:
            image_id, _, _, split_set = row
            image_id_split_set[image_id] = split_set

    for data_sample in dataset:
        study_id, image_path, report, labels = data_sample
        image_id = image_path[0].split("/")[-1].split(".")[0]
        split_set = image_id_split_set[image_id]

        if split_set == "train":
            train_set.append(data_sample)
        elif split_set == "validate":
            val_set.append(data_sample)
        elif split_set == "test":
            test_set.append(data_sample)

    datasets_split_info = "Size of training dataset: {}\nSize of validation dataset: {}\nSize of test dataset: {}" \
        .format(len(train_set), len(val_set), len(test_set))
    print(datasets_split_info)

    return train_set, val_set, test_set


complete_dataset = parse_mimic_cxr_dataset()
print("Size of the dataset: {}".format(len(complete_dataset)))
pickle_data(MIMIC_PATH + "complete_dataset.pkl", complete_dataset)
pickle_data(MIMIC_PATH + "labels.pkl", LABELS_LIST)

train_set, val_set, test_set = split_mimic_dataset(complete_dataset)
paragraphs_vocab = Vocabulary(train_set, MIMIC_VOCAB + "para_vocab")
pickle_data(MIMIC_PATH + 'vocab.pkl', paragraphs_vocab)

pickle_data(MIMIC_SPLITS_PATH + "train_set_1.pkl", train_set)
pickle_data(MIMIC_SPLITS_PATH + "val_set_1.pkl", val_set)
pickle_data(MIMIC_SPLITS_PATH + "test_set_1.pkl", test_set)


