import csv
import glob
import os

from PIL import Image

from config import *
from utils import *


def preprocess_mimic_images(width, height):
    with open(MIMIC_STUDIES_CSV, newline='', encoding="utf-8") as csv_file:
        print("Opening {} file ... ".format(MIMIC_STUDIES_CSV))
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for row in reader:
            print("Reading data sample {}".format(row[1]))
            _, study_id, study_path = row[0], row[1], row[2]

            if study_path.find("files/p19") != -1:
                continue

            sample_path = get_sample_path(study_path) + "/"
            image_path = MIMIC_IMAGES_PATH + sample_path
            processed_image_path = MIMIC_PROCESSED_IMAGES + sample_path

            xray_images = glob.glob(image_path + '*.jpg')
            if not xray_images:
                print("No associated images for study: {}".format(study_id))
                continue

            if not os.path.exists(processed_image_path):
                print("Creating new folder: {}".format(processed_image_path))
                os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)

            for xray_image in xray_images:
                read_image = Image.open(xray_image)
                image_name = xray_image.split("/")[-1]
                resized_image = read_image.resize((width, height))
                print("Resized and saving image: {}".format(image_name))
                resized_image.save(processed_image_path + image_name)


def preprocess_indiana_images(width, height):
    image_path = INDIANA_IMAGES_PATH
    processed_image_path = PROCESSED_IMAGES_PATH
    xray_images = glob.glob(image_path + '*.png')

    for xray_image in xray_images:
        read_image = Image.open(xray_image)
        image_name = xray_image.split("/")[-1]
        resized_image = read_image.resize((width, height))
        print("Resized and saving image: {}".format(image_name))
        resized_image.save(processed_image_path + image_name)


# preprocess_indiana_images(width=224, height=224)
preprocess_mimic_images(width=224, height=224)


