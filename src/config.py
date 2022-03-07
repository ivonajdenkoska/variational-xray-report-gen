from pathlib import Path

PATH = str(Path.cwd().parent)
DATA_PATH = PATH + "/../med-reports-generator/data/"

MODELS_PATH = PATH + "/models/"
FIGURES_PATH = PATH + "/figs/"
LOG_PATH = PATH + "/log/"
INDIANA_LOG_PATH = LOG_PATH + "indiana/"
MIMIC_LOG_PATH = LOG_PATH + "mimic/"
BIO_WORD2VEC_BIN = MODELS_PATH + "BioWordVec_PubMed_MIMICIII_d200.bin"

INDIANA_PATH = DATA_PATH + "indiana_chest_xrays/"
INDIANA_REPORTS_PATH = INDIANA_PATH + "radiology_reports/"
INDIANA_IMAGES_PATH = INDIANA_PATH + "x_ray_images/"
PROCESSED_IMAGES_PATH = INDIANA_PATH + "processed_images/"
INDIANA_SPLITS_PATH = PATH + "/data/indiana_chest_xrays/" + "data_splits/"
INDIANA_VOCAB = PATH + "/data/indiana_chest_xrays/" + "vocabularies/"

MIMIC_PATH = DATA_PATH + "mimic_cxr/"
MIMIC_IMAGES_PATH = MIMIC_PATH + "images/"
MIMIC_REPORTS_PATH = MIMIC_PATH + "reports/"
MIMIC_RECORDS_CSV = MIMIC_PATH + "cxr-record-list.csv"
MIMIC_STUDIES_CSV = MIMIC_PATH + "cxr-study-list.csv"
MIMIC_LABELS_PATH = MIMIC_PATH + "mimic-cxr-2.0.0-chexpert.csv"
MIMIC_METADATA_PATH = MIMIC_PATH + "mimic-cxr-2.0.0-metadata.csv"
MIMIC_PROCESSED_IMAGES = MIMIC_PATH + "processed_images/"
MIMIC_SPLIT_SET = MIMIC_PATH + "mimic-cxr-2.0.0-split.csv"
MIMIC_SPLITS_PATH = MIMIC_PATH + "data_splits/"
MIMIC_VOCAB = MIMIC_PATH + "vocabularies/"
BOUNDING_BOXES_PATH = MIMIC_PATH + "images_bounding_boxes/"
BOUNDING_BOXES_CSV = MIMIC_PATH + "bounding_boxes.csv"

MS_COCO_PATH = PATH + "/data/mscoco/"
MS_COCO_DATASET_PATH = "/Users/home/Documents/PhD/Datasets/MSCOCO/"
MS_COCO_CAPTIONS = MS_COCO_DATASET_PATH + "annotations/"
MS_COCO_IMAGES_PATH = MS_COCO_PATH + "images/"

CURR_DATA_PATH = MIMIC_PATH  # Indicates the currently used dataset
CURR_DATA_SPLITS = CURR_DATA_PATH + "data_splits/"
CURR_VOCAB = CURR_DATA_PATH + "vocabularies/"

EXPERIMENT_ID = "indiana_1"
MODEL_NAME = str(EXPERIMENT_ID) + '_trained_model.tar'

BATCH_SIZE = 128
NUM_EPOCHS = 50
GRAD_CLIP = 1
NUM_WORKERS = 8
EARLY_STOP_PATIENCE = 3
DELTA = 100

MAX_WORDS_IN_SENT = 10
MAX_SENT_NUM = 4
MAX_WORDS_INFERENCE = 8
WORD_OCCURRENCE_THRESHOLD = 100

LR_CVAE = 1e-04
FINE_TUNE_CNN = False
NUM_SAMPLES = 10
NUM_SAMPLES_INFERENCE = 10

# Dimensions
LATENT_SIZE = 256
HIDDEN_CVAE_SIZE = 256
VISUAL_FEAT_SIZE = 256
WORD_EMB_SIZE = 200

LSTM_LAYERS_NUM = 1
ENCODER_LAYERS_NUM = 1
VISUAL_ATT_HEADS = MAX_SENT_NUM
SEMANTIC_ATT_HEADS = 1

DROPOUT = 0.5
KL_WEIGHT = 0.00001
KL_CYC_ANNEALING = True
CYCLE_WIDTH = 3

# ### Evaluation ###
T = 0.90  # temperature sampling (lower - less diverse words; greater - more diverse)
K = 30  # top K sampling
P = 0.95  # p for nucleus sampling
BEAM_SIZE = 5

STOP = "."
SPACE = " "
UNK = "[UNK]"
START = "[START]"
END = "[END]"
PAD = "[PAD]"

LABELS_LIST = ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum", "fracture",
               "lung lesion", "lung opacity", "no finding", "pleural effusion", "pleural other", "pneumonia",
               "pneumothorax", "support devices"]


