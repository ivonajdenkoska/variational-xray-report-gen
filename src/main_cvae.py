import os
import sys
from datetime import date
import logging

from torch.utils.data import DataLoader

from dataset_loader import *
from evaluation_cvae import EvaluationCVAE, get_clinically_coherence_metrics
from trainer_cvae import TrainerCVAE
from vocabulary import *

# set environment variable
os.environ['TORCH_HOME'] = MODELS_PATH
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main_cvae():
    cvae_start_info = "----- START experiment {} ----- Date: {} \nHyperparams: Epochs: {}, Batch size: {}, " \
                      "Num of z-samples: Training: {} / Inference: {}, Latent size: {}, Hidden CVAE size: {}, " \
                      "Visual / Semantic heads: {} / {}, LR: {}, Fine-Tune CNN: {}, " \
                      "KL cyclical annealing: {}, Cycle width: {}, KL weight: {}, " \
                      "Dropout rate: {}, Grad clip: {}, Num. workers: {}, Max words in sent: {}, " \
                      "Max sentences: {}, Early stop patience: {}, Word occur threshold: {}, " \
                      "Num. of encoder layers: {}, Temperature: {}, Beam size: {} Dataset path: {}\n" \
        .format(EXPERIMENT_ID, date.today().strftime("%d/%m/%Y"), NUM_EPOCHS, BATCH_SIZE, NUM_SAMPLES,
                NUM_SAMPLES_INFERENCE, LATENT_SIZE, HIDDEN_CVAE_SIZE, VISUAL_ATT_HEADS, SEMANTIC_ATT_HEADS,
                LR_CVAE, FINE_TUNE_CNN, KL_CYC_ANNEALING, CYCLE_WIDTH, KL_WEIGHT, DROPOUT, GRAD_CLIP,
                NUM_WORKERS, MAX_WORDS_IN_SENT, MAX_SENT_NUM, EARLY_STOP_PATIENCE, WORD_OCCURRENCE_THRESHOLD,
                ENCODER_LAYERS_NUM, T, BEAM_SIZE, CURR_DATA_PATH)

    print(cvae_start_info)
    log_path = INDIANA_LOG_PATH if CURR_DATA_PATH.find("indiana") != -1 else MIMIC_LOG_PATH
    logging.basicConfig(filename=log_path + str(EXPERIMENT_ID) + '.log', level=logging.INFO)
    logging.info(cvae_start_info)
    logging.info('Loading the dataset ...')

    with open(CURR_DATA_SPLITS + "train_set_1.pkl", 'rb') as train_pickle_file:
        train_set = pickle.load(train_pickle_file)

    with open(CURR_DATA_SPLITS + "val_set_1.pkl", 'rb') as val_pickle_file:
        val_set = pickle.load(val_pickle_file)

    with open(CURR_DATA_SPLITS + "test_set_1.pkl", 'rb') as test_pickle_file:
        test_set = pickle.load(test_pickle_file)

    datasets_split_info = "Path: {} \nSize of total dataset: {}\n"\
        .format(CURR_DATA_PATH, len(train_set)+len(val_set)+len(test_set))

    datasets_split_info += "Size of training dataset: {}\nSize of validation dataset: {}\nSize of test dataset: {}"\
        .format(len(train_set), len(val_set), len(test_set))
    print(datasets_split_info)
    logging.info(datasets_split_info)

    paragraphs_vocab = Vocabulary(train_set, CURR_VOCAB + "para_vocab")
    pickle_data(CURR_DATA_PATH + 'vocab.pkl', paragraphs_vocab)

    with open(CURR_DATA_PATH + "vocab.pkl", 'rb') as vocab_pickle_file:
        paragraphs_vocab = pickle.load(vocab_pickle_file)

    para_vocab_size = len(paragraphs_vocab.vocabulary)
    vocab_size_info = "Size of paragraphs vocabulary: {}\n".format(para_vocab_size)

    print(vocab_size_info)
    logging.info(vocab_size_info)

    train_dataset = ChestXRaysDataSet(train_set, paragraphs_vocab, 'train')
    val_dataset = ChestXRaysDataSet(val_set, paragraphs_vocab, 'val')
    test_dataset = ChestXRaysDataSet(test_set, paragraphs_vocab, 'test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    trainer = TrainerCVAE(train_loader, val_loader, paragraphs_vocab)
    trained_model = trainer.train()

    evaluation = EvaluationCVAE(test_loader, trained_model)
    evaluation.generate_paragraphs()

    get_clinically_coherence_metrics(test_loader)

    logging.info("----- END experiment {} ----- \n".format(EXPERIMENT_ID))
    sys.exit()


if __name__ == "__main__":
    main_cvae()


