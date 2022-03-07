import logging

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from rouge import Rouge as RougeLib
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

from model import *


class EvaluationCVAE(nn.Module):
    def __init__(self, test_loader, trained_model):
        super(EvaluationCVAE, self).__init__()
        self.test_loader = test_loader
        self.model = trained_model

        with open(CURR_DATA_PATH + "vocab.pkl", 'rb') as vocab_pickle_file:
            self.vocab = pickle.load(vocab_pickle_file)

        self.labels_vocab = get_labels_vocab()
        self.num_labels_vocab = len(self.labels_vocab)
        self.num_sentences, self.num_words = MAX_SENT_NUM, MAX_WORDS_IN_SENT

        if trained_model is None:
            self.model = Model()
            self.model.load_model()

        self.device = set_device()
        self.model.to(self.device)
        if torch.cuda.is_available():
            print('Evaluation on GPU!')

        # Evaluation mode on
        self.model.eval()

    def generate_paragraphs(self):
        print("Start evaluating the model ...")
        logging.info('Start evaluating the model ...')
        true_paragraphs, temp_generated_paragraphs, top_k_generated_paragraphs = {}, {}, {}
        true_reports_csv, temp_reports_csv, top_k_reports_csv, = [], [], []
        predictions_info = ""

        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
                idx = sample[0][0]
                image = sample[1].to(self.device)
                paragraph = sample[2].reshape(self.num_sentences, self.num_words)

                start_tokens = np.zeros((image.shape[0], 1))
                start_tokens[:, 0] = self.vocab(START)
                start_tokens = torch.tensor(start_tokens, requires_grad=False) \
                    .long().to(self.device)
                true_paragraphs[idx] = self.word_ids_to_paragraph(paragraph)
                true_reports_csv.append((idx, "{}".format(" ".join(true_paragraphs[idx]))))
                predictions_info += "--- ID: {} --- \nGround-truth paragraph: {}\n".format(
                    idx, " ".join(true_paragraphs[idx]))

                top_k_generated_paragraph = self.model.inference(image, start_tokens)
                top_k_generated_paragraphs[idx] = [" ".join(self.word_ids_to_paragraph(top_k_generated_paragraph))]
                top_k_reports_csv.append((idx, "{}"
                                          .format(" ".join(self.word_ids_to_paragraph(top_k_generated_paragraph)))))
                predictions_info += "Generated paragraph w/ top-k sampling {}\n".format(top_k_generated_paragraphs[idx])

        save_text_to_csv(CURR_DATA_PATH + "output_reports/" + EXPERIMENT_ID + "_true_reports.csv", true_reports_csv)
        save_text_to_csv(CURR_DATA_PATH + "output_reports/" + EXPERIMENT_ID + "_top_k_reports.csv", top_k_reports_csv)

        predictions_info += "Top k sampling: \n"
        predictions_info += self.get_eval_scores(true_paragraphs, top_k_generated_paragraphs)

        print(predictions_info)
        logging.info(predictions_info)

    def get_eval_scores(self, true_paragraphs, generated_paragraphs):
        eval_info = ""
        bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, rouge_scores, meteor_scores = [], [], [], [], [], []
        rouge = RougeLib()
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        for scorer, method in scorers:
            eval_info += ("Computing {} score... \n".format(scorer.method()))
            score, scores = scorer.compute_score(true_paragraphs, generated_paragraphs)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval_info += ("{}: {:.8f} \n".format(m, sc))
            else:
                eval_info += ("{}: {:.8f} \n".format(method, score))

        for idx in generated_paragraphs.keys():
            generated_paragraph = generated_paragraphs[idx][0].split()
            bleu_1 = sentence_bleu(true_paragraphs[idx], generated_paragraph, weights=(1, 0, 0, 0),
                                   smoothing_function=SmoothingFunction().method2)
            bleu_1_scores.append(bleu_1)
            bleu_2 = sentence_bleu(true_paragraphs[idx], generated_paragraph, weights=(0.5, 0.5, 0, 0),
                                   smoothing_function=SmoothingFunction().method2)
            bleu_2_scores.append(bleu_2)
            bleu_3 = sentence_bleu(true_paragraphs[idx], generated_paragraph, weights=(0.33, 0.33, 0.33, 0),
                                   smoothing_function=SmoothingFunction().method2)
            bleu_3_scores.append(bleu_3)
            bleu_4 = sentence_bleu(true_paragraphs[idx], generated_paragraph,  weights=[0.25, 0.25, 0.25, 0.25],
                                   smoothing_function=SmoothingFunction().method2)
            bleu_4_scores.append(bleu_4)

            hypothesis = generated_paragraphs[idx][0]
            reference = " ".join(true_paragraphs[idx])

            if hypothesis:
                rouge_score = rouge.get_scores(hypothesis, reference, ignore_empty=True)
                rouge_score = rouge_score[0]['rouge-l']['f']
                rouge_scores.append(rouge_score)

                meteor = meteor_score(hypothesis, reference, alpha=0.9, beta=4, gamma=0.2)
                meteor_scores.append(meteor)

        eval_info += "BLEU-1: {:.8f}, BLEU-2:{:.8f}, BLEU-3:{:.8f}, BLEU-4: {:.8f}\n" \
                     "ROUGE-L: {:.8f}; METEOR: {:.8f}\n"\
            .format(np.mean(bleu_1_scores), np.mean(bleu_2_scores), np.mean(bleu_3_scores), np.mean(bleu_4_scores),
                    np.mean(rouge_scores), np.mean(meteor_scores))

        return eval_info

    def word_ids_to_paragraph(self, generated_word_ids):
        generated_paragraph = []
        for word_ids in generated_word_ids:
            generated_paragraph += self.vocab.vec2sent(word_ids)

        return generated_paragraph


def get_clinically_coherence_metrics(test_loader):
    f1_scores_macro, precision_scores_macro, recall_scores_macro = [], [], []
    f1_scores_micro, precision_scores_micro, recall_scores_micro = [], [], []
    accuracy_scores = []
    labeled_reports_dict = get_labeled_reports_dict()

    for i, sample in enumerate(test_loader):
        idx = sample[0][0]
        labels = sample[4].reshape(-1)
        if idx in labeled_reports_dict:
            labels_pred_ids = labeled_reports_dict[idx].reshape(-1)
            labels_ids = []
            for idx, label in enumerate(labels):
                if label == 1:
                    labels_ids.append(idx)

            print("True: {}, Pred: {}, Labels ids: {}".format(labels, labels_pred_ids, labels_ids))
            accuracy_scores.append(balanced_accuracy_score(y_true=labels, y_pred=labels_pred_ids))

            f1_scores_macro.append(f1_score(y_true=labels, y_pred=labels_pred_ids, average="macro", labels=[1], zero_division=1))
            precision_scores_macro.append(precision_score(y_true=labels, y_pred=labels_pred_ids, average="macro", labels=[1], zero_division=1))
            recall_scores_macro.append(recall_score(y_true=labels, y_pred=labels_pred_ids, average="macro", labels=[1], zero_division=1))

            f1_scores_micro.append(f1_score(y_true=labels, y_pred=labels_pred_ids, average="micro", labels=[1], zero_division=1))
            precision_scores_micro.append(precision_score(y_true=labels, y_pred=labels_pred_ids, average="micro", labels=[1], zero_division=1))
            recall_scores_micro.append(recall_score(y_true=labels, y_pred=labels_pred_ids, average="micro", labels=[1], zero_division=1))

    predictions_info = "Accuracy: {}\n".format(np.mean(accuracy_scores))
    predictions_info += "Macro: F1 score: {:.8f}, Precision: {:.8f}, Recall: {:.8f}\n".format(
        np.mean(f1_scores_macro), np.mean(precision_scores_macro), np.mean(recall_scores_macro))
    predictions_info += "Micro: F1 score: {:.8f}, Precision: {:.8f}, Recall: {:.8f}\n".format(
        np.mean(f1_scores_micro), np.mean(precision_scores_micro), np.mean(recall_scores_micro))

    print(predictions_info)
    logging.info(predictions_info)


def get_labeled_reports_dict():
    labeled_reports_dict = {}
    with open(MIMIC_PATH + LABELED_REPORTS, newline='\n', encoding="utf-8") as csv_file:
        print("Opening {} file ... ".format(MIMIC_PATH + LABELED_REPORTS))
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for sample in reader:
            idx = sample[0]
            report_labeled = next(reader)
            labels = report_labeled[1:]
            labeled_reports_dict[idx] = np.asarray(process_chexpert_labels(labels)).reshape(1, len(txts))

    return labeled_reports_dict




