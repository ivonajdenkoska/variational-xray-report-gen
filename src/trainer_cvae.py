import time
import logging

import torch.optim as optim

from model import *


class TrainerCVAE:
    def __init__(self, train_loader, val_loader, paragraphs_vocab):
        super(TrainerCVAE, self).__init__()
        if torch.cuda.is_available():
            print('Training on GPU!')

        self.vocab = paragraphs_vocab
        self.latent_size = LATENT_SIZE
        self.batch_size = BATCH_SIZE
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_size = len(self.train_loader.dataset)
        self.device = set_device()
        self.vocab_size = len(self.vocab)

        self.model = Model()
        self.model.to(self.device)
        params_all = list(filter(lambda p: p.requires_grad, list(self.model.parameters())))
        params_summed = sum(p.numel() for p in params_all)

        params_cnn = list(filter(lambda p: p.requires_grad, self.model.cnn_att_encoder.parameters()))
        params_cnn_summed = sum(p.numel() for p in params_cnn)

        self.optimizer = optim.Adam(params=params_all, lr=LR_CVAE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True, cooldown=5)
        self.early_stopping = EarlyStopping()

        print("Total num of params: {} / CNN params: {}".format(params_summed, params_cnn_summed))
        logging.info("Total num of params: {} / CNN params: {}".format(params_summed, params_cnn_summed))

        # For KL annealing
        self.num_batches = self.dataset_size // self.batch_size
        print("Num. of batches in one epoch: {}".format(self.num_batches))
        self.kl_cyc_annealing = KL_CYC_ANNEALING
        self.kl_weights = frange_cycle_linear(n_iter=self.num_batches * CYCLE_WIDTH)
        self.kl_weights_len = len(self.kl_weights)-1

        self.global_step = 0
        self.mode = "train"
        self.kl_terms = []

        self.mean_train_losses = []
        self.mean_val_losses = []

        self.all_topics = []
        self.all_images = []
        self.all_sentences = []

    def train(self):
        """
        Performs training of the model through a given num. of epochs
        """
        logging.info('Start training the model ...')
        start_train = time.time()
        kl_weight_id = 0

        for epoch_id in range(0, NUM_EPOCHS):
            print('------ START Epoch [{}/{}] ------'.format(epoch_id+1, NUM_EPOCHS))
            start = time.time()
            train_losses, val_losses = [], []
            train_rec_terms, train_kl_terms, train_kl_weight = [], [], []
            val_rec_terms, val_kl_terms = [], []

            # Training mode on
            self.model.train()
            self.mode = "train"

            for batch_id, batch in enumerate(self.train_loader):
                if self.kl_cyc_annealing:
                    kl_weight = self.kl_weights[kl_weight_id]
                    if kl_weight_id < self.kl_weights_len:
                        kl_weight_id += 1
                    else:
                        kl_weight_id = 0
                else:
                    kl_weight = KL_WEIGHT
                with torch.autograd.detect_anomaly():
                    batch_loss, reconstruction_term, kl_term, kl_weight = self.train_batch(batch, kl_weight)
                train_losses.append(batch_loss.item())
                train_rec_terms.append(reconstruction_term)
                train_kl_terms.append(kl_term)
                train_kl_weight.append(kl_weight)

            # Validation mode on
            with torch.no_grad():
                self.model.eval()
                self.mode = "val"
                for batch in self.val_loader:
                    batch_val_loss, reconstruction_term, kl_term, _ = self.forward_batch(batch, 1)
                    val_losses.append(batch_val_loss.item())
                    val_rec_terms.append(reconstruction_term)
                    val_kl_terms.append(kl_term)

            lr = self.optimizer.param_groups[0]['lr']
            self.mean_train_losses.append(np.mean(train_losses))
            self.mean_val_losses.append(np.mean(val_losses))

            end = time.time()

            train_log = "Training: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, Rec = {:.4f}, KL = {:.4f}, " \
                        "KL weight = {:.6f}, LR CVAE = {}"\
                .format(epoch_id+1, NUM_EPOCHS, np.mean(train_losses), np.mean(train_rec_terms),
                        np.mean(train_kl_terms), np.mean(train_kl_weight), lr)
            val_log = "Validation: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, Rec = {:.4f}, KL = {:.4f}, " \
                      "LR CVAE = {}"\
                .format(epoch_id+1, NUM_EPOCHS, np.mean(val_losses), np.mean(val_rec_terms),
                        np.mean(val_kl_terms), lr)

            print(train_log)
            print(val_log)

            logging.info(train_log)
            logging.info(val_log)

            print('Total time train + val of epoch {}: {:.4f} seconds'.format(epoch_id+1, (end - start)))
            print('------ END Epoch [{}/{}] ------'.format(epoch_id+1, NUM_EPOCHS))

            logging.info('Total time train + val of epoch {}: {:.4f} seconds'.format(epoch_id+1, (end - start)))

            self.early_stopping(val_loss=np.mean(val_losses), model=self.model)
            if self.early_stopping.check_early_stop:
                print("Early stopping ...")
                logging.info("Early stopping ...")
                break

        print("End of training...")

        end_train = time.time()
        train_time_info = 'The training took {:.4f} minutes'.format((end_train - start_train)/60)
        print(train_time_info)
        logging.info(train_time_info)
        save_data_to_csv(CURR_DATA_PATH + EXPERIMENT_ID + '_kl_terms.csv', self.kl_terms)

        return self.model

    def train_batch(self, batch, kl_weight):
        batch_loss, reconstruction_term, kl_term, kl_weight = self.forward_batch(batch, kl_weight)
        self.kl_terms.append(kl_term)

        # The gradients are accumulated, so zero out them
        self.optimizer.zero_grad()

        # Backward pass
        batch_loss.backward()

        # Gradient clipping if needed
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP)

        # Update weights
        self.optimizer.step()

        return batch_loss, reconstruction_term, kl_term, kl_weight

    def forward_batch(self, batch, kl_weight):
        """
        Forwards a batch of examples through all model layers
        :param batch:
        :param kl_weight
        :return: batch_loss
        """
        image = batch[1].to(self.device)
        paragraph = torch.tensor(batch[2]).long().to(self.device)
        paragraph_lengths = torch.tensor(batch[3]).to(self.device)

        reconstruction_term, kl_term = self.model(image, paragraph, paragraph_lengths)

        # Forward pass
        batch_loss = reconstruction_term + kl_weight * kl_term

        return batch_loss, reconstruction_term.item(), float(kl_term), kl_weight

    def kl_anneal(self):
        if self.global_step < self.num_batches:
            kl_weight = 0.000001
        else:
            # KL cyclical annealing
            max_anneal = 1  # max value for beta
            cycle_ep = 20  # determines how small the steps will be
            cycle_t = self.num_batches * cycle_ep
            full_kl_step = cycle_t // 2
            global_t_cyc = self.global_step % cycle_t
            kl_weight = max_anneal * np.minimum((global_t_cyc + 1) / full_kl_step, 1)

        return kl_weight


def frange_cycle_linear(n_iter, start=0, stop=1, n_cycle=1, ratio=0.80):
    """
    :return: 50% of the period is 0, then goes from 0 to 1
    """
    L = np.ones(n_iter)
    half_cycle = n_iter // (2 * n_cycle)

    L[:half_cycle] = 0
    period = half_cycle
    # period = n_iter / n_cycle
    step = (stop-start)/(half_cycle*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[half_cycle+int(i+c*period)] = v
            v += step
            i += 1
    return L


class EarlyStopping:
    """ Adapted from: 
    Title: Early Stopping for PyTorch
    Availability: https://github.com/Bjarten/early-stopping-pytorch """
    """ Early stops the training if validation loss doesn't improve after a given patience """
    def __init__(self):
        # How long to wait after last time validation loss improved
        self.patience = EARLY_STOP_PATIENCE
        # Minimum change in the monitored quantity to qualify as an improvement
        self.delta = DELTA
        self.counter = 0
        self.best_score = None
        self.check_early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Early stopping counter: {}/{}".format(self.counter, self.patience))
            logging.info("Early stopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.check_early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease """
        print("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model ...".format(self.val_loss_min, val_loss))
        logging.info("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model ...".format(self.val_loss_min, val_loss))
        model.save_model()
        self.val_loss_min = val_loss







