import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FastICA

from CONSTANTS import ALPHA, BETA, GAMMA, LOG_ROOT, PROJECT_ROOT, SESSION, device
from models.gru import AttGRUModel
from module.Common import batch_variable_inst, data_iter, generate_tinsts_binary_label
from module.Optimizer import Optimizer
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.datacutter.SimpleCutting import cut_by
from preprocessing.Preprocess import Preprocessor
from representations.sequences.statistics import Sequential_TF
from representations.templates.statistics import (
    Simple_template_TF_IDF,
    Template_TF_IDF_without_clean,
)
from utils.Vocab import Vocab

lstm_hiddens = 100
num_layer = 2
batch_size = 100
epochs = 10


def get_updated_network(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
        else:
            updated_theta[k] = state_dicts[k]
    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)
    return new


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for k, v in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + "." + k))
        else:
            for k, v in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + "." + k)]

    k_param_fn(model)
    return model


class MetaLog:
    _logger = logging.getLogger("MetaLog")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "MetaLog.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        "Construct logger for MetaLog succeeded, current working directory: %s, logs will be written in %s"
        % (os.getcwd(), LOG_ROOT)
    )

    @property
    def logger(self):
        return MetaLog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_size = 128
        self.test_batch_size = 1024
        self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        self.bk_model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
            self.bk_model = self.bk_model.cuda(device)
        elif torch.mps.is_available():
            self.model = self.model.to(device)
            self.bk_model = self.bk_model.to(device)
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def bk_forward(self, inputs, targets):
        tag_logits = self.bk_model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits, dim=1)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id["Anomalous"]
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold=0.5):
        self.logger.info("Start evaluating by threshold %.3f" % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab_BGL, False)
                if torch.cuda.is_available():
                    tinst.to_cuda(device)
                elif torch.mps.is_available():
                    tinst.to_mps(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(
                    onebatch, pred_tags, tag_logits, processor_BGL.id2tag
                ):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == "Normal":
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == "Normal":
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info("TP: %d, TN: %d, FN: %d, FP: %d" % (TP, TN, FN, FP))
            if TP + FP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                fpr = 100 * FP / (FP + TN)
                self.logger.info(
                    "Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f, FPR = %.4f"
                    % (TP, (TP + FP), precision, TP, (TP + FN), recall, f, fpr)
                )
            else:
                self.logger.info("Precision is 0 and therefore f is 0")
                precision, recall, f = 0, 0, 0
        return precision, recall, f


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", default="train", type=str, help="train or test")
    argparser.add_argument(
        "--parser",
        default="IBM",
        type=str,
        help="Select parser, please see parser list for detail. Default Official.",
    )
    argparser.add_argument(
        "--min_cluster_size", type=int, default=100, help="min_cluster_size."
    )
    argparser.add_argument("--min_samples", type=int, default=100, help="min_samples")
    argparser.add_argument(
        "--reduce_dimension",
        type=int,
        default=50,
        help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.",
    )
    argparser.add_argument(
        "--threshold", type=float, default=0.5, help="Anomaly threshold."
    )
    argparser.add_argument(
        "--alpha", type=float, default=ALPHA, help="learning rate for meta training"
    )
    argparser.add_argument(
        "--beta", type=float, default=BETA, help="weight for meta testing"
    )
    argparser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="learning rate for training and testing combine",
    )

    args, extra_args = argparser.parse_known_args()

    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    # process BGL
    dataset = "BGL"
    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, "outputs")
    prob_label_res_file_BGL = os.path.join(
        save_dir,
        "results/MetaLog/"
        + dataset
        + "_"
        + parser
        + "/prob_label_res/mcs-"
        + str(min_cluster_size)
        + "_ms-"
        + str(min_samples),
    )
    rand_state_BGL = os.path.join(
        save_dir,
        "results/MetaLog/" + dataset + "_" + parser + "/prob_label_res/random_state",
    )

    output_model_dir = os.path.join(
        save_dir, "models/MetaLog/" + dataset + "_" + parser + "/model"
    )
    output_res_dir = os.path.join(
        save_dir, "results/MetaLog/" + dataset + "_" + parser + "/detect_res"
    )

    # Training, Validating and Testing instances.
    template_encoder_BGL = (
        Template_TF_IDF_without_clean() if dataset == "NC" else Simple_template_TF_IDF()
    )
    processor_BGL = Preprocessor()
    train_BGL, _, test_BGL = processor_BGL.process(
        dataset=dataset,
        parsing=parser,
        cut_func=cut_by(0.3, 0.1, 0.01),
        template_encoding=template_encoder_BGL.present,
    )

    # Log sequence representation.
    sequential_encoder_BGL = Sequential_TF(processor_BGL.embedding)
    train_reprs_BGL = sequential_encoder_BGL.present(train_BGL)
    for index, inst in enumerate(train_BGL):
        inst.repr = train_reprs_BGL[index]
    test_reprs_BGL = sequential_encoder_BGL.present(test_BGL)
    for index, inst in enumerate(test_BGL):
        inst.repr = test_reprs_BGL[index]

    # Dimension reduction if specified.
    transformer_BGL = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_BGL = FastICA(n_components=reduce_dimension)
        train_reprs_BGL = transformer_BGL.fit_transform(train_reprs_BGL)
        for idx, inst in enumerate(train_BGL):
            inst.repr = train_reprs_BGL[idx]
        print("Finished at %.2f" % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.
    train_normal_BGL = [x for x, inst in enumerate(train_BGL) if inst.label == "Normal"]
    normal_ids_BGL = train_normal_BGL[: int(0.5 * len(train_normal_BGL))]
    label_generator_BGL = Probabilistic_Labeling(
        min_samples=min_samples,
        min_clust_size=min_cluster_size,
        res_file=prob_label_res_file_BGL,
        rand_state_file=rand_state_BGL,
    )
    labeled_train_BGL = label_generator_BGL.auto_label(train_BGL, normal_ids_BGL)

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train_BGL:
        if inst.predicted == "Normal":
            if inst.label == "Normal":
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == "Anomalous":
                TP += 1
            else:
                FP += 1
    from utils.common import get_precision_recall

    print(len(normal_ids_BGL))
    print("TP %d TN %d FP %d FN %d" % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print("%.4f, %.4f, %.4f" % (p, r, f))

    # Load Embeddings
    vocab_BGL = Vocab()
    vocab_BGL.load_from_dict(processor_BGL.embedding)

    # process HDFS
    dataset = "HDFS"
    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, "outputs")
    prob_label_res_file_HDFS = os.path.join(
        save_dir,
        "results/MetaLog/"
        + dataset
        + "_"
        + parser
        + "/prob_label_res/mcs-"
        + str(min_cluster_size)
        + "_ms-"
        + str(min_samples),
    )
    rand_state_HDFS = os.path.join(
        save_dir,
        "results/MetaLog/" + dataset + "_" + parser + "/prob_label_res/random_state",
    )

    # Training, Validating and Testing instances.
    template_encoder_HDFS = (
        Template_TF_IDF_without_clean() if dataset == "NC" else Simple_template_TF_IDF()
    )
    processor_HDFS = Preprocessor()
    train_HDFS, _, _ = processor_HDFS.process(
        dataset=dataset,
        parsing=parser,
        cut_func=cut_by(0.4, 0.1),
        template_encoding=template_encoder_HDFS.present,
    )

    # Log sequence representation.
    sequential_encoder_HDFS = Sequential_TF(processor_HDFS.embedding)
    train_reprs_HDFS = sequential_encoder_HDFS.present(train_HDFS)
    for index, inst in enumerate(train_HDFS):
        inst.repr = train_reprs_HDFS[index]

    # Dimension reduction if specified.
    transformer_HDFS = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_HDFS = FastICA(n_components=reduce_dimension)
        train_reprs_HDFS = transformer_HDFS.fit_transform(train_reprs_HDFS)
        for idx, inst in enumerate(train_HDFS):
            inst.repr = train_reprs_HDFS[idx]
        print("Finished at %.2f" % (time.time() - start_time))

    labeled_train_HDFS = train_HDFS

    # aggregate vocab and label2id
    vocab = Vocab()
    new_embedding = {}
    for key in processor_BGL.embedding.keys():
        new_embedding[key] = processor_BGL.embedding[key]
    for key in processor_HDFS.embedding.keys():
        new_embedding[key + 432] = processor_HDFS.embedding[key]
    # Load Embeddings
    vocab_HDFS = Vocab()
    vocab_HDFS.load_from_dict(processor_HDFS.embedding)
    print(new_embedding.keys())
    vocab.load_from_dict(new_embedding)

    metalog = MetaLog(vocab, num_layer, lstm_hiddens, processor_BGL.label2id)

    # meta learning
    log = "layer={}_hidden={}_epoch={}".format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + "_best.pt")
    last_model_file = os.path.join(output_model_dir, log + "_last.pt")
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if mode == "train":
        # Train
        optimizer = Optimizer(
            filter(lambda p: p.requires_grad, metalog.model.parameters()), lr=gamma
        )
        global_step = 0
        bestF = 0
        for epoch in range(epochs):
            metalog.model.train()
            metalog.bk_model.train()
            start = time.strftime("%H:%M:%S")
            metalog.logger.info(
                "Starting epoch: %d | phase: train | start time: %s | learning rate: %s"
                % (epoch, start, optimizer.lr)
            )

            batch_num = int(np.ceil(len(labeled_train_HDFS) / float(batch_size)))
            batch_iter = 0
            batch_num_test = int(np.ceil(len(labeled_train_BGL) / float(batch_size)))
            batch_iter_test = 0
            total_bn = max(batch_num, batch_num_test)
            meta_train_loader = data_iter(labeled_train_HDFS, batch_size, True)
            meta_test_loader = data_iter(labeled_train_BGL, batch_size, True)

            for i in range(total_bn):
                optimizer.zero_grad()
                # meta train
                meta_train_batch = meta_train_loader.__next__()
                meta_test_batch = meta_test_loader.__next__()
                tinst_tr = generate_tinsts_binary_label(meta_train_batch, vocab_HDFS)
                if torch.cuda.is_available():
                    tinst_tr.to_cuda(device)
                elif torch.mps.is_available():
                    tinst_tr.to_mps(device)
                loss = metalog.forward(tinst_tr.inputs, tinst_tr.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward(retain_graph=True)
                batch_iter += 1
                if torch.cuda.is_available():
                    metalog.bk_model = (
                        get_updated_network(metalog.model, metalog.bk_model, alpha)
                        .train()
                        .cuda()
                    )
                elif torch.mps.is_available():
                    metalog.bk_model = (
                        get_updated_network(metalog.model, metalog.bk_model, alpha)
                        .train()
                        .to(device)
                    )
                else:
                    metalog.bk_model = get_updated_network(
                        metalog.model, metalog.bk_model, alpha
                    ).train()
                # meta test
                tinst_test = generate_tinsts_binary_label(meta_test_batch, vocab_BGL)
                if torch.cuda.is_available():
                    tinst_test.to_cuda(device)
                elif torch.mps.is_available():
                    tinst_test.to_mps(device)
                loss_te = beta * metalog.bk_forward(
                    tinst_test.inputs, tinst_test.targets
                )
                loss_value_te = loss_te.data.cpu().numpy() / beta
                loss_te.backward()
                batch_iter_test += 1
                # aggregate
                optimizer.step()
                global_step += 1
                if global_step % 500 == 0:
                    metalog.logger.info(
                        "Step:%d, Epoch:%d, meta train loss:%.2f, meta test loss:%.2f"
                        % (global_step, epoch, loss_value, loss_value_te)
                    )
                if batch_iter == batch_num:
                    meta_train_loader = data_iter(labeled_train_HDFS, batch_size, True)
                    batch_iter = 0
                if batch_iter_test == batch_num_test:
                    meta_test_loader = data_iter(labeled_train_BGL, batch_size, True)
                    batch_iter_test = 0

            if test_BGL:
                metalog.logger.info("Testing on test set.")
                _, _, f = metalog.evaluate(test_BGL)
                if f > bestF:
                    metalog.logger.info(
                        "Exceed best f: history = %.2f, current = %.2f" % (bestF, f)
                    )
                    torch.save(metalog.model.state_dict(), best_model_file)
                    bestF = f
            metalog.logger.info("Training epoch %d finished." % epoch)
            torch.save(metalog.model.state_dict(), last_model_file)

    if os.path.exists(last_model_file):
        metalog.logger.info("=== Final Model ===")
        metalog.model.load_state_dict(torch.load(last_model_file))
        metalog.evaluate(test_BGL, threshold)
    if os.path.exists(best_model_file):
        metalog.logger.info("=== Best Model ===")
        metalog.model.load_state_dict(torch.load(best_model_file))
        metalog.evaluate(test_BGL, threshold)
    metalog.logger.info("All Finished")
