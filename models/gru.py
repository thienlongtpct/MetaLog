import logging
import os
import sys

import torch
import torch.mps
import torch.nn as nn
from torch.nn.parameter import Parameter

from CONSTANTS import LOG_ROOT, SESSION, device
from module.Attention import LinearAttention
from module.Common import NonLinear, drop_input_independent
from module.CPUEmbedding import CPUEmbedding


class AttGRUModel(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger("AttGRU")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "AttGRU.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        f"Construct logger for Attention-Based GRU succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
    )

    @property
    def logger(self):
        return AttGRUModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0.0, is_backup=False):
        super(AttGRUModel, self).__init__()
        self.dropout = dropout
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(
            vocab_size, word_dims, padding_idx=vocab_size - 1
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False

        if not is_backup:
            self.logger.info("==== Model Parameters ====")
            self.logger.info(f"Input Dimension: {word_dims}")
            self.logger.info(f"Hidden Size: {lstm_hiddens}")
            self.logger.info(f"Num Layers: {lstm_layers}")
            self.logger.info(f"Dropout: {round(dropout, 3)}")
        self.rnn = nn.GRU(
            input_size=word_dims,
            hidden_size=lstm_hiddens,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(
            tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim
        )
        self.proj = NonLinear(self.sent_dim, 2)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(
            vocab.vocab_size, word_dims, padding_idx=vocab.PAD
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        if torch.cuda.is_available():
            embed = embed.cuda(device)
        elif hasattr(torch.mps, "is_available") and torch.mps.is_available():
            embed = embed.to(device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        # represents = hiddens
        represents = represents.sum(dim=1)
        # represents = represents[:, -1, :]
        outputs = self.proj(represents)
        return outputs  # , represents
