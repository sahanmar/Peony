import torch
import torch.nn as nn

from typing import Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NeuralNetLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0,
        num_of_lstm_layers: int = 1,
    ):
        super(NeuralNetLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_of_lstm_layers,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        # self.eval()

    def forward(self, inputs):

        embeddings, seq_lengths, _ = inputs

        total_length = embeddings.size(1)
        packed_inputs = pack_padded_sequence(embeddings, seq_lengths, batch_first=True)
        unpacked_lstm_out, _ = pad_packed_sequence(
            self.lstm(packed_inputs)[0], batch_first=True, total_length=total_length
        )
        lstm_mean = torch.mean(unpacked_lstm_out, 1)
        lstm_out = self.dropout(self.relu(lstm_mean))

        return self.dropout(self.softmax(self.linear1(lstm_out)))

    def predict(self, inputs):
        _, _, sorted_indices = inputs
        return self.forward(inputs)[sorted_indices]


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout=0):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(inplace=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        return self.forward(x)
