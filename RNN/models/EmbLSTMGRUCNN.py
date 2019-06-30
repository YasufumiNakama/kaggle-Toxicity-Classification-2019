import torch
from torch import nn


hidden_size = 80
drop_rate = 0.4


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class EmbLSTMGRUCNN(nn.Module):
    def __init__(self, embedding_matrix, max_features, num_aux_targets=6):
        super().__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(drop_rate)

        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.cnn = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=0)

        self.linear1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(inplace=True)
        )

        self.linear_out = nn.Linear(hidden_size * 2, 1)
        self.linear_aux_out = nn.Linear(hidden_size * 2, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        x = self.cnn(h_lstm2.permute(0, 2, 1))

        # global average pooling
        avg_pool = torch.mean(x.permute(0, 2, 1), 1)
        # global max pooling
        max_pool, _ = torch.max(x.permute(0, 2, 1), 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = self.linear1(h_conc)
        h_conc_linear2 = self.linear2(h_conc)

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)

        out = torch.cat([result, aux_result], 1)

        return out
