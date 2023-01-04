import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(opt):
    # return MyModel2(n_cnn_layers=opt.n_cnn_layers, n_rnn_layers=opt.n_rnn_layers, rnn_dim=opt.rnn_dim,
    #                               n_class=opt.n_class, n_feats=opt.num_features)
    return MyModel2(n_class=opt.n_class)


class MyModel2(nn.Module):
    def __init__(self, rnn_dim=2560, n_class=30, dropout=0.1):
        super(MyModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.batch_norm4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.batch_norm5 = nn.BatchNorm2d(num_features=64)

        self.lstm = nn.LSTM(rnn_dim, hidden_size=1024, num_layers=2, batch_first=True)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=25600, out_features=512)
        # self.fc1 = nn.Linear(in_features=64640, out_features=512)

        self.batch_norm6 = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(in_features=512, out_features=256)

        self.batch_norm7 = nn.BatchNorm1d(num_features=256)

        self.fc3 = nn.Linear(in_features=256, out_features=n_class)

        self.batch_norm8 = nn.BatchNorm1d(num_features=n_class)



    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.batch_norm2(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.batch_norm3(x)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.batch_norm4(x)

        x = self.conv5(x)
        x = F.gelu(x)
        x = self.batch_norm5(x)

        N, C, H, W = x.shape

        x = x.transpose(2,1)
        x = x.transpose(1,3)
        x = x.reshape(N, W, C*H)

        x, _ = self.lstm(x)

        x = self.maxpool2d(x)

        N, W, HC = x.shape
        x = x.reshape(N, W*HC)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.batch_norm6(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = self.batch_norm7(x)
        x = F.gelu(x)

        x = self.fc3(x)

        return x


class MyModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(MyModel, self).__init__()
        n_feats = (n_feats // 2) - 1
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(stride, stride))

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=32)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(608, rnn_dim)

        self.birnn_layers = nn.Sequential(*[
            LSTMLayer(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                      hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        # x = x.unsqueeze(dim=0)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm2d(n_feats)
        self.batch_norm2 = nn.BatchNorm2d(n_feats)

        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.batch_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.batch_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class LSTMLayer(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(LSTMLayer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first)
        self.batch_norm = nn.BatchNorm1d(rnn_dim)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.batch_norm(x)
        x = F.gelu(x)
        x = x.transpose(1,2)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x
