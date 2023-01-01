import config
import numpy as np
import torch
import torch.nn as nn
import DeepSpeech2Model
from dataloader import SpectogramDataset

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def main(opt):
    data = SpectogramDataset(opt)

    model = DeepSpeech2Model.get_model()
    model = nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


if __name__ == '__main__':
    parser = config.get_arguments()
    opt = parser.parse_args()

    main(opt)