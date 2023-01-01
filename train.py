import config
import numpy as np
import torch
import torch.nn as nn
import DeepSpeech2Model
from dataloader import SpectogramDataset

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def plot_spectogram(data):
    spec = data.__get_item__(2000)

    import math
    # import numpy as np
    import matplotlib.pyplot as plt

    # Set the time difference to take picture of
    # the the generated signal.
    Time_difference = 0.0001

    # Generating an array of values
    Time_Array = np.linspace(0, 5, math.ceil(5 / Time_difference))

    # Actual data array which needs to be plot
    Data = spec['data']

    # Matplotlib.pyplot.specgram() function to
    # generate spectrogram
    plt.specgram(Data)

    # Set the title of the plot, xlabel and ylabel
    # and display using show() function
    plt.title('Spectrogram Using matplotlib.pyplot.specgram() Method')
    plt.xlabel("DATA")
    plt.ylabel("TIME")
    plt.show()


def main(opt):
    data = SpectogramDataset(opt)
    # plot_spectogram(data)

    model = DeepSpeech2Model.get_model()
    model = nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


if __name__ == '__main__':
    parser = config.get_arguments()
    opt = parser.parse_args()

    main(opt)