import torch
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='plot results')
    parser.add_argument('--models_dir_path', default='results\\', type=str)
    args = parser.parse_args()

    return args


def plot_res(opt):
    data = {'description': [],
            'method': [],
            'top1 max': [],
            'argmax': [],
            'top1 last': [],
            'top1 last average 5': [],
            'current length': [],
            }
    df = pd.DataFrame(data=data)

    top1s, losses, methods, descriptions = [], [], [], []
    for file in glob.glob(opt.models_dir_path + '**/*.tar', recursive=True):

        if '\\' in file:
            file_array_of_names = file.split('\\')[-1].split('_')
        elif '/' in file:
            file_array_of_names = file.split('/')[-1].split('_')
        else:
            file_array_of_names = file.split('_')

        method = file_array_of_names[0]

        loaded_model = torch.load(file, map_location=torch.device('cpu'))
        top1_max = max(loaded_model['test_prcition1'])
        argmax = np.argmax(loaded_model['test_prcition1'])
        top1_last = loaded_model['test_prcition1'][-1]
        average_5 = np.mean(loaded_model['test_prcition1'][-5:])
        current_length = len(loaded_model['test_prcition1'])

        top1s.append(loaded_model['test_prcition1'])
        losses.append(loaded_model['test_losses'])
        methods.append(method)

        append_list = [file.split('/')[-1], method, top1_max, argmax, top1_last, average_5, current_length]
        df = df.append(pd.Series(append_list, index=df.columns), ignore_index=True)

    df.to_csv(opt.models_dir_path + 'df.csv')

    [plt.plot(top1) for top1 in top1s]
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(methods)
    plt.savefig(opt.models_dir_path + 'accuracy.png')
    plt.show()
    plt.close()

    [plt.plot(loss) for loss in losses]
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(methods)
    plt.savefig(opt.models_dir_path + 'losses.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    opt = parse_args()
    plot_res(opt)