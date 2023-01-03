import argparse


def get_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # basic arguments
    parser.add_argument('--name', default='exp_1', help='experiment name')
    parser.add_argument('--dataroot', default='/home/hay/Downloads/speech_commands_v0.01', help='output file path')
    parser.add_argument('--save_directory', default='save_dir', help='folder to save split data files')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--run_type', default='train', help='for inferencr insert test, and for validation inert valid')
    parser.add_argument('--noise_dir_path', help='directory path to noise')

    # audio parser arguments
    parser.add_argument('--num_features', default=40)
    parser.add_argument('--window_duration_ms', default=20)
    parser.add_argument('--window_step_ms', default=10)
    parser.add_argument('--sample_rate', default=16000)
    parser.add_argument('--sample_length', default=16000)
    parser.add_argument('--noise_strength', default=0.3, help='added noise signal strength')
    parser.add_argument('--max_signal_shift_samples', default=1600,
                        help='used as augmentation in train, shifts the signal')

    # train arguments
    parser.add_argument('--shift_signal_freq', default=0.1, help='probability to shift signal')
    parser.add_argument('--noise_freq', default=0.1, help='probability to add noise to signal')
    parser.add_argument('--silence_freq', default=0, help='probability to replace data with silence')
    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--print_freq', default=25, type=int, help='train and test print frequency')
    parser.add_argument('--n_cnn_layers', default=3, type=int, help='number of CNN layers')
    parser.add_argument('--n_rnn_layers', default=5, type=int, help='number of RNN layers')
    parser.add_argument('--rnn_dim', default=512, type=int, help='number of RNN layers')
    parser.add_argument('--n_class', default=31, type=int, help='number of classes')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout prop')

    return parser
