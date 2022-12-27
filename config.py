import argparse
def get_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    #basic arguments
    parser.add_argument('--name', default='exp_1', help='experiment name')
    parser.add_argument('--dataroot',require = True, default='/home/hay/Downloads/speech_commands_v0.01', help='output file path')
    parser.add_argument('--save_directory',  default='save_dir', help='folder to save split data files')
    parser.add_argument('--is_train',require = True, action = 'store_true')
    parser.add_argument('--run_type', default='train', help = 'for inferencr insert test, and for validation inert valid')
    parser.add_argument('--noise_dir_path',help= 'directory path to noise')
    
    #audio parser arguments    
    parser.add_argument('--num_features', default=40)
    parser.add_argument('--window_duration_ms', default=20)
    parser.add_argument('--window_step_ms',  default=10)
    parser.add_argument('--sample_rate',  default=16000)
    parser.add_argument('--sample_length',  default=16000)
    parser.add_argument('--noise_strength',  default=0.3, help = 'added noise signal strength')
    parser.add_argument('--max_signal_shift_samples',  default=1600, help = 'used as augmentation in train, shifts the signal')
    
    #train arguments
    parser.add_argument('--shift_signal_freq',default= 0.5, help='priibability to shift signal')
    parser.add_argument('--noise_freq',default= 0.5, help='priibability to add noise to signal')
    parser.add_argument('--silence_freq',default= 0.05, help='priibability to replace data with silence')
    return parser


