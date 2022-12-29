import glob
from sklearn.model_selection import train_test_split
import os
import config


def write_paths_to_file(paths, file_name):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_name, 'w') as f:
        for path in paths:
            f.write(path + '\n')


def split_data(opt, ignore_dirs):
    # Initialize empty lists to hold file paths
    file_paths = []

    # Use glob to find all files under dir_path and add file paths to file_paths list
    file_paths = glob.glob(opt.dataroot + '/**/*', recursive=True)
    file_paths = [path for path in file_paths if not any(ignore_dir in path for ignore_dir in ignore_dirs)]

    # Split file paths into train, test, and validation sets
    train_paths, val_test_paths, _, _ = train_test_split(file_paths, file_paths, test_size=0.2, random_state=42)
    test_paths, val_paths, _, _ = train_test_split(val_test_paths, val_test_paths, test_size=0.5, random_state=42)

    # Write file paths to respective files
    write_paths_to_file(train_paths, os.path.join(opt.save_directory, 'train.txt'))
    write_paths_to_file(test_paths, os.path.join(opt.save_directory, 'test.txt'))
    write_paths_to_file(val_paths, os.path.join(opt.save_directory, 'validation.txt'))


parser = config.get_arguments()
opt = parser.parse_args()
split_data(opt, ['_background_noise_'])
