import sys
import os
import shutil
import logging

_DATASET_DIR = ''
_BASE_DIR = './dataset'
_TRAIN_DIR = 'train'
_VALIDATION_DIR = 'validation'
_TEST_DIR = 'test'


def welcome():
    print('\n-------------------------------------------------------------\n' +
          'This script are used for creation training, validation and\n' +
          'testing data from datasets UTKFace and any others in future.\n' +
          'author: Igor Sitnikov\n' +
          'organization: AILabs\n' +
          '-------------------------------------------------------------\n')


def utkface_create_data(valid_dir=True, label_mask=0):
    try:
        print('UTKFace dataset\nlink: https://susanqq.github.io/UTKFace/\n')
        print('# Creation of folders...')

        if not os.path.exists(_BASE_DIR):
            os.mkdir(_BASE_DIR)

        train_dir = os.path.join(_BASE_DIR, _TRAIN_DIR)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        if (valid_dir):
            valid_dir = os.path.join(_BASE_DIR, _VALIDATION_DIR)
            if not os.path.exists(valid_dir):
                os.mkdir(valid_dir)

        test_dir = os.path.join(_BASE_DIR, _TEST_DIR)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        print("# Folders are created")

        files = os.listdir(_DATASET_DIR)
        train_indices = list(filter(lambda i: i % 4 != 0, range(len(files))))
        test_indices = list(filter(lambda i: i % 4 == 0, range(len(files))))
        valid_indices = list(filter(lambda i: i % 2 == 0, test_indices))
        test_indices = list(filter(lambda i: i % 2 != 0, test_indices))

        print("# Found {0} files".format(len(files)))
        print("# Copying of files...")

        for i in train_indices:
            print(i)
            copy_file(train_dir, files[i], label_mask)

        for i in valid_indices:
            copy_file(valid_dir, files[i], label_mask)

        for i in test_indices:
            copy_file(test_dir, files[i], label_mask)

        # with open("y_train.csv".format(_TRAIN_FOLDER), "w") as tr_f, open("y_test.csv".format(_TEST_FOLDER), "w") as t_f:
        #     for i in range(len(files)):
        #         print("{0}-th file is moving".format(i))
        #         old_filepath = os.path.join(_DATASET_FOLDER, files[i])
        #         if i % 4 == 0:
        #             new_filepath = os.path.join(_TEST_FOLDER, files[i])
        #             shutil.copy2(old_filepath, new_filepath)
        #             t_f.write(', '.join([files[i], files[i].split("_")[0]]))
        #             t_f.write('\n')
        #         else:
        #             new_filepath = os.path.join(_TRAIN_FOLDER, files[i])
        #             shutil.copy2(old_filepath, new_filepath)
        #             tr_f.write(', '.join([files[i], files[i].split("_")[0]]))
        #             tr_f.write('\n')

        print("# Creation of dataset is success")
    except Exception as err:
        logging.error("Error in creation of dataset: {0}".format(err))


def copy_file(base_path, filepath, label_mask):
    label_text = filepath.split("_")[label_mask]
    label_dir = os.path.join(base_path, label_text)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    src = os.path.join(_DATASET_DIR, filepath)
    dst = os.path.join(label_dir, filepath)
    shutil.copyfile(src, dst)

if __name__ == '__main__':
    welcome()
    args_num = len(sys.argv)
    if args_num >= 3 and args_num <= 4:
        dataset_title = sys.argv[1]
        _DATASET_DIR = sys.argv[2]
        valid_dir = True
        if args_num == 4:
            valid_dir = False

        if (dataset_title == 'utkface'):
            utkface_create_data(valid_dir=valid_dir)
    else:
        Exception('Usage: python dataset_creation_util.py' +
                  'utkface dataset_folder_path --no-valid-dir')
