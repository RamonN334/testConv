import sys
import os
import shutil
import logging

_DATASET_FOLDER = ""
_TRAIN_FOLDER = './train_folder/'
_TEST_FOLDER = './test_folder/'


def create_data():
    try:
        if not os.path.exists(_TRAIN_FOLDER):
            os.mkdir(_TRAIN_FOLDER)

        if not os.path.exists(_TEST_FOLDER):
            os.mkdir(_TEST_FOLDER)

        files = os.listdir(_DATASET_FOLDER)
        print("Found {0} files".format(len(files)))
        with open("y_train.csv".format(_TRAIN_FOLDER), "w") as tr_f, open("y_test.csv".format(_TEST_FOLDER), "w") as t_f:
            for i in range(len(files)):
                print("{0}-th file is moving".format(i))
                old_filepath = os.path.join(_DATASET_FOLDER, files[i])
                if i % 4 == 0:
                    new_filepath = os.path.join(_TEST_FOLDER, files[i])
                    shutil.copy2(old_filepath, new_filepath)
                    t_f.write(', '.join([files[i], files[i].split("_")[0]]))
                    t_f.write('\n')
                else:
                    new_filepath = os.path.join(_TRAIN_FOLDER, files[i])
                    shutil.copy2(old_filepath, new_filepath)
                    tr_f.write(', '.join([files[i], files[i].split("_")[0]]))
                    tr_f.write('\n')

        print("Creation of dataset is success")
    except Exception as err:
        logging.error("Error in creation of dataset {0}".format(err))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        _DATASET_FOLDER = sys.argv[1]
        create_data()
    else:
        Exception("usage: python dataset_creation_util.py dataset_folder_path")
