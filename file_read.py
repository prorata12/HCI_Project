import os

# Folder Path
f1_train_path = "D:/Desktop_D/HCI_Project/archive/train/happy"
f1_test_path = "D:/Desktop_D/HCI_Project/archive/test/happy"
f1_label = 0

f2_train_path = "D:/Desktop_D/HCI_Project/archive/train/angry"
f2_test_path = "D:/Desktop_D/HCI_Project/archive/test/angry"
f2_label = 1

# File Path generator (Folder Path + File Name)
import copy
def file_path_generator(path):
    # path : folder path that contains pictures (same emtions, same train/test)
    # return value : filelist(list type) that contain full paths of pictures
    filelist = os.listdir(path)
    for idx,filename in enumerate(filelist):
        filelist[idx] = path + '/' + filename
    return filelist

# These contain File Path
f1_train_list = file_path_generator(f1_train_path)
f1_test_list = file_path_generator(f1_test_path)
f2_train_list = file_path_generator(f2_train_path)
f2_test_list = file_path_generator(f2_test_path)


# Return value: list of [train list, test list, label]
file_list = [[f1_train_list, f1_test_list, f1_label],\
    [f2_train_list, f2_test_list, f2_label]]
