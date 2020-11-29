import os

# File Path generator (Folder Path + File Name)
import copy
def file_path_generator(path):
    # path : folder path that contains pictures (same emtions, same train/test)
    # return value : filelist(list type) that contain full paths of pictures
    filelist = os.listdir(path)
    for idx,filename in enumerate(filelist):
        filelist[idx] = path + '/' + filename
    return filelist

# Picture paths for each emotion
train_paths = file_path_generator("D:/Desktop_D/HCI_Project/archive/train")
test_paths = file_path_generator("D:/Desktop_D/HCI_Project/archive/test")
# Folder path example : archive/train/angry/pic1.jpg
# Folder path example : archive/train/angry/pic2.jpg
# Folder path example : archive/train/sad/pic1.jpg
# Folder path example : archive/test/angry/pic1.jpg

# These contain File Path
train_pictures = []
test_pictures = []


for folder_path in train_paths:
    train_pictures.append(file_path_generator(folder_path))
# train_pictures[0][3] : it stores the path of third image in 0 th emotion, 

for folder_path in test_paths:
    test_pictures.append(file_path_generator(folder_path))

# Return value : train_pictures, test_pictures