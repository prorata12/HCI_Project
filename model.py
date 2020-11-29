import tensorflow as tf
import numpy as np
import copy

x_pos_train = np.load('D:/Desktop_D/HCI_Project/x_pos_train.npy', allow_pickle = True)
print(len(x_pos_train[0]))
y_pos_train = np.load('D:/Desktop_D/HCI_Project/y_pos_train.npy', allow_pickle = True)
z_pos_train = np.load('D:/Desktop_D/HCI_Project/z_pos_train.npy', allow_pickle = True)

x_pos_test = np.load('D:/Desktop_D/HCI_Project/x_pos_test.npy', allow_pickle = True)
y_pos_test = np.load('D:/Desktop_D/HCI_Project/y_pos_test.npy', allow_pickle = True)
z_pos_test = np.load('D:/Desktop_D/HCI_Project/z_pos_test.npy', allow_pickle = True)

# x_pos_train : [emotion][pictures][x_pos(468)]
num_emotion = len(x_pos_train)
X_train = []
y_train = []
X_test = []
y_test = []
for emotion in range(num_emotion):
    for idx, _ in enumerate(x_pos_train[emotion]):
        one_sample = x_pos_train[emotion][idx] + y_pos_train[emotion][idx] + z_pos_train[emotion][idx]
        X_train.append(one_sample)
        y_train.append(emotion)

for emotion in range(num_emotion):
    for idx, _ in enumerate(x_pos_test[emotion]):
        one_sample = x_pos_train[emotion][idx] + y_pos_train[emotion][idx] + z_pos_train[emotion][idx]
        X_test.append(one_sample)
        y_test.append(emotion)


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_norm = (X_train - mean_vals) / std_val
X_test_norm = (X_test - mean_vals) / std_val

y_train_onehot = tf.keras.utils.to_categorical(y_train)
np.random.seed(42)
