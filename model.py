import math
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_data():
    folders = [
        'forwards-lap-data',
        # 'backwards-lap-data',
        # 'corrective-data',
        # 'corner-data',
        # 'jungle-one-lap-data',
        # 'jungle-backwards-lap-data',
    ]

    image_load_data = []
    for data_folder in folders:
        print('Gathering csv lines for folder:', data_folder)
        with open('./training-data/{}/driving_log.csv'.format(data_folder)) as f:
            reader = csv.reader(f)
            for csv_line in reader:
                image_load_data.append((csv_line, data_folder))

    image_load_data = shuffle(image_load_data)

    images = []
    angles = []
    for line, folder in image_load_data:
        center_img_path = line[0]
        file_name = center_img_path.split('/')[-1]
        image = cv2.imread('./training-data/{}/IMG/{}'.format(folder, file_name))
        if image is not None:
            images.append(image)
            images.append(np.fliplr(image))

            steering_angle = float(line[3])
            angles.append(steering_angle)
            angles.append(-steering_angle)

    return shuffle(np.array(images), np.array(angles))


data, labels = get_data()

from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

batch = 128
epochs = 1
activation_type = 'elu'
dropout_p = .2

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 127.5) - 1))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(64, (3, 3), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(64, (3, 3), activation=activation_type))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1162, activation=activation_type))
model.add(Dense(100, activation=activation_type))
model.add(Dense(50, activation=activation_type))
model.add(Dense(10, activation=activation_type))
model.add(Dense(1))

print('Training the model')

model.compile(optimizer='adam', loss='mse')
model.fit(data, labels, batch_size=batch, epochs=epochs, validation_split=.2)

print('Saved the model')
model.save('model.k')
