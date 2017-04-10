import csv
import cv2
import numpy as np


def get_data(data_folder):
    lines = []
    with open('./{}/driving_log.csv'.format(data_folder)) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    images = []
    steering_angles = []
    for line in lines:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        image = cv2.imread('./{}/IMG/{}'.format(data_folder, file_name))
        images.append(image)

        steering_angle = float(line[3])
        steering_angles.append(steering_angle)

    return np.array(images), np.array(steering_angles)


from keras.models import Sequential
from keras.layers import Dense, Flatten

features, labels = get_data('one-lap-data')
features = features / 255

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(features, labels, batch_size=128, epochs=10, validation_split=.2, shuffle=True)

model.save('model.k')
