import math
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

folders = [
    # 'forwards-lap-data',
    # 'backwards-lap-data',
    # 'corrective-data',
    'corner-data',
    # 'jungle-one-lap-data',
    # 'jungle-backwards-lap-data',
]

csv_lines = []
for data_folder in folders:
    print('Gathering images and steering angles for folder:', data_folder)
    with open('./training-data/{}/driving_log.csv'.format(data_folder)) as f:
        reader = csv.reader(f)
        for line in reader:
            csv_lines.append(line)

csv_lines = shuffle(csv_lines)
train_lines, validation_lines = train_test_split(csv_lines, test_size=.2, random_state=42)


def data_gen(lines, batch_size):
    images = []
    steering_angles = []
    while True:
        for offset in range(0, len(lines), batch_size):
            for line in lines[offset:offset+batch_size]:
                center_img_path = line[0]
                file_name = center_img_path.split('/')[-1]
                image = cv2.imread('./training-data/{}/IMG/{}'.format(data_folder, file_name))
                images.append(image)
                images.append(np.fliplr(image))

                steering_angle = float(line[3])
                steering_angles.append(steering_angle)
                steering_angles.append(-steering_angle)

                yield shuffle(np.array(images), np.array(steering_angles))


from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

batch = 128
epochs = 1
activation_type = 'elu'
dropout_p = .2

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 127.5) - 1))
model.add(Conv2D(24, (5, 5), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(36, (5, 5), activation=activation_type))
# model.add(MaxPooling2D())
# model.add(Dropout(dropout_p))
model.add(Conv2D(48, (5, 5), activation=activation_type))
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
model.fit_generator(
    generator=data_gen(train_lines, batch),
    steps_per_epoch=math.ceil(len(train_lines)/batch),
    epochs=epochs,
    validation_data=data_gen(validation_lines, batch),
    validation_steps=math.ceil(len(validation_lines)/batch)
)

print('Saved the model')
model.save('model.k')
