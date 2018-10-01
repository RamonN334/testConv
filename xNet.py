from keras import layers
from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import ads

img_width, img_height = 200, 200

train_data_dir = './train_folder'
validation_data_dir = './test_folder'
nb_train_samples = 7333
nb_validation_samples = 2445
epochs = 50
batch_size = 16


x_train, y_train, x_test, y_test = ads.load_data()

x_train = x_train.reshape(nb_train_samples, img_width, img_height, 1)
x_test = x_test.reshape(nb_validation_samples, img_width, img_height, 1)
y_max = y_train.max() + 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test, y_max)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train.min())
print(y_train.max())
print(y_test.min())
print(y_test.max())


# if K.image_data_format() == 'channels_first':
#     input_shape = (1, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 1)

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), use_bias=False,
                        input_shape=(200, 200, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(y_max, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

model.save('first_try_net.h5')
