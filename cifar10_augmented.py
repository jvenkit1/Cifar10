import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as kb
from keras.preprocessing.image import ImageDataGenerator

kb.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

# Load datasets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizing Datasets :

X_train = X_train.astype('float32') # We convert the data into float32 type and normalize by dividing by 255
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0

# One Hot Encoding :
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
n_classes = y_test.shape[1]

# Model Definition
# Our Model has 6 Convolutional Layers with Dropout and MaxPooling Layers alternating between each Conv Layer
# We use 2 Dimensional Convolutional Networks
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (3, 32, 32), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))

# Fully Connected Layers
model.add(Dense(1024, activation = 'relu', kernel_constraint = maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu', kernel_constraint = maxnorm(3)))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(n_classes, activation = 'softmax'))

sgd = SGD(lr = 0.01, momentum = 0.9, nesterov = False) # We use Stochastic Gradient Descent as the Optimizer
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy']) # Compiling the Model

# Training the Model :
np.random.seed(seed)
# Training the network in multiple Passes :
# Pass 1 : Rotating the Image
datagen = ImageDataGenerator(rotation_range = 90)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64), samples_per_epoch = X_train.shape[0], nb_epoch = 100, validation_data = (X_test, y_test), verbose = 0)

# Pass 2 : Flipping the image
# Vertical Flip
datagen = ImageDataGenerator(vertical_flip = True)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64), samples_per_epoch = X_train.shape[0], nb_epoch = 100, validation_data = (X_test, y_test), verbose = 0)

datagen = ImageDataGenerator(horizontal_flip = True)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64), samples_per_epoch = X_train.shape[0], nb_epoch = 100, validation_data = (X_test, y_test), verbose = 0)


# Pass 3 : Shifting the Image

datagen = ImageDataGenerator(width_shift_range = 0.2, height_shift_range = 0.2)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64), samples_per_epoch = X_train.shape[0], nb_epoch = 100, validation_data = (X_test, y_test), verbose = 0)

# Training on the original image also [ without any augmentation ]
model.fit(X_train, y_train, epochs = 100, batch_size = 64)
# Model.fit uses the datasets provided to train the Neural Network. The Network is trained for 25 epochs with a batch size of 64.
#scores = model.evaluate(X_test, y_test, verbose = 0)
#print X_test.shape, X_test.shape[0]
preds = model.predict(X_test, batch_size = 64, verbose=0)
pred_class = np.argmax(preds, axis = 1)
actual_class = np.argmax(y_test, axis = 1)
acc = np.sum(pred_class == actual_class)
acc = float(acc)/preds.shape[0]
print acc


#print 'Accuracy is ' + str(scores[1]*100)
