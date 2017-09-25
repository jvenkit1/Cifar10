import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as kb

kb.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

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
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
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
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 25, batch_size = 64)
# Model.fit uses the datasets provided to train the Neural Network. The Network is trained for 25 epochs with a batch size of 64.
scores = model.evaluate(X_test, y_test, verbose = 0)
print 'Accuracy is ' + str(scores[1]*100)
