
import os
import tensorflow.compat.v2 as tf 
tf.enable_v2_behavior()
from tensorflow.python.framework.ops import disable_eager_execution 
disable_eager_execution()
#This code is for using tensorflow on M1 Mac
#If your device is not M1 Mac, just import tensorflow as usual

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

#Setting directory
base_dir = './'

train_dir = os.path.join(base_dir, 'train')
train_angry = os.path.join(train_dir,'angry')
train_disgusted = os.path.join(train_dir,'disgusted')
train_fearful = os.path.join(train_dir,'fearful')
train_sad = os.path.join(train_dir,'sad')
train_surprised = os.path.join(train_dir,'surprised')
train_happy = os.path.join(train_dir,'happy')

test_dir = os.path.join(base_dir, 'test')
test_angry = os.path.join(test_dir,'angry')
test_disgusted = os.path.join(test_dir,'disgusted')
test_fearful = os.path.join(test_dir,'fearful')
test_sad = os.path.join(test_dir,'sad')
test_surprised = os.path.join(test_dir,'surprised')
test_happy = os.path.join(test_dir,'happy')

#Dataset Generator
train_datagen = ImageDataGenerator(rescale = 1/255.0,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale= 1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 64,
                                                    class_mode = 'categorical',
                                                    color_mode="grayscale",
                                                    target_size = (48,48),
                                                   shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size = 64,
                                                  class_mode = 'categorical',
                                                  color_mode="grayscale",
                                                  target_size = (48,48),
                                                 shuffle = False)







reduce_lr = ReduceLROnPlateau(monitor='val_loss' , factor=0.25, patience=2, min_lr=0.00001,model='auto')


#CNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.metrics import Precision, Recall

model = Sequential()

# Conv Block 1
model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 2
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 3
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 3
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected Block 1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected Block 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy',Precision(), Recall()])



history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs=100,
                    verbose=1,
                    callbacks=[reduce_lr])


acc      = history.history[ 'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[ 'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

plt.plot  ( epochs, acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )


model.save('CNN.h5')


#Pretrained_Model : VGG16

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall


vgg16 = tf.keras.applications.vgg16.VGG16(weights=None,include_top=False,
                                             input_shape=(48,48,3))


for layer in vgg16.layers:
    layer.trainable = False
    

input_tensor = Input(shape=(48,48,1))
x = layers.Conv2D(3,(1,1),padding='same')(input_tensor) 
x = vgg16(x)
x = layers.Flatten()(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dropout(0.25)(x)
out = layers.Dense(6, activation='softmax')(x)

model_2 = Model(inputs=input_tensor, outputs=out)

model_2.compile(optimizer=Adam(learning_rate= 0.00005),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy',Precision(), Recall()])


history_2 = model_2.fit(train_generator,
                        validation_data=test_generator,
                        epochs=50,
                        verbose=1,
                         callbacks=[reduce_lr])


model_2.save('VGG16.h5')

