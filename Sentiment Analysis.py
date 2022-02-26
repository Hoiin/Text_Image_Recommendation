#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install graphviz')


# # Sentimental Analysis

# In[ ]:


import os
import tensorflow.compat.v2 as tf 

from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[ ]:


tf.enable_v2_behavior()


# In[ ]:


from tensorflow.python.framework.ops import disable_eager_execution 
disable_eager_execution()


# In[ ]:


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


# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images


# In[ ]:


train_angry_fnames = os.listdir( train_angry )
train_happy_fnames = os.listdir( train_happy )

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_angry_pix = [os.path.join(train_angry, fname) 
                for fname in train_angry_fnames[ pic_index-8:pic_index] 
               ]

next_happy_pix = [os.path.join(train_happy, fname) 
                for fname in train_happy_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_angry_pix+next_happy_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') 

    img = mpimg.imread(img_path)
    plt.imshow(img,cmap='gray')

plt.show()


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


# In[ ]:


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss' , factor=0.25, patience=2, min_lr=0.00001,model='auto')


# ## CNN

# In[ ]:


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


# In[ ]:


model.summary()


# In[ ]:


tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None, show_layer_activations=False
)


# In[ ]:


history = model.fit(train_generator,
                    validation_data=test_generator,
#                     steps_per_epoch=100,
                    epochs=100,
#                     validation_steps=50,
                    verbose=1,
                    callbacks=[reduce_lr])


# In[ ]:


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


# In[ ]:


model.save('CNN_7.h5')


# ## ResNet

# In[ ]:


from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall


# In[ ]:


resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet',include_top= False, 
                                                input_shape=(48,48,3)) 

vgg16 = tf.keras.applications.vgg16.VGG16(weights=None,include_top=False,
                                             input_shape=(48,48,3))

vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet',include_top=False,
                                             input_shape=(48,48,3))


# In[ ]:


for layer in resnet.layers:
    layer.trainable = False

for layer in vgg16.layers:
    layer.trainable = False
    
for layer in vgg19.layers:
    layer.trainable = False


# In[ ]:


input_tensor = Input(shape=(48,48,1))
x = layers.Conv2D(3,(1,1),padding='same')(input_tensor)
# x = resnet(x) 
x = vgg16(x)
x = layers.Flatten()(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dropout(0.25)(x)
out = layers.Dense(6, activation='softmax')(x)

model_2 = Model(inputs=input_tensor, outputs=out)

model_2.compile(optimizer=Adam(learning_rate= 0.00005),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy',Precision(), Recall()])
#Precision : 예측한 것 중 실제로 맞은 것
#Recall : 실제 참값 중 예측을 잘 한 것


# In[ ]:


model_2.summary()


# In[ ]:


history_2 = model_2.fit(train_generator,
                        validation_data=test_generator,
                        epochs=50,
                        verbose=1,
                         callbacks=[reduce_lr])


# In[ ]:


model_2.save('Resnet_3.h5')

