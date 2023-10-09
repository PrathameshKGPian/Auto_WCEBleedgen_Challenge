#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing required libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[9]:


# setting the image size  
IMAGE_SIZE = [224, 224]

# storing the training and validation datasets path
train_path = r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train'
valid_path = r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\val'


# In[6]:


# defining the VGG19 model
vgg19 = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[8]:


#training with existing weights
for layer in vgg19.layers:
    layer.trainable = False


# In[11]:


# useful for getting number of output classes
folders = glob(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train/*')


# In[12]:


folders


# In[14]:


# flattening out the output layer
x = Flatten()(vgg19.output)


# In[15]:


len(folders)


# In[16]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg19.input, outputs=prediction)


# In[17]:


# model's architecture
model.summary()


# In[18]:


# telling the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[20]:


# Using the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[21]:


# setting up training set
training_set = train_datagen.flow_from_directory(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[23]:


# setting up testing set
test_set = test_datagen.flow_from_directory(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[24]:


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[27]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_vgg19.h5')


# In[28]:


# predictions for testdataset
y_pred = model.predict(test_set)


# In[29]:


y_pred


# In[30]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[31]:


# final predictions for test dataset 
y_pred


# In[32]:





# In[ ]:





# In[ ]:




