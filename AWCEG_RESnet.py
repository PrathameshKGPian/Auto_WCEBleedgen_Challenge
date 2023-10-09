#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[3]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train'
valid_path = r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\val'


# In[6]:


resnet = tf.keras.applications.ResNet50( include_top=False, 
                                        weights="imagenet", 
                                        input_shape = [224, 224, 3], 
                                        pooling='avg', 
                                        classes=2)


# In[7]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[8]:


# useful for getting number of output classes
folders = glob(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train/*')


# In[9]:


folders


# In[10]:


x = Flatten()(resnet.output)


# In[11]:


len(folders)


# In[12]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[13]:


model.summary()


# In[14]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[15]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[16]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[17]:


test_set = test_datagen.flow_from_directory(r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen\val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[18]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


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


# In[21]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet.h5')


# In[22]:


y_pred = model.predict(test_set)


# In[23]:


y_pred


# In[24]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[25]:


y_pred


# In[ ]:





# In[ ]:




