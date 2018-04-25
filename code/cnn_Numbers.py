



from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils
from sklearn.model_selection import train_test_split

input_shape = 64 
no_classes = 10


# dataset prebaration

print('colect dataset') 
numbersPath = glob.glob('../dataset/**/*')

stack =[]
def blure(image):
    img = cv2.imread(image)
    className = image.split('/')[-2][-1]
    stack.append([img,className]) 

list(map(blure,numbersPath))
X = [item[0] for item in stack]
Y = [item[1] for item in stack]
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.25, random_state=42)

print('dataset gethering finished , the trained is : ',len(X_train))





# ### Modal architecture.

# print('model Init start')

# model = Sequential()

# model.add(Conv2D(32,(3,3),strides=1,input_shape=(input_shape,input_shape,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
          
# model.add(Conv2D(64,(3,3),input_shape=(input_shape,input_shape,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
          
# model.add(Conv2D(128,(3,3),input_shape=(input_shape,input_shape,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())

# model.add(Dense(units=1000,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(units = 500, activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(units = 100, activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(units = no_classes, activation = 'softmax'))

# model.summary()
# print(' ended')

# ## code compilation line 
# print('compilation start')

# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# print('end')
# print('image augmantation process')

# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.3,
#                                    zoom_range = 0.35,
#                                    validation_split=0.2)

# validation_datagen = ImageDataGenerator(rescale = 1./255,
#                                         shear_range = 0.3,
#                                         zoom_range = 0.35)


# validation_set = validation_datagen.flow(X_test, y_test,
#                                             save_to_dir='validation',
#                                             save_format='png',
#                                             batch_size = 50)
# training_set = train_datagen.flow(X_train, y_train,
#                                             batch_size = 50)

# print('image augmantation process ended')

# model.fit_generator(training_set,
#                         #  samples_per_epoch = 825,
#                          nb_epoch = 50,
#                          validation_data = validation_set,
#                          nb_val_samples = 204)

# # train_datagen.fit(images, augment=True, seed=seed)
# print('model started saving your prediction model')
# model.save('/output/​ar_numbers_newset_10sept17.h5')
# print('conguratolations you are free to use the classifier :)')
 
