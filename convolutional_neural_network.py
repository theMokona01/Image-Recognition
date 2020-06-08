import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import tkinter as tk

##########################
#Image Data Preprocessing
##########################

#Training set preprocesing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True    
    )

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64), #image size
    batch_size=32, #default value
    class_mode='binary' #or categorical for more choices
    )

#Test set preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

###############################
#Building the CNN Architecture
###############################

cnn = tf.keras.models.Sequential()

#Step#1 Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#Step#2 Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#second conv layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Step#3 Flattening
cnn.add(tf.keras.layers.Flatten())

#Step#4 Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Step#5 Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

###############################
#Training the CNN Architecture
###############################

#compile the cnn
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#train cnn on training set and evaluate with test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

#######################
#Making the prediction
#######################

test_image = image.load_img('dataset/single_prediction/layka.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) #add extra fake dimension
result = cnn.predict(test_image)

#get which value (0 or 1) corresponds to either cat or dog from training set
training_set.class_indices
if result[0][0] == 1: #dog
    prediction = 'Dog'
else: #cat
    prediction = 'Cat'
    
window = tk.Tk()
windowWidth = window.winfo_reqwidth()
windowHeight = window.winfo_reqheight()
positionRight = int(window.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(window.winfo_screenheight()/2 - windowHeight/2)
window.geometry("+{}+{}".format(positionRight, positionDown))

greeting = tk.Label(text=f'The image is a {prediction}!')
greeting.config(font=("Courier", 24))
greeting.pack()

window.mainloop()



