import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import mixupGenerator as mixupgen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMG_HEIGHT = 224
IMG_WIDTH = 224 
EPOCHS = 15
batch_size = 100

#Base data
trainDirectory = "../Data/train"
validationDirectory = "../Data/valid"
testDirectory = "../Data/test"
 
#Fourier
#trainDirectory = "../FourierData/train"
#validationDirectory = "../FourierData/valid"
#testDirectory = "../FourierData/test"


#Used a smaller dataset for testing
#trainDirectory = "../SmallData/train"
#validationDirectory = "../SmallData/valid"
#testDirectory = "../SmallData/test"


classes = os.listdir(trainDirectory)
num_classes = len(classes) #Note that the dataset on the virtual machine has 195 classes (due to an update of the data set)


#Base Generators
trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

#Basic Augemntation
#trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  #horizontal_flip = True,
								  #rotation_range = 45,
								  #zoom_range = 0.2,
								  #sheer_range = 0.2)
#validDataGen = ImageDataGenerator(rescale = 1./255.) 
#testDataGen = ImageDataGenerator(rescale = 1./255.)

#====================================================================================											
#with mixup
#====================================================================================
trainGen = mixupgen.MixupImageDataGenerator(trainDataGen, 
											trainDirectory,
											batch_size = batch_size,
											img_height=IMG_HEIGHT,
											img_width=IMG_WIDTH,
											distr = "beta",
											params = [0.2, 0.2],
											majority_vote = 1)
											
validGen = validDataGen.flow_from_directory(validationDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 


testGen = testDataGen.flow_from_directory(testDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 										
#====================================================================================											
#with fourier / IGNORE THIS, PREPROCESSING IS HANDLED BY SAVEING NEW IMAGES, see transform_image_and_save.py
#====================================================================================
#trainGen = fouriergen.FourierImageDataGenerator(trainDataGen, 
											#trainDirectory,
											#batch_size = batch_size,
											#img_height=IMG_HEIGHT,
											#img_width=IMG_WIDTH)											

#validGen = validDataGen.flow_from_directory(validationDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 


#testGen = testDataGen.flow_from_directory(testDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 
#====================================================================================											
#Standard
#====================================================================================

#trainGen = trainDataGen.flow_from_directory(trainDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 

##950 img belonging to 190 classes
#validGen = validDataGen.flow_from_directory(validationDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 

##950 img belonging to 190 classes
#testGen = testDataGen.flow_from_directory(testDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 


#====================================================================================											
#
#====================================================================================



mobilenet = tf.keras.applications.MobileNetV2(input_shape = (IMG_HEIGHT, IMG_WIDTH, 3),
											   include_top = False,
                                               weights='None') #'None' for fourier and 'imagenet' for regular data

#We cant examine fourier transform without using trainable = True as the preset weights are based on actual imagery and not angles/amps
mobilenet.trainable = True #we dont alter the pre-trained weights in mobilenet

model = Sequential([
	mobilenet, #mobilenet
	#GlobalAveragePooling2D(), #pooling
	Flatten(), #different apåproach
	Dense(num_classes, activation = "softmax") #predictive
	])

model.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

model.summary()

es = EarlyStopping(monitor = 'loss', min_delta = 0.005, patience = 3, mode = "min", verbose = 1, restore_best_weights=True)
mc = ModelCheckpoint("bestModel.h5", monitor = "val_loss", verbose = 1, save_best_only = True)

#====================================================================================											
#Use this for mixup
#====================================================================================
history = model.fit_generator(trainGen.generate(),
							   steps_per_epoch = 26769//batch_size, #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 50//batch_size,
							   verbose = 1,
							   callbacks = [es, mc])

#====================================================================================											
#Standard
#====================================================================================

#history = model.fit_generator(trainGen,
							  #steps_per_epoch = 26769//batch_size, #training images / batch size
							   #epochs = EPOCHS,
							   #validation_data = validGen,
							   #validation_steps = 975//batch_size,
							   #verbose = 1,
							   #callbacks = [es, mc])

#bestModel = load_model("bestModel.h5")

score = model.evaluate_generator(testGen,
								 975//batch_size)


