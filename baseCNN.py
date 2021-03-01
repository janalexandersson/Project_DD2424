import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import mixupGenerator as mixupgen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import makesubsets


IMG_HEIGHT = 224
IMG_WIDTH = 224 
EPOCHS = 15
batch_size  = 100

#Base data
trainDirectory = "../Data/train"
validationDirectory = "../Data/valid"
testDirectory = "../Data/test"

#Fourier
#trainDirectory = "../FourierDataPhase/train"
#validationDirectory = "../FourierDataPhase/valid"
#testDirectory = "../FourierDataPhase/test"

#Used a smaller dataset for testing
#trainDirectory = "../SmallData/train"
#validationDirectory = "../SmallData/valid"
#testDirectory = "../SmallData/test"


classes = os.listdir(trainDirectory)
num_classes = len(classes)


#Base Generators
trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

# With basic augmentations:
#trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  #horizontal_flip = True,
								  #rotation_range = 25,
								  #zoom_range = 0.2,
								  #shear_range = 0.4,
                                  #brightness_range = (0.5,1.5))
#validDataGen = ImageDataGenerator(rescale = 1./255.) 
#testDataGen = ImageDataGenerator(rescale = 1./255.)


#====================================================================================											
#with mixup
#====================================================================================
#trainGen = mixupgen.MixupImageDataGenerator(trainDataGen, 
											#trainDirectory,
											#batch_size = batch_size,
											#img_height=IMG_HEIGHT,
											#img_width=IMG_WIDTH,
											#distr = "beta",
											#params = [0.2, 0.2],
											#majority_vote = 1)
											
#validGen = validDataGen.flow_from_directory(validationDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 


#testGen = testDataGen.flow_from_directory(testDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 										
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
trainGen = trainDataGen.flow_from_directory(trainDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#950 img belonging to 190 classes
validGen = validDataGen.flow_from_directory(validationDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#950 img belonging to 190 classes
testGen = testDataGen.flow_from_directory(testDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#====================================================================================											
#
#====================================================================================



#General Model. Reliable Model 60% val after a few epochs. Tried changing layers to ascending in base 2 (i.e. 16, 32, 64 etc) as well as the stride from 3 to perhaps 2 or 1. This did however not seem to yield significant changes in the test accuracy. Usually dropout is not used in the convolutional layers due to the low number of parameters but after some experimentation this did actually seem to generate an increase in the testing accuracy.

model = Sequential([
	Conv2D(64, 3, activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
	MaxPooling2D(2,2),
	BatchNormalization(),
	#Dropout(0.4),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Conv2D(64, 3, activation = "relu"),
	MaxPooling2D(2,2),
	BatchNormalization(),
	#Dropout(0.4),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Flatten(),
	Dropout(0.5),
	Dense(512, activation = "relu"),
	BatchNormalization(),
	Dense(num_classes, activation = "softmax")
	])


model.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

model.summary()


###################################################################################
#this is an example on how to plot a batch of images in the training data
#(X, y) = next(trainGen)
#
#def plotImages(images_arr):
#    fig, axes = plt.subplots(1, 5, figsize=(10,10))
#    axes = axes.flatten()
#    for img, ax in zip( images_arr, axes):
#        ax.imshow(img)
#        ax.axis('off')
#    plt.tight_layout()
#    plt.show()
#
#plotImages(X)
###################################################################################


es = EarlyStopping(monitor = 'loss', min_delta = 0.005, patience = 3, mode = "min", verbose = 1, restore_best_weights=True)
mc = ModelCheckpoint("bestModel.h5", monitor = "val_loss", verbose = 1, save_best_only = True)

#====================================================================================											
#Use this for mixup
#====================================================================================
#history = model.fit_generator(trainGen.generate(),
							   #steps_per_epoch = 26769//batch_size, #training images / batch size
							   #epochs = EPOCHS,
							   #validation_data = validGen,
							   #validation_steps = 975//batch_size,
							   #verbose = 1,
							   #callbacks = [es, mc])

#====================================================================================											
#Standard
#====================================================================================
history = model.fit_generator(trainGen,
							   steps_per_epoch = 26769//batch_size, #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 975//batch_size,
							   verbose = 1,
							   callbacks = [es, mc])

#bestModel = load_model("bestModel.h5")

score = model.evaluate_generator(testGen,
								 975//batch_size)



'''
#loop through everything:                               
           
                    
# Create different training sets:
       
makesubsets.create_subsets(small = 5, medium = 50)
                        
#loop to go through all datasets:

subsets = ["RandomSmall", "RandomMedium", "AllData", "BrightSmall",
               "BrightMedium", "BrightFull", "DullSmall", "DullMedium",
               "DullFull", "BWSmall", "BWMedium", "BWFull"]

for subset in subsets:
    
    trainDirectory = "../Subsets/"+subset+"/train"
    if "small" in subset:
        validationDirectory = "../Subsets/"+subset+"/valid"
        testDirectory = "../Subsets/"+subset+"test"    
    elif subset is RandomMedium:
        validationDirectory = "../Subsets/RandomSmall/valid"
        testDirectory = "../Subsets/RandomSmall/test"
    elif subset is BrightMedium:
        validationDirectory = "../Subsets/BrightSmall/valid"
        testDirectory = "../Subsets/BrightSmall/test"    
    elif subset is DullMedium:
        validationDirectory = "../Subsets/DullSmall/valid"
        testDirectory = "../Subsets/DullSmall/test"
    elif subset is BWMedium:
        validationDirectory = "../Subsets/BWSmall/valid"
        testDirectory = "../Subsets/BWSmall/test"


    for aug in ["none", "mixup", "fourier", "basicnocolor", "basicwcolor"]:
        
        trainDataGen = ImageDataGenerator(rescale = 1./255.) 
        validDataGen = ImageDataGenerator(rescale = 1./255.) 
        testDataGen = ImageDataGenerator(rescale = 1./255.)
        
        if aug is "basicnocolor":
            
            trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  horizontal_flip = True,
								  rotation_range = 30,
								  shear_range = 0.4)
                                
            validDataGen = ImageDataGenerator(rescale = 1./255.) 
            testDataGen = ImageDataGenerator(rescale = 1./255.)
            
        if aug is "basicwcolor":
            
            trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  horizontal_flip = True,
								  rotation_range = 30,
								  shear_range = 0.4, 
                                  brightness_range = (0.3, 1.3)) 
                                
            validDataGen = ImageDataGenerator(rescale = 1./255.) 
            testDataGen = ImageDataGenerator(rescale = 1./255.)
            
        if aug is "none" or "basicnocolor" or "basicwcolor":
            trainGen = trainDataGen.flow_from_directory(trainDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

            validGen = validDataGen.flow_from_directory(validationDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 
            
            testGen = testDataGen.flow_from_directory(testDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 
            history = model.fit_generator(trainGen,
							   steps_per_epoch = 26769//batch_size, #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 975//batch_size,
							   verbose = 1)
            
        if aug is "mixup":
           
            trainGen = mixupgen.MixupImageDataGenerator(trainDataGen, 
											trainDirectory,
											batch_size = batch_size,
											img_height=IMG_HEIGHT,
											img_width=IMG_WIDTH,
											distr = "trunc_norm",
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


            history = model.fit_generator(trainGen.generate(),
							   steps_per_epoch = trainGen.steps_per_epoch(), #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 40,
							   verbose = 1)


        #Finish with Fourir as it overwrites the data. 
        if aug is "fourier":
            #use transform_image_and_save.py and edit so that it has path to the created subset folder
'''    
        
                                                
    

                           
