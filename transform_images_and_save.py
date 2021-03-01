from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence
import scipy.stats
import matplotlib.image as mimage
import math


#MAKE A BACKUP OF THE IMAGES BEFORE RUNNING THIS BECAUSE IT WILL OVERWRITE EXISTING IMAGES
def fourier_transform_folder_amplitude(directory, to_directory):
	for i in os.listdir(directory):
		folders = []
		folders.append(os.path.join(directory, i))
		to_folderpath = os.path.join(to_directory, i)
		try:
			os.makedirs(to_folderpath)
		except FileExistsError:
			print("Directory already exist, saving images in the existing folders and overwriting existing images.")
			
		for j in folders:
			for k in os.listdir(j):
				filename = os.path.join(j, k)
				#print(filename)
				img=mimage.imread(filename)
				img = img/255
				img = np.fft.fftshift(np.fft.fft2(img))
				img = np.log(np.abs(img)+1)
				
				#rescale to [0,1] if wanted
				img = (img - img.min()) / (img.max() - img.min())
				
				filename = to_directory + filename[len(directory)::]
				mimage.imsave(filename, img)

def fourier_transform_folder_phase(directory, to_directory):
	for i in os.listdir(directory):
		folders = []
		folders.append(os.path.join(directory, i))
		to_folderpath = os.path.join(to_directory, i)
		try:
			os.makedirs(to_folderpath)
		except FileExistsError:
			print("Directory already exist, saving images in the existing folders and overwriting existing images.")
			
		for j in folders:
			for k in os.listdir(j):
				filename = os.path.join(j, k)
				#print(filename)
				img=mimage.imread(filename)
				img = img/255
				img = np.fft.fftshift(np.fft.fft2(img))
				#gives the angles in radians in the range [-pi, pi]
				#Important! imaginary part first in arctan2
				ang = np.arctan2(img.imag, img.real)
				
				#rescale to [0,1] if wanted
				ang = (ang - ang.min()) / (ang.max() - ang.min())
				
				filename = to_directory + filename[len(directory)::]
				mimage.imsave(filename, ang)
				

def fourier_transform_folder_both(directory, to_directory):
	for i in os.listdir(directory):
		folders = []
		folders.append(os.path.join(directory, i))
		to_folderpath = os.path.join(to_directory, i)
		try:
			os.makedirs(to_folderpath)
		except FileExistsError:
			print("Directory already exist, saving images in the existing folders and overwriting existing images.")

		for j in folders:
			for k in os.listdir(j):
				filename = os.path.join(j, k)
				
				img=mimage.imread(filename)
				img = img/255
				img = np.fft.fftshift(np.fft.fft2(img))
				#gives the angles in radians in the range [-pi, pi]
				#Important! imaginary part first in arctan2
				ang = np.arctan2(img.imag, img.real)
				
				#rescale to [0,1] if wanted
				ang = (ang - ang.min()) / (ang.max() - ang.min())
				
				img = np.log(np.abs(img)+1)
				img = (img - img.min()) / (img.max() - img.min())
				
				#concatenate along the second axis, that is, put the images side by side
				both = np.concatenate([ang, img], -2)
				
				filename = to_directory + filename[len(directory)::]
				#print(filename)
				mimage.imsave(filename, both)

#MAKE A BACKUP OF THE IMAGES BEFORE RUNNING THIS BECAUSE IT WILL OVERWRITE EXISTING IMAGES

#fourier_transform_folder_both("../SmallData/valid", "../SmallData2/valid")

print("Amp")
fourier_transform_folder_amplitude("../Data/valid", "../FourierDataAmp/valid")
fourier_transform_folder_amplitude("../Data/test", "../FourierDataAmp/test")
fourier_transform_folder_amplitude("../Data/train", "../FourierDataAmp/train")
print("Phase")
fourier_transform_folder_phase("../Data/valid", "../FourierDataPhase/valid")
fourier_transform_folder_phase("../Data/test", "../FourierDataPhase/test")
fourier_transform_folder_phase("../Data/train", "../FourierDataPhase/train")		
print("Both")
fourier_transform_folder_both("../Data/valid", "../FourierDataBoth/valid")
fourier_transform_folder_both("../Data/test", "../FourierDataBoth/test")
fourier_transform_folder_both("../Data/train", "../FourierDataBoth/train")




