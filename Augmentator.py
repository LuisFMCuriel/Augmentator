import os
import numpy as np
import tensorflow as tf
from tifffile import imread, imsave
import shutil
import random
import matplotlib.pyplot as plt
#tf.config.run_functions_eagerly(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1)


def Augmentate_WFandLF(WF, LF, SEED):
  n_WF = np.zeros_like(WF)
  #RR = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, seed=[1,2,3])
  #Start changing the stack (WF)
  for i in range(WF.shape[0]):
    tf.random.set_seed(SEED)
    RR = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = SEED)
    RF = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed = SEED)
    RZ = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, seed=SEED)
    #tf.random.set_seed(1)
    img = WF[i,:,:]
    img = tf.expand_dims(img, 0)
    img = tf.expand_dims(img, -1)
    img = tf.cast(img, tf.float32)

    #tf.random.set_seed(1)
    #RR = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = 1)
    n_img = RR(img)
    #tf.random.set_seed(1)
    #RZ = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, seed=1)
    n_img = RZ(n_img)
    #tf.random.set_seed(1)
    n_img = RF(n_img)
    #RF = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed = 1)
    n_WF[i,:,:] = n_img[0,:,:,0]
  #Finish with LF
  img = tf.expand_dims(LF, 0)
  img = tf.expand_dims(img, -1)
  img = tf.cast(img, tf.float32)
  tf.random.set_seed(SEED)
  RR = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = SEED)
  RF = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed = SEED)
  RZ = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, seed=SEED)
  n_LF = RR(img)
  n_LF = RZ(n_LF)
  n_LF = RF(n_LF)

  return n_WF.astype("uint16"), np.array(n_LF[0,:,:,0]).astype("uint16")


path_r_WF = r"F:\LF-MuscleWormDataset\WF"
path_r_LF = r"F:\LF-MuscleWormDataset\LF"
Path_s_WF = r"F:\LF-MuscleWormDataset\3D-AugmentationData\Augmentation_WF"
Path_s_LF = r"F:\LF-MuscleWormDataset\3D-AugmentationData\Augmentation_LF"

if os.path.exists(Path_s_WF): #Create directories for the registered images
  shutil.rmtree(Path_s_WF)
if os.path.exists(Path_s_LF): #Create directories for the registered images
  shutil.rmtree(Path_s_LF)

#Delete checkpoints if you already have one
#if os.path.exists(os.path.join(path_r_WF, ".ipynb_checkpoints")):
#  shutil.rmtree(os.path.join(path_r_WF, ".ipynb_checkpoints"))
#if os.path.exists(os.path.join(path_r_LF, ".ipynb_checkpoints")):
#  shutil.rmtree(os.path.join(path_r_LF, ".ipynb_checkpoints"))

os.makedirs(Path_s_LF)
os.makedirs(Path_s_WF)

Times_dataset = 3
for i, filename in enumerate(os.listdir(path_r_WF)):
	WF = imread(os.path.join(path_r_WF, filename))
	LF = imread(os.path.join(path_r_LF, filename))
	for j in range(Times_dataset):
		s = np.random.randint(0,10000)
		n_WF, n_LF = Augmentate_WFandLF(WF, LF, s)
		imsave(os.path.join(Path_s_WF, str(j+1)+"-"+filename), n_WF)
		imsave(os.path.join(Path_s_LF, str(j+1)+"-"+filename), n_LF)