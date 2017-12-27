#------------------------------------------------------------------------------
#
# program: Dagger Simulation
#
# description: simulates the dagger algorithm using a cart racing game
#              a neural network learns to autonomously control a car to
#              navigate a course
#
#------------------------------------------------------------------------------
#
# imports
#
#------------------------------------------------------------------------------

# to control cart simulation
# 
from gym_torcs import TorcsEnv

# data modules
#
from PIL import Image
import numpy as np
import imageio

# modules for deep learning
#
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam


#------------------------------------------------------------------------------
#
# globals
#
#------------------------------------------------------------------------------

# image dimensions
# 
IMG_DIM = (64,64,3)

# dimensions for output
# 
ACTION_DIM = 1

# number of steps initial simulation run
# 
STEPS = 1000

# number of dagger iterations
#
DAGGER_ITER = 5

# batch size and number of epochs
# 
BATCH_SIZE = 32
NB_EPOCH = 100

# kernel initializer for neural network
#
KERNEL_INIT = "uniform"

# size to resize images
# 
NEW_IMAGE_SIZE = (256, 256)

# filename prefix for gifs
#
FILE_PREFIX = "dagger_run_{:d}.gif"

# file for neural network results
#
NEURAL_NET_RESULTS = 'results.txt'

#------------------------------------------------------------------------------
#
# functions
#
#------------------------------------------------------------------------------

# function: get_teacher_action
#
# arguments:
#  ob: object to interact with simulation
#
# return: array with 1 element of a float action taken in simulation
#
# get the action of the simulation
# 
def get_teacher_action(ob):

	# get action based on track position and steer angle
	# 
	steer = ob.angle*10/np.pi
	steer -= ob.trackPos*0.10

	# exit gracefully
	# 
	return np.array([steer])
#
# end of function

# function: img_reshape
#
# arguments:
#  input_img: image from simulation to convert to standard image array format
#
# return: three dimensional array of image
#
# converts image from simulation to format recognizable my PIL
# 
def img_reshape(input_img):

	# reshape the image
	# 
	_img = np.transpose(input_img, (1, 2, 0))
	_img = np.flipud(_img)
	_img = np.reshape(_img, (1, IMG_DIM[0], IMG_DIM[1], IMG_DIM[2]))

	# exit gracefully
	# 
	return _img
#
# end of function

# function: img_prepare
#
# arguments:
#  img: image to be prepared to be written to gif
#
# return: three dimensional array of resized image
#
# resizes an image to prepare it to be written to a gif
# 
def img_prepare(img):

	# reshape and resize the image
	# 
	im = Image.fromarray(img_reshape(img)[0])
	im = im.resize(NEW_IMAGE_SIZE, Image.ANTIALIAS)

	# exit gracefully
	# 
	return np.array(im)
#
# end of function

#------------------------------------------------------------------------------
#
# main
#
#------------------------------------------------------------------------------

def main():

	#----------------------------------
	#
	# initial run
	#
	#----------------------------------
	
	# variables to hold simulation output
	# 
	images_all = np.zeros((0, IMG_DIM[0], IMG_DIM[1], IMG_DIM[2]))
	actions_all = np.zeros((0,ACTION_DIM))
	rewards_all = np.zeros((0,))

	img_list = []
	action_list = []
	reward_list = []

	# initialize objects for simulation
	# 
	env = TorcsEnv(vision=True, throttle=False)
	ob = env.reset(relaunch=True)

	# start initial data collection
	# 
	print('Collecting data...')
	with imageio.get_writer(FILE_PREFIX.format(0), mode="I") as w:
		for i in range(STEPS):

			# step throguh simulation and get actions
			# 
			if i == 0:
				act = np.array([0.0])
			else:
				act = get_teacher_action(ob)

			if i%100 == 0:
				print(i)
			ob, reward, done, _ = env.step(act)

			# append simulation outputs to lists
			# 
			img_list.append(ob.img)
			action_list.append(act)
			reward_list.append(np.array([reward]))

			# create videos from images
			#
			w.append_data(img_prepare(ob.img))
		#
		# end of for

	env.end()

	# pack the data into arrays
	# 
	print('Packing data into arrays...')
	for img, act, rew in zip(img_list, action_list, reward_list):
		images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
		actions_all = np.concatenate([actions_all, np.reshape(act, [1,ACTION_DIM])], axis=0)
		rewards_all = np.concatenate([rewards_all, rew], axis=0)

	#----------------------------------
	#
	# create neural network
	#
	#----------------------------------
	
	# credit for model design:
	# model from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
	# 
	model = Sequential()


	model.add(Convolution2D(32, 3, 3, border_mode='same',
							input_shape=IMG_DIM, kernel_initializer=KERNEL_INIT))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, kernel_initializer=KERNEL_INIT))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same', kernel_initializer=KERNEL_INIT))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, kernel_initializer=KERNEL_INIT))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, kernel_initializer=KERNEL_INIT))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(ACTION_DIM, kernel_initializer=KERNEL_INIT))
	model.add(Activation('linear'))

	model.compile(loss='mean_squared_error',
				  optimizer=Adam(lr=1e-3),
				  metrics=['mean_squared_error'])

	# train model from initial run
	# 
	model.fit(images_all, actions_all,
			  batch_size=BATCH_SIZE,
			  nb_epoch=NB_EPOCH,
			  shuffle=True)

	# output file for neural network results
	# 
	output_file = open(NEURAL_NET_RESULTS, 'w')

	#----------------------------------
	#
	# iteratively train model
	#
	#----------------------------------
	
	# aggregate and retrain for multiple dagger iterations
	#
	for itr in range(DAGGER_ITER):

		# create file object
		#
		w = imageio.get_writer(FILE_PREFIX.format(itr+1), mode="I")

		# list to hold observations from simulation
		# 
		ob_list = []

		# initialize objects for simulation
		# 
		env = TorcsEnv(vision=True, throttle=False)
		ob = env.reset(relaunch=True)

		# sum for rewards
		# 
		reward_sum = 0.0

		# step through simulations
		# 
		for i in range(STEPS):

			# predict the action corresponding to the current position of the
			# car
			# 
			act = model.predict(img_reshape(ob.img))

			# take observations for each step
			# 
			ob, reward, done, _ = env.step(act)
			if done is True:
				break
			else:
				ob_list.append(ob)

				# create videos from images
				#
				w.append_data(img_prepare(ob.img))


			# add the reward to the total
			# 
			reward_sum += reward

			# print the current action and rewards
			# 
			print(i, reward, reward_sum, done, str(act[0]))
		#
		# end of for

		# close the file for writing
		#
		w.close()
		
		# announce completion of the episode
		# 
		print('Episode done ', itr, i, reward_sum)

		# print the final rewards total
		# 
		output_file.write('Number of STEPS: %02d\t Reward: %0.04f\n'%(i, reward_sum))
		env.end()

		# if the car has gone for the number of steps without crashing, stop
		# the simulation and save the gif
		# 
		if i==(STEPS-1):
			break

		# get the images and actions for each observation
		# 
		for ob in ob_list:

			# add images to arrays
			# 
			images_all = np.concatenate([images_all, img_reshape(ob.img)],
										axis=0)
			actions_all = np.concatenate([actions_all,
										  np.reshape(get_teacher_action(ob),
													 [1,ACTION_DIM])], axis=0)
		#
		# end of for

			
		# iteratively retrain the model
		# 
		model.fit(images_all, actions_all,
					  batch_size=BATCH_SIZE,
					  nb_epoch=NB_EPOCH,
					  shuffle=True)

	#
	# end of for
#
# end of main

# begin gracefully
#
if __name__ == "__main__":
	main()
#
# end of program
