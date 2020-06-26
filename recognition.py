import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import numpy as np

def getData():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

	num_pixel_height = 32
	num_pixel_width = 32
	num_color_channels = 3

	input_shape = (num_pixel_height, num_pixel_width, num_color_channels)
	x_train = x_train.reshape(x_train.shape[0], num_pixel_height, num_pixel_width, num_color_channels)
	x_test = x_test.reshape(x_test.shape[0], num_pixel_height, num_pixel_width, num_color_channels)

	return (x_train, y_train), (x_test, y_test), input_shape


def loadModel():
	return tf.keras.models.load_model("model.h5")

def trainModel(x_train, y_train, input_shape, conv_filters=32, conv_kernel=8, pool=2, dense_num=64, epochs=10, verbose=1, batch_size=None, save=True):
	model = keras.Sequential()
	model.add(layers.Conv2D(conv_filters, kernel_size=(conv_kernel,conv_kernel), input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size=(pool,pool)))
	model.add(layers.Flatten())
	model.add(layers.Dense(dense_num, activation=tf.nn.relu))
	model.add(layers.Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)
	if save:
		model.save("model.h5")
	return (model, history)

def test(model, x_test, y_test):
	results = model.evaluate(x_test, y_test, batch_size=128)
	print("current model loss, accuracy: ", results)

def configModel(x_train, y_train, input_shape):
	bestModel = None
	bestLoss = 1000
	cfb = 0
	ckb = 0
	poolb = 0
	dense_numb = 0
	num = 0
	for cf in range(24, 40):
		for ck in range(4, 8):
			for pool in range(2, 5):
				for dense_num in range(64, 128):
					(model, history) = trainModel(x_train, y_train, input_shape, conv_filters=cf, conv_kernel=ck, pool=pool, dense_num=dense_num, epochs=1, batch_size=180, verbose=0, save=False)
					if(history.history['loss'][0] < bestLoss):
						bestModel = model
						bestLoss = history.history['loss'][0]
						cfb = cf
						ckb = ck
						poolb = pool
						dense_numb = dense_num
						print("new best {}".format(num))
					num = num+1
	print("dense_num: {}".format(dense_numb))
	print("pool: {}".format(poolb))
	print("conv_kernel: {}".format(ckb))
	print("conv_filters: {}".format(cfb))


(x_train, y_train), (x_test, y_test), input_shape = getData()
x_train = x_train/255
x_test = x_test/255

if(len(sys.argv) > 1 and sys.argv[1] == "train"):
	trainModel(x_train, y_train, input_shape)
elif(len(sys.argv) > 1 and sys.argv[1] == "config"):
	configModel(x_train, y_train, input_shape)
else:
	model = loadModel()
	test(model, x_test, y_test)


