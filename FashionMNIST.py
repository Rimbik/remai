# ref Section:39 FASHION MNIST
#----------------------------#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape :",x_train.shape)
# download done in above with: x_train.shape : (60000, 28, 28)



# the data is in only 2D
# the daya contains (28 x 28): Grey scale image: but CNN needs 3D image for convolution operation
# convolution expects height x width x color 
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train.shape :",x_train.shape) # converted to 3D now
# After Reshape: to 3D: x_train.shape : (60000, 28, 28, 1)

# number of classes
K = len(set(y_train))
print("number of classes: ",K)

# build the model using the function API
i = Input(shape=x_train[0].shape)
x = Conv2D(32,  (3,3), strides=2, activation='relu')(i)
x = Conv2D(64,  (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(i,x)


# compile and fit
# Note: Make sure you are GPU for this
model.compile(
	       optimizer = 'adam',
	       loss='sparse_categorical_crossentropy',
	       metrics=['accuracy']
             )
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# plot it (loss per iteration)
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Val-Loss')
plt.legend()

# NOTE: run seperately otherwise img get mixed in same plot
#plot accuracy per iteration
plt.plot(r.history['accuracy'], label='Acc')
plt.plot(r.history['val_accuracy'], label='Val-Accuracy')
plt.legend()
		

#load image from ggole drive with mount
from google.colab import drive
drive.mount("/content/drive", force_remount=True); # force to get new added images/files mounted


# external ref : https://saturncloud.io/blog/using-tensorflow-to-pass-an-image-to-a-simple-mnist-data-model/
#predict an image ------------------ IMAGE  PREDICTION ------------------->>

from PIL import Image
image = Image.open('/content/drive/My Drive/Colab Notebooks/images/Jeans.jpg').convert('L')
image = image.resize((28, 28))

import numpy as np
x = np.array(image)
x = x.reshape(1, 28, 28) #x.reshape((1, 784))
#x = x / 255.0


# run prediction
prediction = model.predict(x)
#prd = model.predict(x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
digit = np.argmax(prediction)
print("prediction :", digit) # output: prediction : 4

# is the prection closer to accurate as seen 4/10 ? : Need to evaluate (Confusion matrix code not made yet)




