from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical




# load the mnist data
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print('size of train_images is: '+str(train_images.shape))
print('length of train_labels is: '+str(len(train_labels)))
print('size of test_images is: '+str(test_images.shape))
print('length of test_labels is: '+str(len(test_labels)))

# prepare image data 
train_images = train_images.reshape((60000,28*28)).astype('float32')/255
test_images = test_images.reshape((10000,28*28)).astype('float32')/255

# prepare image labels 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# construct the net
network = models.Sequential()
# add two layers, choose the activation function and input&output size
network.add(
        layers.Dense(512,activation = 'relu',input_shape = (28*28,))
        )
network.add(
        layers.Dense(10,activation = 'softmax')
        )
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])

# start to train
network.fit(train_images, train_labels, epochs =  5, batch_size = 128)

# test loss and accuracy
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_loss: ", test_loss)
print("test_acc: ", test_acc)
