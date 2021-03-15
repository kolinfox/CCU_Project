#https://www.youtube.com/watch?v=FyXlvui--Vw&t=348s&ab_channel=%E6%9D%8E%E6%94%BF%E8%BB%92
#我的第一支快速利用TensorFlow 2.0建立CNN進行手寫數字分類
#date 2021.3.5

#tensorflow version 2.4.1

import tensorflow as tf

#x是資料 y是label
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


#正規化資料
x_train = x_train/255
x_test = x_test/255


#可以看到(60000, 28, 28) 表示有60000張 28*28的大小
#x_test是(10000, 28, 28)
print(x_train.shape, x_test.shape)


#三維的其中一維，所以只有1(灰階) 若是RGB則為3
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))


#測試
print("After reshape")
print(x_train.shape, x_test.shape)


#開始建網路
from tensorflow import keras
from tensorflow.keras import layers

CNN = keras.Sequential(name = "CNN")
CNN.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28 ,1))) # 2D的影像，32是convolution完後要有幾個node，(3, 3)是convolution的size，中間層儘量都用relu，第一層一定要有input_shpae

keras.utils.plot_model(CNN ,to_file='./Practice/model.png' ,show_shapes=True) # 使用Graphviz顯示layers，會將結果儲存在to_file的路徑裡