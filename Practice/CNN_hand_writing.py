#https://www.youtube.com/watch?v=FyXlvui--Vw&t=348s&ab_channel=%E6%9D%8E%E6%94%BF%E8%BB%92
#我的第一支快速利用TensorFlow 2.0建立CNN進行手寫數字分類
#date 2021.3.15

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
#1層
CNN.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28 ,1))) # 2D的影像，32是convolution完後要有幾個neuron，(3, 3)是convolution的size，
                                                                                   # 中間層儘量都用relu，第一層一定要有input_shpae。
                                                                                   # Output會變成26*26，影像處理的概念。
CNN.add(layers.MaxPooling2D(2, 2)) # (subsampling)通常會搭配MaxPooling(即取最大) 2*2是習慣 13*13*32
#2層
CNN.add(layers.Conv2D(64, (3, 3), activation = "relu")) # 11*11*64
CNN.add(layers.MaxPooling2D(2, 2)) # (subsampling)通常會搭配MaxPooling(即取最大) 2*2是習慣 5*5*64

CNN.add(layers.Flatten()) # 壓平(把矩陣拉直)，5*5*64 = 1600 (64個5*5的圖像)。
CNN.add(layers.Dense(128, activation = "relu"))
CNN.add(layers.Dense(64, activation = "relu"))
CNN.add(layers.Dense(10, activation = "softmax")) # 手寫資料集，0~9共10個。分類用softmax

keras.utils.plot_model(CNN ,to_file='./Practice/model.png' ,show_shapes=True) # 使用Graphviz顯示layers，會將結果儲存在to_file的路徑裡。


#編譯
CNN.compile(optimizer = "Adam",
            loss = keras.losses.sparse_categorical_crossentropy,
            metrics = ["accuracy"])


#開始訓練
CNN.fit(x_train, y_train, epochs = 5) # epochs可以用很多，但可能會overfitting


#算正確率
import numpy as np
np.mean((CNN.predict_classes(x_test) == y_test)) # model.predict_classes()是影片中使用的function，但是在2021-1-1被移除了。

#  UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   
#  if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   
#  if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
#  warnings.warn('`model.predict_classes()` is deprecated and '