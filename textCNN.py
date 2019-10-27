
"""
name: textCNN with tf2.0
copyright @SkyOrca

1.x->2.0版本注意事项：
为什么要在这里用tf.nn.conv2d而不是Conv2D层：Conv2D是层类，不接受显示输入，是sequential模型堆叠时的组件。而tf.nn.conv2d是
一个函数，在这里我们的卷积层参数是根据卷积核大小3,4,5而改变而且是一个中间计算步骤，不适合Conv2D。
tf.nn.maxpool同理
然后通过一个自定义的卷积层完成textCNN的计算,接在biLSTM后面。

卷积核的计算：假设向量嵌入维数是embed_size， 二维卷积核是[3, embed_size]，[4,embed_size],[5,embed_size]。它在第三维上是1，第四维是输出channels设定成128即有128个核同时计算。
卷积核：[3,embed_size,1,128]  [4,embed_size,1,128]  [5,embed_size,1,128] (height, width, in_channels, out_channels)
注意：如果CNN接在LSTM前面或者是textCNN版本，则embed_size是词向量维数如256.接在LSTM后面时，就变成LSTM编码输入的维数2*LSTM_CELLS

池化核的计算：根据论文，每种卷积池化后的向量仅有1维，所以 (sequence_length-conv_filter_size+1)-pool_filter_size+1 = 1,即只在第二维上有池化
池化核：[1, pool_filter_size, 1,1]

张量形状计算：输入是二维张量（batch, sequence_length），先加入一个embedding layer扩展为(batch, sequence_length, embed_length)
之后扩展一个维度变成(batch, sequence_length, embed_size，1)适配四维卷积。

卷积后height=sequence_length-3(or4,5)+1  width=embed_size-embed_size+1 = 1, 对应论文配图里的那个一维长向量
经过三个并行卷积层，每层出来的结果是(batch, height, width, 128)，再经过池化变成（batch, height-pool_filter_size+1, width, 128）即（batch, 1, 1, 128）
然后三个卷积-池化的结果在最后一个维度拼起来，变成(batch, 1, 1, 384)
最后reshape成（batch, 384）

?不知道为什么我的池化核大小比原码少一个+1。原论文做的1-max pooling而我是2-max?但是按照原码的总是维数差1报错...

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from batchgenerater import BatchGenerator
from utils import preprocess_v3

## hparams
EMBEDDING_SIZE = 256
EPOCHS = 20
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH=3000 #increase, acc起点低，上升速度变慢，收敛结果也变差？
LABEL_SMOOTH = 0.3
reg = tf.keras.regularizers.l2(l=0.001)
n_classes = 3
filter_sizes=[2,3,4]
num_filters=128
LSTM_UNITS = 100

train_data, train_labels, test_id, test_data, embeddings_matrix = preprocess_v3()

class Expand(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        return tf.expand_dims(input, -1)

class MyConvLayer(tf.keras.Model):
    def __init__(self, filter_sizes, num_filters, embed_size,sequence_length):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.sequence_length = sequence_length
       # self.bn = BatchNormalization()
        self.initializer = tf.random_normal_initializer(stddev=0.1)   
        #在这里修改卷积核参数，add_variable加入自定义变量，因为形状完全确定所以在init里初始化而不放在build里
        self.w1 = self.add_variable(name='w1',shape=[3,self.embed_size, 1, self.num_filters],initializer=self.initializer)
        self.w2 = self.add_variable(name='w2',shape=[4,self.embed_size, 1, self.num_filters],initializer=self.initializer)
        self.w3 = self.add_variable(name='w3',shape=[5,self.embed_size, 1, self.num_filters],initializer=self.initializer)
        self.W =[self.w1, self.w2, self.w3]   


    def call(self,input):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # CONVOLUTION LAYER
            filter = self.W[i]
            
            x = tf.nn.conv2d(input, filter,strides=[1, 1, 1, 1],padding="VALID",name="conv")
            # NON-LINEARITY
            x = tf.nn.relu(x)
            # MAXPOOLING  
            x_pooled = tf.nn.max_pool(x, ksize=[1, self.sequence_length-filter_size, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(x_pooled)
            print(x_pooled.shape)

        # COMBINING POOLED FEATURES
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print('concat', self.h_pool.shape)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print('reshape', self.h_pool_flat.shape)
        return self.h_pool_flat

def create_model(filter_sizes, num_filters, embed_size,sequence_length):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = EMBEDDING_SIZE,  # 词向量 长度
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'embedding_layer'
                                ),
    Expand(),
    MyConvLayer(filter_sizes, num_filters, embed_size,sequence_length),
    Dropout(0.5),
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    print(model.summary())
    return model


X_train, X_, Y_train, Y_ = train_test_split(train_data,train_labels, test_size = 0.2, random_state = 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, test_size = 0.1, random_state = 42)


lstmcnn = create_model(filter_sizes, num_filters, EMBEDDING_SIZE, MAX_SEQUENCE_LENGTH) #因为是lstm后接cnn所以输入第三维不是词向量维数而是biLSTM输出维数2*units
lstmcnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH), optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
train_gen = BatchGenerator(X_train, Y_train, input_dim=X_train.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE  )
val_gen = BatchGenerator(X_val, Y_val, input_dim=X_val.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE )
history = lstmcnn.fit_generator(generator=train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)],\
                            workers=6, use_multiprocessing=True)
lstmcnn.save_weights('./lstm_weights.h5', overwrite=True)

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['categorical_accuracy'], label='train')
plt.plot(history.history['val_categorical_accuracy'], label='test')
plt.legend()
plt.show()


# 在测试集上检测
y_pred = lstmcnn.predict(X_test)
y_pred = y_pred.argmax(axis = 1)
print('Testing Macro-F1={}'.format(f1_score(Y_test, y_pred, average='macro')))

print('Begin predicting...')
pred = lstmcnn.predict(test_data)
result= pred.argmax(axis=1)
print(result)
output = pd.DataFrame( data={'id':test_id,"label":result} )
# Use pandas to write the comma-separated output file
output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
print("Done")


