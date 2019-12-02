import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import f1_score,accuracy_score
from batchgenerater import BatchGenerator
from utils import preprocess_v3
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

## hparams
MAX_SEQUENCE_LENGTH = 3000
EMBEDDING_LEN = 256
LSTM_UNITS1 = 64
LSTM_UNITS2 = 64
DIM1=32
EPOCHS = 20
BATCH_SIZE = 32
#MAX_NUM_WORDS=10000 #increase, acc起点低，上升速度变慢，收敛结果也变差？
LABEL_SMOOTH = 0.1
n_classes = 3

class attention(tf.keras.Model):
    """
    利用Attention机制得到句子的向量表示
    """
    def __init__(self,LSTM_DIM2,dropout,sequenceLength):
        super().__init__()
        self.hiddenSize = LSTM_DIM2 # 获得最后一层LSTM的神经元数量 
        self.dropout = dropout
        self.sequenceLength=sequenceLength
        self.initializer = tf.random_normal_initializer(stddev=0.1) 
        # 初始化一个权重向量，是可训练的参数
        self.W = self.add_weight(name='W', shape=[self.hiddenSize], initializer=self.initializer )       
        
    def call(self,input):
        outputs=tf.split(input, 2, -1)
        H=outputs[0]+outputs[1]
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, self.hiddenSize]), tf.reshape(self.W, [-1, 1]))       
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.sequenceLength])        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.sequenceLength, 1]))       
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, self.hiddenSize])       
        sentenceRepren = tf.tanh(sequeezeR)        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropout)
        
        return output

def main():
    train_data, train_labels, test_id, test_data, embeddings_matrix = preprocess_v3()
    ##第三步 设计模型与训练
    def create_model():
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                    output_dim = EMBEDDING_LEN,  # 词向量 长度
                                    weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                    input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                    trainable=True,  # 是否在 训练的过程中 更新词向量
                                    name= 'embedding_layer'
                                    ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS1,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS2,return_sequences=True)),
        attention(LSTM_UNITS1,0.5,MAX_SEQUENCE_LENGTH),
        tf.keras.layers.Dense(DIM1, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        print(model.summary())
        return model

    X_train, X_, Y_train, Y_ = train_test_split(train_data,train_labels, test_size = 0.1, random_state = 42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, test_size = 0.6, random_state = 32)

    if tf.test.is_gpu_available():
        with tf.device('GPU:0'):
            lstm = create_model()
            lstm.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH), optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
            train_gen = BatchGenerator(X_train, Y_train, input_dim=X_train.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE  )
            val_gen = BatchGenerator(X_val, Y_val, input_dim=X_val.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE )
            history = lstm.fit_generator(generator=train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)],\
                                        workers=6, use_multiprocessing=True)

    lstm.save_weights('./lstm_weights.h5', overwrite=True)

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
    y_pred = lstm.predict(X_test)
    y_pred = y_pred.argmax(axis = 1)
    print('Testing Macro-F1={}'.format(f1_score(Y_test, y_pred, average='macro')))
    print('Testing acc={}'.format(accuracy_score(Y_test, y_pred)))
    

if __name__ == "__main__":
    main()

