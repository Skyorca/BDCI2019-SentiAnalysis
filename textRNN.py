import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from batchgenerater import BatchGenerator
from utils import preprocess_v3

## hparams
EMBEDDING_LEN = 256
LSTM_UNITS1 = 100
LSTM_UNITS2 = 100
DIM1=8
EPOCHS = 20
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH=3000 #increase, acc起点低，上升速度变慢，收敛结果也变差？注意这里修改过,在utils.py/preprocess_v3
LABEL_SMOOTH = 0.3
reg = tf.keras.regularizers.l2(l=0.001)
n_classes = 3


## 第一步 加载预处理好的各项数据
train_data, train_labels, test_id, test_data, embeddings_matrix = preprocess_v3()

##第二步 设计模型与训练
def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = EMBEDDING_LEN,  # 词向量 长度
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'embedding_layer'
                                ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS1, \
                                                       kernel_regularizer=reg, \
                                                       recurrent_regularizer=reg, \
                                                       activity_regularizer=reg, \
                                                       dropout=0.5, \
                                                       recurrent_dropout=0.5, return_sequences=True)
                                 ), #Return the full sequences of successive outputs for each timestep (a 3D tensor of shape (batch_size, timesteps, output_features)).
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS1, \
                                                       kernel_regularizer=reg, \
                                                       recurrent_regularizer=reg, \
                                                       activity_regularizer=reg, \
                                                       dropout=0.5, \
                                                       recurrent_dropout=0.5, return_sequences=False)
                                 ), #Return only the last output for each input sequence (a 2D tensor of shape (batch_size, output_features)).
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    print(model.summary())
    return model

X_train, X_, Y_train, Y_ = train_test_split(train_data,train_labels, test_size = 0.2, random_state = 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, test_size = 0.1, random_state = 42)


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

print('Begin predicting...')
pred = lstm.predict(test_data)
result= pred.argmax(axis=1)
print(result)
output = pd.DataFrame( data={'id':test_id,"label":result} )
# Use pandas to write the comma-separated output file
output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
print("Done")


