import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec, KeyedVectors
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from batchgenerater import BatchGenerator
from utils import preprocess_v3
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

## hparams
MAX_SEQUENCE_LENGTH = 3000
EMBEDDING_LEN = 256
LSTM_UNITS = 64
DIM1=32
EPOCHS = 20
BATCH_SIZE = 32
#MAX_NUM_WORDS=10000 #increase, acc起点低，上升速度变慢，收敛结果也变差？
LABEL_SMOOTH = 0.1
reg = tf.keras.regularizers.l2(l=0.01)
n_classes = 3


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
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)),
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
    
    """
    print('Begin predicting...')
    pred = lstm.predict(test_data)
    result= pred.argmax(axis=1)
    print(result)
    output = pd.DataFrame( data={'id':test_id,"label":result} )
    # Use pandas to write the comma-separated output file
    output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
    print("Done")
    """

if __name__ == "__main__":
    main()

