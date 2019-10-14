%tensorflow_version 2.x
%matplotlib inline
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
from batchgenerater import BatchGenerator

## hparams
MAX_SEQUENCE_LENGTH = 3000
EMBEDDING_LEN = 256
LSTM_UNITS = 64
EPOCHS = 20
BATCH_SIZE = 32
MAX_NUM_WORDS=10000 #increase, acc起点低，上升速度变慢，收敛结果也变差？
LABEL_SMOOTH = 0.3
reg = tf.keras.regularizers.l2(l=0.01)
n_classes = 3
DIM1=32

## 第一步 加载预训练的词向量

Word2VecModel = KeyedVectors.load('all_word2vec')
#构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]# 存储 所有的 词语
word_index = {" ": 0}# 初始化 `词-词索引` ，后期 tokenize 语料库就是用该词典。
word_vector = {} # 初始化`词-词向量`字典
# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如256。
embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

#填充 上述 的字典 和 大矩阵
for i in range(len(vocab_list)):
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1 # 词语：词索引
    word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
    embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
#print(embeddings_matrix.shape)


## 第二步 把每个文档转换为词索引句阵，每个单词以索引表示
# 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
def tokenizer(texts, word_index):
    data = []
    for doc in texts:
        new_txt = []
        for word in doc:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
            
        data.append(new_txt)

    texts = pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts

df = pd.read_csv('train_data.csv')
df['data'].fillna('0',inplace=True)
train_data = list(df['data'].values) #每个元素是分割好单词的字符串，这样的字符串也可以迭代
train_data = tokenizer(train_data, word_index)
train_labels = pd.read_csv('./Train/Train_DataSet_Label.csv')['label'].values

##第三步 设计模型与训练
def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = EMBEDDING_LEN,  # 词向量 长度
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'embedding_layer'
                                ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)),
    tf.keras.layers.Dense(DIM1, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    print(model.summary())
    return model

X_train, X_, Y_train, Y_ = train_test_split(train_data,train_labels, test_size = 0.2, random_state = 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, test_size = 0.1, random_state = 42)

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

print('Begin predicting...')
pred = lstm.predict(test_data)
result= pred.argmax(axis=1)
print(result)
output = pd.DataFrame( data={'id':test_id,"label":result} )
# Use pandas to write the comma-separated output file
output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
print("Done")


