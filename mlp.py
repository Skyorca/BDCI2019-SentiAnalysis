import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from batchgenerater import BatchGenerator

#preprocess("./data/Train/Train_DataSet.csv",'train')
#preprocess_test('./data/Test_DataSet.csv','Test')

EPOCHS = 20
BATCH_SIZE = 32
MAX_NUM_WORDS=10000 #increase, acc起点低，上升速度变慢，收敛结果也变差？
LABEL_SMOOTH = 0.3
DIM1 = 2048
DIM2 = 512
DIM3 = 128
DIM4 = 16
reg = tf.keras.regularizers.l2(l=0.01)
n_classes = 3


df = pd.read_csv('middle/train_data.csv')
df['data'].fillna('0',inplace=True)
train_data = df['data'].values
train_labels = pd.read_csv('./data/Train/Train_DataSet_Label.csv')['label'].values
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=False)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')


# 8-1-1 train/val/test
#X: (sample_num, feature), y:(sample,) which is not one-hot encoder
X_train, X_, Y_train, Y_ = train_test_split(data,train_labels, test_size = 0.2, random_state = 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, test_size = 0.1, random_state = 42)


def create_model(data, labels):
    model = Sequential()
    model.add(Dense(DIM1, input_shape=(data.shape[1],), activation='relu',kernel_regularizer = None))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(DIM2, activation='relu',kernel_regularizer = None))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(DIM3, activation='relu',kernel_regularizer = None))
    model.add(Dropout(0.5))
    model.add(Dense(DIM4, activation='relu',kernel_regularizer = None))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH), optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    print(model.summary())
    return model


mlp = create_model(X_train, Y_train)
'''
history = mlp.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, class_weight=class_weights, \
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
                    )
'''

train_gen = BatchGenerator(X_train, Y_train, input_dim=X_train.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE  )
val_gen = BatchGenerator(X_val, Y_val, input_dim=X_val.shape[1], n_classes=n_classes, batch_size=BATCH_SIZE )
history = mlp.fit_generator(generator=train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)],\
                            workers=6, use_multiprocessing=True)

mlp.save_weights('./mlp_weights.h5', overwrite=True)

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
y_pred = mlp.predict(X_test)
y_pred = y_pred.argmax(axis = 1)
print('Testing Macro-F1={}'.format(f1_score(Y_test, y_pred, average='macro')))

'''
#生成混淆矩阵
conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=['0','1','2'], yticklabels=['0','1','2'])
plt.ylabel('u/实际结果',fontsize=18)
plt.xlabel('u/预测结果',fontsize=18)
plt.show()
'''
# predict #
print('Begin predicting...')
df2 = pd.read_csv('middle/test_data.csv')
df2['data'].fillna('0',inplace=True) #填充缺失值很重要！
test_data_ = df2['data'].values
test_id = df2['id']
tokenizer.fit_on_texts(test_data_)
test_sequences = tokenizer.texts_to_sequences(test_data_)
word_index = tokenizer.word_index
print('Test: Found %s unique tokens.' % len(word_index))
test_data = tokenizer.sequences_to_matrix(test_sequences, mode='tfidf')
pred = mlp.predict(test_data)
result= pred.argmax(axis=1)
print(result)
output = pd.DataFrame( data={'id':test_id,"label":result} )
# Use pandas to write the comma-separated output file
output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
print("Done")


'''
log:
没加正则化的效果比正则化好。

如果给1施加惩罚？ class_weight$

label smoothing down$
'''