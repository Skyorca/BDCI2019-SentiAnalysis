import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight

EPOCHS = 20
BATCH_SIZE = 128
MAX_NUM_WORDS=10000
DIM1 = 2048
DIM2 = 512
DIM3 = 128
DIM4 = 16
reg = tf.keras.regularizers.l2(l=0.01)

df = pd.read_csv('middle/train_data.csv')
df['data'].fillna('0',inplace=True) #填充缺失值很重要！
train_data = df['data'].values
train_labels = pd.read_csv('Train/Train_DataSet_Label.csv')['label'].values
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=False)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')
labels = to_categorical(np.asarray(train_labels),num_classes=3)


#X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size = 0.3, random_state = 42)



model = Sequential()
model.add(Dense(DIM1, input_shape=(data.shape[1],), activation='relu',kernel_regularizer = None))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(DIM2, activation='relu',kernel_regularizer = None))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(DIM3, activation='relu',kernel_regularizer = None))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(DIM4, activation='relu',kernel_regularizer = None))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
print(model.summary())

 
history = model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=0.1, class_weight=class_weights, \
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
                    )
model.save_weights('./mlp_weights.h5', overwrite=True)

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


'''
# 在测试集上检测
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis = 1)
Y_test = Y_test.argmax(axis = 1)

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
pred = model.predict(test_data)
result= pred.argmax(axis=1)
print(result)
output = pd.DataFrame( data={'id':test_id,"label":result} )
# Use pandas to write the comma-separated output file
output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
print("Done")


'''
log:
没加正则化的效果比正则化好。

如果给1施加惩罚？
'''