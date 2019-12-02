'''
For GPU/TPU use, generate batch whose  number could be divided by batch_size on-the-fly
borrow from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly with modification
'''

import numpy as np
import tensorflow as tf

class BatchGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, input_dim, n_classes, batch_size=32, n_channels=0, shuffle=True):
        'Initialization, here ID is sample matrix itself rather than real ids'
        self.dim = input_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.y = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels != 0: X = np.empty((self.batch_size, self.dim, self.n_channels))
        else: X = np.empty((self.batch_size, self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = ID

        #在这里转换成One-hot变量
        return X, tf.keras.utils.to_categorical(np.asarray(self.y), num_classes=self.n_classes)



'''
输入：特征矩阵，train/val
     y需要是int，之后再变成独热类别编码

使用格式：

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
'''