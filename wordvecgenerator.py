from gensim.models import word2vec, KeyedVectors
import pandas as pd

#train和test形成一个大文档训练词向量

## 训练自己的词向量，并保存。
def trainWord2Vec(file):
    '输入要首先是把middle/train_csv test_csv的data内容分别复制到txt里面，一行是一个文档（句子）'
    sentences =  word2vec.LineSentence(file) # 读取txt文件
    model = word2vec.Word2Vec(sentences, size=128, window=5, iter=15) # 训练模型
    model.wv.save('all_word2vec_128')


def testMyWord2Vec( word):
    # 读取自己的词向量，并简单测试一下 效果。
    inp = 'all_word2vec'  # 读取词向量
    model = KeyedVectors.load(inp)

    print('{}的词向量（128维）:'.format(word), model['{}'.format(word)])
    print('打印与{}最相近的20个词语：'.format(word), model.most_similar('{}'.format(word), topn=20))


if __name__ == '__main__':
    #trainWord2Vec('./middle/train_and_test_data.txt')
    testMyWord2Vec('数据')
