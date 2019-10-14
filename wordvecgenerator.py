from gensim.models import word2vec, KeyedVectors
import pandas as pd

## 训练自己的词向量，并保存。
def trainWord2Vec(file, cat):
    '输入是把middle/train_csv test_csv的内容复制到txt里面，一行是一个文档（句子）'
    sentences =  word2vec.LineSentence(file) # 读取txt文件
    model = word2vec.Word2Vec(sentences, size=256, window=5, iter=15) # 训练模型
    model.wv.save('{}_word2vec'.format(cat))


def testMyWord2Vec(cat):
    # 读取自己的词向量，并简单测试一下 效果。
    inp = '{}_word2vec'.format(cat)  # 读取词向量
    model = KeyedVectors.load(inp)

    print('空间的词向量（256维）:',model['坚定'])
    print('打印与空间最相近的20个词语：',model.most_similar('坚定', topn=20))


if __name__ == '__main__':
    #trainWord2Vec('./middle/train.txt','train')
    #trainWord2Vec('./middle/test.txt','test')
    testMyWord2Vec('train')
    