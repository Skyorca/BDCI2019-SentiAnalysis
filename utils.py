import string
import jieba as jb
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec, KeyedVectors

#参数集,window=2#
w1 = 3 #most-pos
w2 = 1.5 #more-pos,pos-pos
w3 = -1 #fou-pos,neg-pos,pos-neg,neg only
w4 = 1 #pos only, fou-neg
w5 = -0.5 #fou
w6 = -1.5 #more-neg,neg-neg
w7 = -3 #most-neg
theta = 3 #pos-neg threshold

def clean(s):
    '''
    去掉中英文标点，字母。数字,每篇文章是一个字符串
    '''
    punctuation = '！？｡ ↓  △ • 。《 》 ＂→ ＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
    exclude = list(set(string.punctuation+string.ascii_letters+string.digits+punctuation)) #合并中英文表点和英文字母以及数字
    if type(s)==str: 
        s = ''.join(ch for ch in s if ch not in exclude)
        return s
    else: return str()


def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  


def preprocess(file,cat):
    '''
    file: train/test
    '''
    #加载停用词
    stopwords = stopwordslist("./senti-dict/stop.txt")
    data = pd.read_csv(file) #id-title-content, 7340 samples
    data_id = data['id']
    data_title = data['title']
    data_content = data['content']
    print('Cleaning {} data...'.format(cat))
    data_title = data_title.apply(clean)
    data_content = data_content.apply(clean)
    print('cutting and stripping') 
    data_title = data_title.apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
    data_content = data_content.apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
    data_all = data_title+data_content
    data_all.fillna('0',inplace=True)
    out = pd.DataFrame({'id':data_id, 'data':data_all})
    out.to_csv('./middle/{}_data.csv'.format(cat),index=False)
    print('{} data Done.'.format(cat))


def preprocess_v2(file,cat):
    '''
    file: train/test
    not stripping stop words for word2vec model
    '''
    data = pd.read_csv(file) #id-title-content, 7340 samples
    data_id = data['id']
    data_title = data['title']
    data_content = data['content']
    print('Cleaning {} data...'.format(cat))
    data_title = data_title.apply(clean)
    data_content = data_content.apply(clean)
    print('cutting and stripping') 
    data_title = data_title.apply(lambda x: " ".join([w for w in list(jb.cut(x))]))
    data_content = data_content.apply(lambda x: " ".join([w for w in list(jb.cut(x))]))
    data_all = data_title+data_content
    data_all.fillna('0',inplace=True)
    out = pd.DataFrame({'id':data_id, 'data':data_all})
    out.to_csv('./middle/{}_data_v2.csv'.format(cat),index=False)
    print('{} data Done.'.format(cat))


def mark(content):
    '''
    加载字典，并对文档打分
    输入： content字符串,已经把标题和内容合并并且分好词，并没有对标题和内容分开调用。
    输出：{0,1,2}
    ''' 
    if type(content)!=str: return 1
    #加载字典
    pos_ = pd.read_csv('senti-dict/pos.txt',header=None)
    pos = list(pos_[0].values) #必须要有默认的列索引为数字0
    neg_ = pd.read_csv('senti-dict/neg.txt',header=None)
    neg = list(neg_[0].values)
    more_ = pd.read_csv('senti-dict/more.txt',header=None)
    more = list(more_[0].values)
    most_ = pd.read_csv('senti-dict/most.txt',header=None)
    most = list(most_[0].values)
    fouding_ = pd.read_csv('senti-dict/fouding.txt',header=None)
    fouding = list(fouding_[0].values)

    #加载已评分的boson词典
   # bsn = pd.read_csv('senti-dict/boson.csv',index_col=['word'])
    #bsn_dict=bsn.to_dict()['score']

    value = 0
    words = content.split()
    l = len(words)
    for i in range(l):
        if words[i] in pos:
            pre = words[i-1]
            if pre in most: 
                value += w1 
                #print('{}most-pos'.format(i+1))
            elif pre in more or pre in pos: 
                value += w2
                #print('{}more-pos'.format(i+1))
            elif pre in fouding or pre in neg : 
                value += w3
                #print('{}fou-pos/neg-pos'.format(i+1))    
            else: 
                value += w4
                #print('{}pos-only'.format(i+1))

        elif words[i] in fouding: 
            value += w5
            #print('{}fou'.format(i+1))

        elif words[i] in neg:
            pre = words[i-1]
            if pre in most: 
                value += w7
                #print('{}most-neg'.format(i+1))
            elif pre in more or pre in neg: 
                value += w6 #add 1 case neg-neg
                # print('{}more-neg/neg-neg'.format(i+1))
            elif pre in fouding: 
                value += w4
                # print('{}fou-neg'.format(i+1))
            elif pre in pos:
                value += w3
                # print('{}pos-neg'.format(i+1))
            else: 
                value += w3
                # print('{}neg-only'.format(i+1))
        else: value += 0 #本单词对极性无贡献
        #print(value)
       # if words[i] in bsn_dict: value += bsn_dict[words[i]]

    if value>=theta: 
        #print('label = 0')
        return 0
    elif value>-theta and value<theta: 
        #print('label = 1')
        return 1 #允许微扰
    else: 
        #print('label = 2')
        return  2



def to_wordbag(x):
    '''
    为词袋模型使用
    输入：清洗好的字符串
    输出：分好词的字符串
    '''
    words = jb.lcut(x)
    s = str()
    #stop = list(pd.read_csv('senti-dict/stop.txt',header=None).values)
    i = 0
    for w in words:
        #if i<8000 and w not in stop:
        if i<10000:
            s += w
            s += ' '
            i += 1
    return s


#################################以下是深度学习模型专用辅助函数######################################

# 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
def tokenizer(texts, word_index):
    MAX_SEQUENCE_LENGTH = 3000
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


def preprocess_v3():
    '''
    textRNN textCNN RCNN等模型前期处理函数
    主要是给模型搭配预训练词向量, 第一层embedding需要
    返回train_data, train_label, test_id, test_data, embed_matrix
    '''
    ## 第一步 加载预训练的词向量
    Word2VecModel = KeyedVectors.load('all_word2vec')
    #构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]# 存储 所有的 词语
    word_index = {" ": 0}# 初始化 `词-词索引` ，后期 tokenize 语料库就是用该词典。
    word_vector = {} # 初始化`词-词向量`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如256。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    #填充上述的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：词索引
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
    #print(embeddings_matrix.shape)

    ## 第二步 把每个文档转换为词索引句阵，每个单词以索引表示
    df = pd.read_csv('./middle/train_data.csv')
    df['data'].fillna('0',inplace=True)
    train_data = list(df['data'].values) #每个元素是分割好单词的字符串，这样的字符串也可以迭代
    train_data = tokenizer(train_data, word_index)
    train_labels = pd.read_csv('./data/Train/Train_DataSet_Label.csv')['label'].values

    df2 = pd.read_csv('./middle/test_data.csv')
    df2['data'].fillna('0',inplace=True) #填充缺失值很重要！
    test_data = list(df2['data'].values)
    test_data = tokenizer(test_data, word_index)
    test_id = df2['id']
    return (train_data, train_labels, test_id, test_data, embeddings_matrix)
    

 



  
