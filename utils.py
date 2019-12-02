import string
import jieba as jb
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec, KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


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

############################################# 机器学习模型的函数 ################################################
def BagofClf(features, labels):
    """
    整合多个线性分类器，选最优三个投票决定。分类器的超参记得搜索
    """
    rf = RandomForestClassifier(n_estimators = 70, oob_score=True)
    svc = LinearSVC(class_weight="balanced") #assume this is important
    nb  = MultinomialNB()
    lr =  LogisticRegression(random_state=0)
    models = [rf, svc, nb, lr]
    """
    print("Doing model selection...WAIT")
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    """
    print("Doing majority voting...WAIT")
    x_train,x_test,y_train,y_test=train_test_split(features, labels,test_size=0.1,random_state=44)
    _rf = rf.fit(x_train, y_train )
    _svm = svc.fit(x_train, y_train)
    _lr  = lr.fit(x_train, y_train)
    pred1 = _rf.predict(x_test)
    pred2 = _svm.predict(x_test)
    pred3 = _lr.predict(x_test)
    def __majority_element(a):
        c = Counter(a)
        value, count = c.most_common()[0]
        if count > 1:
            return value
        else:
            return a[0]

    merged_predictions = [[s[0],s[1],s[2]] for s in zip(pred1,pred2,pred3)]
    majority_prediction = [__majority_element(p) for p in merged_predictions]
    f1score = f1_score(y_test, majority_prediction, average='macro')
    acc     = accuracy_score(y_test, majority_prediction)
    print("Macro F1-score=",f1score)
    print('acc=',acc)


def random_forest(x_train, y_train, x_test, y_test):
    '''
    使用随机森林模型,以后也要带上参数搜索
    '''
    print('Using Random-Forest')
    #训练随机森林,如何达到较好效果
    forest = RandomForestClassifier(n_estimators = 70, oob_score=True) 
    forest = forest.fit(x_train, y_train )
    print ("random forest is Done")
    y_pred = forest.predict(x_test)
    print("our test predictions Done") 
    score = f1_score(y_test, y_pred, average='macro')
    print('our test Macro-F1=',score)

def svm(x_train, y_train):
    '''
    此时搜索超参会带来过拟合？
    '''
    print('Using SVM')
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid={"degree":[2,3,4]}, scoring='f1_macro', cv=4)
    grid.fit(x_train, y_train)
    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
    print ("svm is Done")
    return grid


def xgbclassifier(x_train, y_train):
    '''
    网格搜索，则不需要额外划分测试集训练集
    '''
    print('Using XGBoost')
    parameters = {
              'max_depth': [5, 10, 15, 20, 25],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'n_estimators': [500, 1000, 2000, 3000, 5000],
              'min_child_weight': [0, 2, 5, 10, 20],
              'max_delta_step': [0, 0.2, 0.6, 1, 2],
              'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
              'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]}

    xlf = xgb.XGBClassifier(max_depth=10,
                learning_rate=0.01,
                n_estimators=2000,
                silent=False,
                objective='multi:softmax',
                n_jobs=4,
                num_class=3,
                gamma=0,
                min_child_weight=1,
                max_delta_step=0,
                subsample=0.85,
                colsample_bytree=0.7,
                colsample_bylevel=1,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                seed=1440,
                missing=None)
                
    # 有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='f1_macro', cv=4)
    gsearch.fit(x_train, y_train)

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return gsearch


def KMeansVisual(features, labels):
    """KMeans聚类可视化"""
    pca = TruncatedSVD()
    pca.fit(features)
    new_feats = pca.transform(features)
    clf = KMeans(n_clusters=3)
    clf.fit(new_feats)
    cents = clf.cluster_centers_#质心
    _labels = clf.labels_#样本点被分配到的簇的索引
    #画出聚类结果，每一类用一种颜色
    colors = ['b','g','r']
    plt.figure(figsize=(60,60))
    for i in range(3):
        index = np.nonzero(_labels==i)[0]
        x0 = new_feats[index,0]
        x1 = new_feats[index,1]
        y_i = labels[index]
        for j in range(len(x0)):
            plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],fontdict={'weight': 'bold', 'size': 20})
            plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=12)
    plt.title("3-KMeans")
    plt.axis([-0.1,0.5,-0.5,0.5])
    plt.show()






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
    #word_vector = {} # 初始化`词-词向量`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如256。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    #填充上述的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：词索引
        #word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵，因为词索引是从1开始的所以也要从第二行开始填充
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
    

 



  
