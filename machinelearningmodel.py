import pandas as pd
import numpy as np
from utils import *
from sklearn.metrics import f1_score,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def pipeline():
    '''
    词袋的输入：元素是分词好的字符串的列表/设置属性为str的csv的一列，先转成values
    '''
    print('Training TF-IDF...')
    train_data = pd.read_csv('./middle/train_data.csv')['data']
    vectorizer = CountVectorizer(analyzer = "word",   \
                                tokenizer = None,    \
                                preprocessor = None, \
                                stop_words = None,   \
                                max_features = 120000, \
                                ngram_range=(1,2),\
                                min_df=3, \
                                max_df=0.9,\
                               # token_pattern=u"(?u)\\b\\w+\\b"
                                ) #如果不设置max_feature的话，train和test的特征数就不一致.怎么处理中文词太多？token_pattern实现单字+词模式
    
    train_data_features = vectorizer.fit_transform(train_data.values.astype('str')) #sparse
    transformer = TfidfTransformer(sublinear_tf=1)
    train_tfidf = transformer.fit_transform(train_data_features) # this is sparse
    #print(vectorizer.vocabulary_)   单词-编号的词典
    #print(vectorizer.get_feature_names()) 所有的单词
    train_lbl = pd.read_csv('./data/Train/Train_DataSet_Label.csv')['label'].values
    x_train,x_test,y_train,y_test=train_test_split(train_tfidf, train_lbl,test_size=0.2,random_state=0)
    print("Splitting Done")

    #rf
    #random_forest(x_train, y_train, x_test, y_test)
    #svm   
    #grid=svm(train_tfidf, train_lbl)
    #xgboost
    #grid=xgbclassifier(train_tfidf, train_lbl)
    #overall
    #clf = BagofClf(train_tfidf, train_lbl)

    #聚类可视化
    KMeansVisual(train_tfidf, train_lbl)

    #xgb itself
    xgbclf= xgb.XGBClassifier(max_depth=10, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=.01, reg_alpha=4, objective='multi:softmax')
    xgbclf.fit(x_train, y_train)
    pred = xgbclf.predict(x_test)
    score = f1_score(y_test, pred, average='macro')
    print('our test Macro-F1=',score)
    print("acc=",accuracy_score(y_test,pred))


    """
    test_id = pd.read_csv("Test_DataSet.csv")['id']
    print('Doing predictions on real test-data')
    test_data = pd.read_csv('middle/test_data.csv')['data']
    test_data_features = vectorizer.fit_transform(test_data.values.astype('str')) #sparse
    result = grid.predict(test_data_features)
    output = pd.DataFrame( data={'id':test_id,"sentiment":result} )
    # Use pandas to write the comma-separated output file
    output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
    print("Done")
    """



if __name__ == "__main__":
    #preprocess("./data/Train/Train_DataSet.csv",'train')
    #preprocess_test('./data/Test_DataSet.csv','Test')
    pipeline()
    

