import pandas as pd
import numpy as np
from utils import *
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb



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
    grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, scoring='f1_macro', cv=4)
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



def train_and_predict():
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
    train_tfidf = transformer.fit_transform(train_data_features)
    #print(vectorizer.vocabulary_)   单词-编号的词典
    #print(vectorizer.get_feature_names()) 所有的单词
    train_lbl = pd.read_csv('./data/Train/Train_DataSet_Label.csv')['label'].values
    x_train,x_test,y_train,y_test=train_test_split(train_tfidf, train_lbl,test_size=0.3,random_state=0)
    print("Splitting Done")

    #rf
    #random_forest(x_train, y_train, x_test, y_test)
    #svm
    #grid=svm(train_tfidf, train_lbl)
    #xgboost
    grid=xgbclassifier(train_tfidf, train_lbl)

    #预测并输出
    test_id = pd.read_csv("Test_DataSet.csv")['id']
    print('Doing predictions on real test-data')
    test_data = pd.read_csv('middle/test_data.csv')['data']
    test_data_features = vectorizer.fit_transform(test_data.values.astype('str')) #sparse
    result = grid.predict(test_data_features)
    output = pd.DataFrame( data={'id':test_id,"sentiment":result} )
    # Use pandas to write the comma-separated output file
    output.to_csv("中国抖学院-奶茶技术研究所-final.csv", index=False)
    print("Done")






if __name__ == "__main__":
    #preprocess("./data/Train/Train_DataSet.csv",'train')
    #preprocess_test('./data/Test_DataSet.csv','Test')
    train_and_predict()
    

