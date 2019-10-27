import pandas as pd
import numpy as np
from utils import *
from sklearn.metrics import f1_score


#调用标准预处理函数
#preprocess('Train/Train_DataSet.csv','train')
#preprocess('Test_DataSet.csv','test')

#打分与评级
print('Training on train data...')
train_data = pd.read_csv('./middle/train_data.csv')['data']
train_label = pd.read_csv('./Train/Train_DataSet_Label.csv')['label']
pred_label = train_data.apply(mark)
print('Macro-F1={}'.format(f1_score(train_label,pred_label,average='macro')))
test = pd.read_csv('./middle/test_data.csv')
test_id = test['id']
test_data = test['data']
test_label = test_data.apply(mark)
out = pd.DataFrame({'id':test_id, 'label':test_label})
out.to_csv('中国抖学院-奶茶技术研究所-final.csv',index=False)
print('Senti-Dict Model Done.')


