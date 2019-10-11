import string
import jieba as jb
import pandas as pd
import re

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
    punctuation = '！？｡ ↓ • 。《 》 ＂→ ＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
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


def mark(content):
    '''
    加载字典，并对文档打分
    输入： content字符串,已经把标题和内容合并并且分好词，并没有对标题和内容分开调用。
    输出：{0,1,2}
    ''' 
    assert(type(content)==str)
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
        print('label = 0')
        return 0
    elif value>-theta and value<theta: 
        print('label = 1')
        return 1 #允许微扰
    else: 
        print('label = 2')
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


 

 



  
