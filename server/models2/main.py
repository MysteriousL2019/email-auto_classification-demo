import math
import re
import pandas as pd
import codecs
import jieba
import logging
import os
import pickle

base_path = os.path.abspath(os.path.dirname(__file__))
logging.info("base path :{}".format(base_path))

def load_formatted_data():
    
    # 加载数据集
    base_path = os.path.abspath(os.path.dirname(__file__))
    feature_vect_path = os.path.join(base_path, 'index')

    index = pd.read_csv(feature_vect_path, sep=' ', names=['spam', 'path'])
    index.spam = index.spam.apply(lambda x: 1 if x == 'spam' else 0)
    index.path = index.path.apply(lambda x: x[1:])
    return index


def load_stop_word():
    print(base_path)
    feature_vect_path = os.path.join(base_path, 'stop')
    with codecs.open(feature_vect_path, "r", encoding="utf_8") as f:
        lines = f.readlines()
    _stop_words = [i.strip() for i in lines]
    print(_stop_words)
    return _stop_words


def get_mail_content(path):
    """
    遍历得到每封邮件的词汇字符串
    :param path: 邮件路径
    :return:(Str)content
    """
    path=os.path.join(base_path,path)
    with codecs.open(path, "r", encoding="gbk", errors="ignore") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[i] == '\n':
            # 去除第一个空行，即在第一个空行之前的邮件协议内容全部舍弃
            lines = lines[i:]
            break
    content = ''.join(''.join(lines).strip().split())
    # print(content)
    return content


def create_word_dict(content, stop_words_list):
    """
    依据邮件的词汇字符串统计词汇出现记录，依据停止词列表除去某些词语
    :param content: 邮件的词汇字符串
    :param stop_words_list:停止词列表
    :return:(Dict)word_dict
    """
    word_list = []
    word_dict = {}
    # word_dict key:word, value:1
    content = re.findall(u"[\u4e00-\u9fa5]", content)
    content = ''.join(content)
    word_list_temp = jieba.cut(content)
    for word in word_list_temp:
        if word != '' and word not in stop_words_list:
            word_list.append(word)
    for word in word_list:
        word_dict[word] = 1
    return word_dict


def train_dataset(dataset_to_train):
    """
    对数据集进行训练, 统计训练集中某个词在普通邮件和垃圾邮件中的出现次数
    :param dataset_to_train: 将要用来训练的数据集
    :return:Tuple(词汇出现次数字典_train_word_dict, 垃圾邮件总数spam_count, 正常邮件总数ham_count)
    """
    _train_word_dict = {}
    # train_word_dict内容，训练集中某个词在普通邮件和垃圾邮件中的出现次数
    for word_dict, spam in zip(dataset_to_train.word_dict, dataset_to_train.spam):
        # word_dict某封信的词汇表 spam某封信的状态
        for word in word_dict:
            # 对每封信的每个词在该邮件分类进行出现记录 出现过为则记录数加1 未出现为0
            _train_word_dict.setdefault(word, {0: 0, 1: 0})
            _train_word_dict[word][spam] += 1
    ham_count = dataset_to_train.spam.value_counts()[0]
    spam_count = dataset_to_train.spam.value_counts()[1]
    return _train_word_dict, spam_count, ham_count


def predict_dataset(_train_word_dict, _spam_count, _ham_count, data):
    """
    测试算法
    :param _train_word_dict:词汇出现次数字典
    :param _spam_count:垃圾邮件总数
    :param _ham_count:正常邮件总数
    :param data:测试集
    :return:
    """
    total_count = _ham_count + _spam_count
    word_dict = data['word_dict']

    # 先验概率 已经取了对数
    ham_probability = math.log(float(_ham_count) / total_count)
    spam_probability = math.log(float(_spam_count) / total_count)

    for word in word_dict:
        word = word.strip()
        _train_word_dict.setdefault(word, {0: 0, 1: 0})

        # 求联合概率密度 += log
        # 拉普拉斯平滑
        word_occurs_counts_ham = _train_word_dict[word][0]
        # 出现过这个词的信件数 / 垃圾邮件数
        ham_probability += math.log((float(word_occurs_counts_ham) + 1) / _ham_count + 2)

        word_occurs_counts_spam = _train_word_dict[word][1]
        # 出现过这个词的信件数 / 普通邮件数
        spam_probability += math.log((float(word_occurs_counts_spam) + 1) / _spam_count + 2)

    if spam_probability > ham_probability:
        is_spam = 1
    else:
        is_spam = 0

    # 返回预测正确状态
    if is_spam == data['spam']:
        return 1
    else:
        return 0

"""
下面都是将训练过程中的结果——train的中文词语的词典、spam的个数、ham的个数、test集合的中文词语的词典保存为.pkl文件
方便predict的时候直接调用
"""
def save_train_word_dicts(train_word_dict):
    feature_vect_path = os.path.join(base_path, 'train_word_dict.pkl')
    with codecs.open(feature_vect_path, 'wb') as f:
        pickle.dump(train_word_dict, f)
def read_word():
    feature_vect_path = os.path.join(base_path, 'train_word_dict.pkl')
    df11 = pd.read_pickle(feature_vect_path)
    print(type(df11))
    return df11

def save_spam_count(spam_count):
    feature_vect_path = os.path.join(base_path, 'train_spam_count.pkl')
    with codecs.open(feature_vect_path, 'wb') as f:
        pickle.dump(spam_count, f)

def read_spam():
    feature_vect_path = os.path.join(base_path, 'train_spam_count.pkl')
    df11 = pd.read_pickle(feature_vect_path)
    print(type(df11))
    return df11

def save_ham_count(ham_count):
    feature_vect_path = os.path.join(base_path, 'train_ham_count.pkl')
    with codecs.open(feature_vect_path, 'wb') as f:
        pickle.dump(ham_count, f)
        
def read_ham():
    feature_vect_path = os.path.join(base_path, 'train_ham_count.pkl')

    df11 = pd.read_pickle(feature_vect_path)
    print(type(df11))
    return df11

def save_test_set(test_set):
    feature_vect_path = os.path.join(base_path, 'train_test_set.pkl')
    with codecs.open(feature_vect_path, 'wb') as f:
        pickle.dump(test_set, f)
def read_test_set():
    feature_vect_path = os.path.join(base_path, 'train_test_set.pkl')
    df11 = pd.read_pickle(feature_vect_path)
    print(type(df11))
    return df11

def train():
    #训练过程，首次运行之时选用这个，将数据集自动化分，之后进行分词训练，并将分词结果，保存
    index_list = load_formatted_data()
    stop_words = load_stop_word()
    # get_mail_content(index_list.path[0])
    index_list['content'] = index_list.path.apply(lambda x: get_mail_content(x))
    index_list['word_dict'] = index_list.content.apply(lambda x: create_word_dict(x, stop_words))

    train_set = index_list.loc[:len(index_list) * 0.8]
    test_set = index_list.loc[len(index_list) * 0.8:]

    train_word_dict, spam_count, ham_count = train_dataset(train_set)

    save_train_word_dicts(train_word_dict)
    save_spam_count(spam_count)
    save_ham_count(ham_count)
    save_test_set(test_set)


if __name__ == '__main__':
    # train()
    train_word_dict=read_word()
    spam_count=read_spam()
    ham_count=read_ham()
    test_set=read_test_set()
    test_mails_predict = test_set.apply(
        lambda x: predict_dataset(train_word_dict, spam_count, ham_count, x), axis=1)
    corr_count = 0
    false_count = 0
    for i in test_mails_predict.values.tolist():
        if i == 1:
            corr_count += 1
        if i == 0:
            false_count += 1

    print("测试集样本总数", (corr_count + false_count))
    print("正确预计个数", corr_count)
    print("错误预测个数", false_count)
    result = float(corr_count / (corr_count + false_count))
    print('预测准确率：', result)
