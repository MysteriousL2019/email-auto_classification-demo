import jieba
from models2.main import *
# from main import * #单独调试要这样
import matplotlib.pyplot as plt
import pandas as pd
import os

def plotter():
    df = pd.read_table(os.path.join(base_path, 'index'),
                sep='\t',
                header=None,
                names=['label', 'message'])
    print(type(df))
    print(df.info())
    print(df['label'])
    df_np = df['label'].to_numpy()
    print(type(df_np[1]))

    cnt_ham = 0
    cnt_spam = 0
    X = ['ham', 'spam']
    for i in df_np:
        if 'ham' in i:
            cnt_ham += 1
        elif 'spam' in i:
            cnt_spam += 1
    Y = [cnt_ham, cnt_spam]

    plt.bar(X, Y, 0.4, color='green')

    plt.xlabel("Mail type")
    plt.ylabel("Number")

    plt.title("Train Data Type")
    plt.show()
    print("123")

def predict_dataset(_train_word_dict, _spam_count, _ham_count,query):
    # query=''
    # query=input("input")
    stop_words = load_stop_word()
    index_list={}
    content=query
    res=''
    word_dict=create_word_dict(content, stop_words)
        # 先验概率 已经取了对数
    total_count = _ham_count + _spam_count

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
        res = 'spam'

    else:
        is_spam = 0
        res = 'ham'
    return res

if __name__=='__main__':
    # train_word_dict=read_word()
    # spam_count=read_spam()
    # ham_count=read_ham()
    # predict_dataset(train_word_dict,spam_count,ham_count)
    plotter()
