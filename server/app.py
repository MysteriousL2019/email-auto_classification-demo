import email
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

import models.sms_classify_main as clf
import models2.main as new
import models2.Analyzer as test
app = Flask(__name__)
# 解决跨域问题
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return jsonify('sms classify api, request </predict> get result.')


# 垃圾邮件检测(英文版)
@app.route('/predict', methods=['GET', 'POST'])
def sms_classify():
    email = request.args.get('email', '')
    print(email+'测试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')#打印出email
    if email == '':
        return jsonify(email=email, lable='')
    result = clf.predict(email)
    print(result+" result !!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    flag=0
    if result=='spam' or result=='ham':
        flag=1
    return jsonify(email=email, label=result,flag=flag)

@app.route('/lookUp',methods=["GET","POST"])
def sms_classify_chinese():
    email=request.args.get('email','')
    print(email+"  测试")
    train_word_dict = new.read_word()
    spam_count = new.read_spam()
    ham_count = new.read_ham()
    result = test.predict_dataset(train_word_dict, spam_count,ham_count,email)
    print(result)
    flag=0
    if result=='spam' or result=='ham':
        flag=1
    return jsonify(label=result,flag=flag)


if __name__ == '__main__':
    app.run(debug=True)
