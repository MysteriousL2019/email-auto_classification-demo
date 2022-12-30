# mini-sms-classify
>垃圾邮件分类（朴素贝叶斯算法），使用flask+vue


## 朴素贝叶斯分类
## 如何运行
### server端
```bash
cd server
# 创建虚拟环境
python -m venv env
# 激活虚拟环境
source env/bin/activate
# 安装依赖包
pip install -r requirements.txt
#下载jieba，中文分词
https://pypi.org/project/jieba/#files
解压后，打开，使用
python setup.py install, 安装到特定的python运行环境
# 启动flask
python app.py

```
### client端
```bash
npm install
npm run serve
```
* 解释 ： models文件夹下面是英文分词的模型
* models2文件夹下面是中文分词的模型
* 其中的Analyzer.py是评估中文分词效果，运行时候需要将