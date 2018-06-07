# -*- coding:utf-8 -*-
import urllib
import os
import hashlib
import random
import requests
import json
import sys
import re
from flask import Flask, request, render_template


def dealInput(content):
    #rep = r'[^\u4e00-\u9fa5]'
    #newStr = re.sub(rep, '', content)
    if content == '':
        return '这是一只鸟'
    else:
        return content



defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/show', methods=['POST'])
def show():
    content = request.form['input']
    appKey = '' # youdao app appkey
    secretKey = '' # youdao app secretKey
    content = dealInput(content)
    print(content)
    myurl = 'https://openapi.youdao.com/api'
    q = content.encode('utf-8')
    q=dealInput(q)
    print(q)

    fromLang = 'zh-CHS'
    toLang = 'EN'
    salt = random.randint(1, 65536)
    sign = appKey + q + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode('utf-8'))
    sign = m1.hexdigest()
    myurl = myurl + '?appKey=' + appKey + '&q=' + urllib.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    try:
        http = requests.get(myurl)
        result = json.loads(http.text)['translation'][0]
        result=result.lower()
        print(result)
    except:
        print('error')

    py = "cd ./static/code/ && ./generate0.sh " + "\"" + result + "\"  " +" &"
    command = "echo this is command" + "&&" + "ls"   #command for test
    os.system(py)
    return render_template('show.html')


if __name__ == '__main__':
    app.run(debug=True)
