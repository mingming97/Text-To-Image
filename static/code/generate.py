# -*- coding: UTF-8 -*-
import argparse
import os
import hashlib
import urllib
import random
import requests
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="generate")
    parser.add_argument('--content', dest='content', default='这是一只鸟', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    defaultencoding = 'utf-8'
    if sys.getdefaultencoding() != defaultencoding:
        reload(sys)
        sys.setdefaultencoding(defaultencoding)
    args = parse_args()
    appKey = '2a6e165bec543fcc'
    secretKey = 'gMLRjhtH6vF2LaPoXtN4Rb5TMFuxkfi0'

    myurl = 'https://openapi.youdao.com/api'
    q = args.content
    fromLang = 'zh-CHS'
    toLang = 'EN'
    salt = random.randint(1, 65536)
    sign = appKey + q + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode('utf8'))
    sign = m1.hexdigest()
    myurl = myurl + '?appKey=' + appKey + '&q=' + urllib.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        http = requests.get(myurl)
        txt = json.loads(http.text)['translation'][0]
        txt=txt.lower()
        print(txt)
    except:
        print('error')

    #sh = "./generate.sh" + "\"" + txt + "\""
    print(txt)
    #os.system(sh)
