# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 01:39:35 2020

@author: jsx
"""

import requests
from urllib.parse import urlencode
import json
import time

aweme_id_lst = []

def login(account, password):
    print ("开始模拟登录蝉妈妈")
    
    postUrl = "https://api-service.chanmama.com/v1/access/token"
    
    headers = {
        'Host': 'api-service.chanmama.com',
        'Referer': 'https://www.chanmama.com/login',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
    }

    postData = {"appId": 10000,
                "timeStamp": int(time.time()),
                "username": account,
                "password": password,
                }
    responseRes = requests.post(postUrl, data = postData, headers = headers)
    print(f"statusCode = {responseRes.status_code}")
    print(f"text = {responseRes.text}")
 
# 定义一个方法来获取每次请求的结果。在请求时，day、page是可变参数，所以我们将它作为方法的参数传递进来
def get_aweme_id_list(page, targetDay):
    # 表示请求的URL的前半部分
    base_url = 'https://api-service.chanmama.com/v1/home/rank/productAweme?'
    
    headers = {
        'Host': 'api-service.chanmama.com',
        'Referer': 'https://www.chanmama.com/promotionAwemeRank?category=',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
    }
    #  构造参数字典，其中day_type、category、order_by和size是固定参数，day、page是可变参数
    params = {
        'day_type': 'day',
        'day': targetDay,
        'category': '',
        'order_by': 'digg',
        'page': page,
        'size': '50'
    }
    # 调用urlencode()方法将参数转化为URL的GET请求参数，
    # 即类似于day_type=day&day=2020-07-23&category=&order_by=digg&page=1&size=50这样的形式
    # base_url与参数拼合形成一个新的URL。
    url = base_url + urlencode(params)
    try:
        # 我们用requests请求这个链接，加入headers参数
        session = requests.Session()
        response = session.get(url, headers = headers)
        # 然后判断响应的状态码，如果是200，则直接调用json()方法将内容解析为JSON返回，否则不返回任何信息。
        if response.status_code == 200 and response.json():
            awemes = response.json().get('data')
            for index, aweme in enumerate(awemes):
                aweme_id_lst.append(aweme.get('aweme_id'))
    # 如果出现异常，则捕获并输出其异常信息。
    except requests.ConnectionError as e:
        print('Error', e.args)
        
def get_aweme_page(aweme_id):
    base_url = 'https://api-service.chanmama.com/v1/home/aweme/info?'
    
    headers = {
        'Host': 'api-service.chanmama.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
    }
    
    params = {
        'aweme_id': aweme_id
    }
    
    url = base_url + urlencode(params)
    try:
        session = requests.Session()
        response = session.get(url, headers = headers)
        return response.json()
        
    except requests.ConnectionError as e:
        print('Error', e.args)

def parse_aweme(aweme_rank,aweme_page,category):#归档
    
    download_url = 'https://www.chanmama.com/awemeDetail/' + aweme_rank.get('aweme_id')
    
    aweme =  {'aweme_id': aweme_rank.get('aweme_id'),
              
              'aweme_title': aweme_page.get('data').get('aweme').get('aweme_title'),
              'aweme_url': aweme_page.get('data').get('aweme').get('aweme_url'),
              'download_url': download_url,
              'digg_count': aweme_page.get('data').get('aweme').get('digg_count'),
              'digg_incr': aweme_rank.get('digg_count'),
              'aweme_create_time': aweme_page.get('data').get('aweme').get('aweme_create_time'),
              'category': category,#category按1：女装，2：男装，3：美妆护理，4：鞋包饰品，5：日用百货，6：母婴玩具，7：食品生鲜，8：运动户外，9：鲜花家纺，10：宠物农资，11：汽车配件，12：手机数码，13：生活家电，14：家装建材，15：图书音像
              'volume': aweme_rank.get('volume'),
              'amount': aweme_rank.get('amount')
              }
    author = {'author_id': aweme_page.get('data').get('author').get('author_id'),
              'nickname': aweme_page.get('data').get('author').get('nickname'),
              'follower_count': aweme_page.get('data').get('author').get('follower_count')
              }
    aweme_detail = {'aweme': aweme,
                    'author': author
                    }
    return aweme_detail

def get_day_incr(aweme_id):#校准点赞日增数
    
    postUrl = "https://api-service.chanmama.com/v1/home/aweme/multiDataChart"
    
    headers = {
        'Host': 'api-service.chanmama.com',
        'Referer': 'https://www.chanmama.com/awemeDetail/' + aweme_id,
        'Origin': 'https://www.chanmama.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
    }

    postData = {"aweme_id": aweme_id,
                "start_date": "2020-06-27",
                "end_date": "2020-07-27",
                }
    responseRes = requests.post(postUrl, data = postData, headers = headers)
    print(f"statusCode = {responseRes.status_code}")
    print(f"text = {responseRes.text}")

if __name__ == '__main__':
    for category in range(1,16):
        selected_data = {'data': []}
        count = 0
        for index in range(1,5):
            filepath_read = r'./2020年7月24日/' + str(category) + r'/' + str(index) + '.json'
            try:
                with open(filepath_read,'r', encoding = 'utf-8') as load_f:
                    load_rank = json.load(load_f,strict = False)
                    awemes = load_rank.get('data')
                    for index, aweme in enumerate(awemes):
                        aweme_page = get_aweme_page(aweme.get('aweme_id'))
                        if(aweme_page.get('data').get('author').get('follower_count') < 1000000 and
                           aweme_page.get('data').get('aweme').get('digg_count') >1000):
                            aweme_detail = parse_aweme(aweme,aweme_page,category)
                            selected_data['data'].append(aweme_detail)
                            count = count + 1
                        time.sleep(2)
                load_f.close()
                print(filepath_read + " has been maanaged")
            except Exception as e:
                print(e)
                continue
        filepath_write = r'./2020年7月24日/record/' + str(category) + '.json'
        with open(filepath_write, "w", encoding = 'utf-8') as f:
            json.dump(selected_data, f, ensure_ascii=False)
            print("加载入文件完成...")
        f.close()
        
        print(count)
    