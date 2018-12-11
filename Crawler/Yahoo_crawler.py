#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使い方
基本的にsearchを呼ぶだけ．
第一引数は検索語リスト，第二引数は検索するページ数(ヒットするページは最大でページ数×10)，第三引数はスリープ時間(デフォルトは3秒)
類似ページについてはyahoo側で省いてくれているはず(省きたくない場合は"&dups=1"オプションをurlに追加する)
"""

import re
import urllib.request
import urllib.error
import urllib.parse
import lxml.html
import time
import datetime
import random
import os
from functools import reduce

"""
python2：str型=>unicode型
python3：bytes型=>str型
"""
def tounicode(data):
    f = lambda d, enc: d.decode(enc)
    codecs = ['shift_jis','utf-8','euc_jp','cp932',
              'euc_jis_2004','euc_jisx0213','iso2022_jp','iso2022_jp_1',
              'iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext',
              'shift_jis_2004','shift_jisx0213','utf_16','utf_16_be',
              'utf_16_le','utf_7','utf_8_sig']

    for codec in codecs:
        try: return f(data, codec)
        except: continue
    return None

"""
yahoo検索ページのURL作成
=> http://search.yahoo.co.jp/search?p=ラーメン&ei=UTF-8&fl=2&dups=1&b= 1,11,21,...,max_page+1
"""
def make_url(searchwordlist,page):
    url_a = "http://search.yahoo.co.jp/search?p="
    url_b = "&ei=UTF-8&fl=2&dups=1&b=" + str(page*10+1)

    if type(searchwordlist) is str :#searchwordlistが一要素のとき，リストにし忘れても通るように回避
        searchword = searchwordlist.replace(" ","+")
    else:
        searchword = reduce(lambda x, y: x+"+"+y,searchwordlist)#["aaa","bbb"] => "aaa+bbb"

    url = url_a + urllib.parse.quote_plus(searchword,encoding='UTF-8') + url_b
    return url

"""
URL正規化(重複対策でプロトコルのゆれとurlのパラメータの変化に対応)
https://macaro-ni.jp/35287 => http://macaro-ni.jp/35287
"""
re_c = re.compile("[;?:@=+$#!'()*]")
def url_normarization(url):
    url = url.replace("https","http")
    url = url.replace("\\","/")
    dirname,basename = os.path.split(url.strip("/"))
    pos = re.search(re_c,basename)
    if pos != None:
        new_basename = basename[:pos.start()]
        url = os.path.join(dirname,new_basename)
    url=url.strip("/")
    return url

"""
取得したい記事のURL取得(検索ページ全部)
=>http://ja.wikipedia.org/wiki/%E3%83%A9%E3%83%BC%E3%83%A1%E3%83%B3
"""
def search_page(src_url,collected_urls):
    """通信成功するまで繰り返してurl取得"""
    while True:
        try:
            req = urllib.request.Request(src_url)#reqを介すとhttp以外もプロトコルに応じて対応できる？
            response = urllib.request.urlopen(req)
            html = response.read()
            break

        except Exception as e:
            print(e)
            print(datetime.datetime.today().now())

            randint = random.randrange(10)
            randflt = random.randrange(10)

            opener = urllib.request.build_opener()
            ua = 'Mozilla/'+str(randint)+"."+str(randflt)+'(Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
            opener.addheaders = [('User-agent',ua)]
            urllib.request.install_opener(opener)

    # with open("c.html","wb") as file_html:
	#     file_html.write(html)

    """URL取得(検索ページ全部)"""
    urls = []
    dom = lxml.html.fromstring(html)
    for tree in dom.xpath('//div[@id = "WS2m"]//h3/a'):
        try:
            url = tree.attrib["href"]
            url = url_normarization(url)
            if url in collected_urls:
                return urls,True
            urls.append(url)
        except:
            print("htmlからのurlの取り出し失敗")
    return urls,False

"""
取得したい記事のURL取得(取得したい数の分)
=>http://ja.wikipedia.org/wiki/%E3%83%A9%E3%83%BC%E3%83%A1%E3%83%B3
"""
def search(searchwordlist,max_collect=None,sleep_time=10):
    """UA偽装"""
    opener = urllib.request.build_opener()
    opener.addheaders=[ ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36")]
    urllib.request.install_opener(opener)

    urls = []
    if max_collect == None:#得られる限り取得
        i = 0
        while(1):
            time.sleep(sleep_time)#最後に入れると1ページずつしか巡回しない場合待ち時間0になる
            target = make_url(searchwordlist,page=i)
            temp_urls,overlapp_flag = search_page(target,urls)

            """重複フラグ"""
            if overlapp_flag is True:
                break

            urls += temp_urls

            """1回で集めたページ数が10以下"""
            if len(temp_urls) < 10 :
                break

            i += 1

        """
        検索結果からwebページのurlを抜き出してurlsに追加．
        終了条件は指定ページ数の巡回が終わるか，1回で集めたページ数が10を下回るか，重複フラグが立つまで．
        ヒットしたページ数が10の倍数だった場合以外は，取得url数から終了判定ができるが，
        10の倍数の場合はもう一度最終検索結果を表示するため，重複する
        """
    else:
        max_page_num = int((max_collect/10) + 1)
        rest = max_collect % 10
        for i in range(max_page_num):
            time.sleep(sleep_time)#最後に入れると1ページずつしか巡回しない場合待ち時間0になる
            target = make_url(searchwordlist,i)
            temp_urls,overlapp_flag = search_page(target,urls)

            """指定ページ数の巡回が終了する時"""
            if i is (max_page_num-1):#最後のコレクション
                temp_urls = temp_urls[:rest]

            """重複フラグ"""
            if overlapp_flag is True:
                break

            urls += temp_urls

            """1回で集めたページ数が10以下"""
            if len(temp_urls) < 10 :
                break
    return urls

def main():
    searchwords = ['"https://www.microsoft.com/surface/ja-jp/accessories/browse#accessories5"']
    urls = search(searchwords,max_collect=None,sleep_time=3)

if __name__ == "__main__":
    main()
