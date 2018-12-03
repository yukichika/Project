#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import urllib.error
import lxml.html
import re
import os
import json
import codecs
from langdetect import detect
import configparser
from distutils.util import strtobool
from tqdm import tqdm

from Crawler import suffix_generator
import Mymodule

CONTENT_TAGS = set(("p","div","pre","blockquote","section","main","article","ul","ol","li","form","center"))
CONTENT_TAGS.update(("strong","small","font","basefont","span","i","b","tt","strike","big","em"))
CONTENT_TAGS.update(("h1","h2","h3","h4","h5","h6"))
CONTENT_TAGS.update(("dt","dl","dd"))
CONTENT_TAGS.update(("td","table","tbody","th","tr"))
CONTENT_TAGS.update(("img","a"))

"""
htmlデータからタグ除去したテキストデータを抽出する(Crawler.pyのものとは少し違う点に注意)
※scriptタグとstyleタグを無視
doc.textcontent()より性能がいい気がする．(タグの除去で性能ってのもよくわからないが)
Args:
	html: str, パースしたいhtmlデータ
Returns:
	text: str, タグ除去されたテキストデータ

############################################################
Crawler.py中のものとは違うことに注意．
異なる点は，テキストデータに加えて，画像に付与された説明文も取得している点．
############################################################
"""
def strip_tags(html,no_less=20):
    et = lxml.html.fromstring(html)
    try:
        title = et.find(".//title").text
    except:
        title = ""

    """テキストデータの抽出"""
    xpath = r'//text()[name(..)!="script"][name(..)!="style"]'
    texts = []
    for text in et.xpath(xpath):
        #if text.strip() == None:#この条件なんだ？空白除去してもNoneにはならんし．．．
        #	continue
        text = re.sub("\\n|\\r|\\t|\&#13;","",text)
        if text == None or len(text) < no_less:
            continue
        texts.append(text)

    """画像に付与された説明文の取得"""
    xpath = r'//img'#altの値がsrcと異なるimg(urlがそのままaltになっていないimg)
    for img_node in et.xpath(xpath):
        text = img_node.get("alt","")
        text = re.sub("\\n|\\r|\\t|\&#13;","",text)#改行コード等のほか，スペースの多重連続も除去
        if text == None or len(text) < no_less:
            continue
        try:
            if detect(text) != "ja":
                continue
        except Exception as e:
            print(e)
            continue
        texts.append(text)

    text = ''.join(texts)
    return title,text

"""bodyの頭から探索を行い，CONTENT_TAGSに該当しないタグは切り落としていく"""
def recurrent_content_search(dom):
    for item in list(dom):
        if item.tag in CONTENT_TAGS:
            if item.tag == "p":
                pass
            recurrent_content_search(item)
        else:
            item.drop_tree()

"""
収集済みのjsonファイルからhtmlを抜き出し，コンテンツに関係のないタグをドロップする．
ドロップしたhtmlと取得したテキスト文の保存．
"""
def from_collected_jsons(root_dir):
    """関連フォルダの存在確認"""
    if not os.path.exists(root_dir):
        print(root_dir + " is not exist")
        exit()

    src_pages_dir = os.path.join(root_dir,"Pages")
    if not os.path.exists(src_pages_dir):
        print(src_pages_dir + " is not exist")
        exit()

    """保存フォルダの作成(既にある場合は終了)"""
    dropped_html_dir = os.path.join(root_dir,"Dropped_html")
    if os.path.exists(dropped_html_dir):
        print(dropped_html_dir + " is already exist")
        exit()
    os.mkdir(dropped_html_dir)

    myExtText_dir = os.path.join(root_dir,"Myexttext_raw")
    if os.path.exists(myExtText_dir):
        print(myExtText_dir + " is already exist")
        exit()
    os.mkdir(myExtText_dir)

    """jsonファイルのリストとソート"""
    json_files = os.listdir(src_pages_dir)
    Mymodule.sort_nicely(json_files)

    for json_file in tqdm(json_files):
        root,ext = os.path.splitext(json_file)
        with open(os.path.join(src_pages_dir,json_file),"r",encoding='UTF-8') as fj:
            json_data = json.load(fj)

        html_ = json_data.get("html")
        dom = lxml.html.fromstring(html_)
        body = dom.body
        recurrent_content_search(body)
        new_html = lxml.html.tostring(dom)
        title,text = strip_tags(new_html)

        """htmlの保存"""
        with open(os.path.join(dropped_html_dir,root+".html"),"wb") as fo:
            fo.write(new_html)
        """textの保存"""
        with open(os.path.join(myExtText_dir,root+".txt"),"w",encoding='UTF-8') as fo:
            fo.write(text)

        json_data["myexttext"] = text
        with codecs.open(os.path.join(src_pages_dir,json_file),"w","utf8") as fo:
            json.dump(json_data,fo,indent=4,ensure_ascii=False)

        # print(str(root) + "finished")

"""
URLを指定してhtmlを取得し，コンテンツに関係のないタグをドロップする．
ドロップしたhtmlと取得したテキスト文を保存．
"""
def main():
    url = r"http://www.ymobile.jp/iphone/"

    opener = urllib.request.build_opener()
    opener.addheaders=[ ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36")]
    urllib.request.install_opener(opener)

    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    html = response.read()

    html_ = Mymodule.tounicode(html)
    root = lxml.html.fromstring(html_)
    body = root.body
    recurrent_content_search(body)
    new_html = lxml.html.tostring(root)
    title,text = strip_tags(new_html)

    """htmlの保存"""
    with open("doroped_html.html","wb") as fo:
        fo.write(new_html)
    """textの保存"""
    with open("doroped_text.txt","w",encoding='UTF-8') as fo:
        fo.write(text)

if __name__ == "__main__":
    """設定ファイルの読み込み"""
    inifile = configparser.ConfigParser(allow_no_value = True,interpolation = configparser.ExtendedInterpolation())
    inifile.readfp(codecs.open("./Crawler.ini",'r','utf8'))

    """検索パラメータの設定"""
    search_word = inifile.get('options','search_word')
    max_page = int(inifile.get('options','max_page'))
    add_childs = strtobool(inifile.get('options','add_childs'))
    append = strtobool(inifile.get('options','append'))

    """設定"""
    options = {"search_word":search_word,
               "max_page":max_page,
               "add_childs":add_childs,
               "append":append,
               }

    options["root_word"] = options["search_word"]

    """保存先"""
    save_dir = inifile.get('other_settings','save_dir')
    root_dir = save_dir + suffix_generator(options)

    from_collected_jsons(root_dir)
