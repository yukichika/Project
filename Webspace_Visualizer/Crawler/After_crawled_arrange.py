#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crawlerで収集した後に，やり忘れたことを再実行してjsonを変更するスクリプト．
当然だが，基本的に収集結果から減らしていくことしかできない．
・外部リンク・内部リンクの定義変更(ドメイン名の判定位置を変える)して再選定
"""

import os
import json
import codecs
import configparser
from distutils.util import strtobool

import Mymodule
from Crawler import suffix_generator

"""
urlからドメイン部分を抽出して返す．
FQDNのドットで区切られたブロック数が3以下の場合，ドメイン名はFQDN自身
4以上の場合，下3ブロックをドメインとして返す．
@arg
[IN]url:ドメインを取得したいURL
@ret
(unicode) domain
"""
def domain_detect(url):
    FQDN = url.split("/")[2]
    FQDN_list = FQDN.split(".")
    return ".".join(FQDN_list[-3:])#開始点がリスト長より長くても問題なく無く動く

"""外部リンク・内部リンクの選別"""
def separate_links(links,domain,num_to_url_dict):
    int_links = []
    ext_links = []

    for link in links:
        link_url = num_to_url_dict[link]
        if domain in link_url:
            int_links.append(link)
        else:
            ext_links.append(link)
    return int_links,ext_links

def main(root_dir,tasks):
    """関連フォルダの存在確認"""
    if not os.path.exists(root_dir):
        print(root_dir + " is not exist")
        exit()

    src_pages_dir = os.path.join(root_dir,"Pages")
    if not os.path.exists(src_pages_dir):
        print(src_pages_dir + " is not exist")
        exit()

    """jsonファイルのリストとソート"""
    json_files = os.listdir(src_pages_dir)
    Mymodule.sort_nicely(json_files)

    """取得済みのWebページのURLのリストを先に作成しておく"""
    num_to_url_dict = {}
    for json_file in json_files:
        root,ext = os.path.splitext(json_file)
        with open(os.path.join(src_pages_dir,json_file),"r",encoding='UTF-8') as fj:
            json_data = json.load(fj)

        url = json_data.get("url")
        num_to_url_dict[int(root)] = url

    """外部リンク・内部リンクの選定"""
    for json_file in json_files:
        root,ext = os.path.splitext(json_file)
        with open(os.path.join(src_pages_dir,json_file),"r",encoding='UTF-8') as fj:
            json_data = json.load(fj)

        if "sep_int_or_ext" in tasks:
            """ドメインの取得"""
            src_url = json_data.get("url")
            domain = domain_detect(src_url)

            """リンク先取得"""
            to_links = json_data.get("childs")
            #from_links = json_data.get("parents")

            """再選定"""
            to_int_links,to_ext_links = separate_links(to_links,domain,num_to_url_dict)
            json_data["to_int_links"] = to_int_links
            json_data["to_ext_links"] = to_ext_links

        with codecs.open(os.path.join(src_pages_dir,json_file),"w","utf8") as fo:
            json.dump(json_data,fo,indent=4,ensure_ascii=False)
        print(str(root) + "finished")

if __name__=="__main__":
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

    tasks = inifile.get('options','tasks')
    if tasks == "":
        tasks = []
    else:
        tasks = tasks.split(",")

    """保存先"""
    save_dir = inifile.get('other_settings','save_dir')
    root_dir = save_dir + suffix_generator(options)

    main(root_dir,tasks)
