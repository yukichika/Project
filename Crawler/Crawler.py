#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import urllib.request
import urllib.error
import lxml.html
import copy
import re
import json
import pickle
import codecs
from datetime import datetime
from langdetect import detect
from extractcontent3 import ExtractContent
import configparser
from distutils.util import strtobool

import Yahoo_crawler as yahoo

(BOTH,PARENT_ONLY,CHILD_ONLY) = range(0,3)

"""
あくまで1ページに対して一つのインスタンス．
ノードの作成(重複を許すかなど)は後で形成するときに考慮する．
更新のたびに読み込んで，終わったら書き出して解放．
"""
class Page:
    def __init__(self,url=None,title=None,text=None,text2=None,html=None,getpage_time=None,langdetect_time=None):
        self.json_dict = {"url":url,
                          "title":title,
                          "text":text,
                          "text2":text2,
                          "html":html,
                          "getpage_time":getpage_time,
                          "langdetect_time":langdetect_time}

    def load(self,pages_dir,id):
        f_path = os.path.join(pages_dir,str(id))+".json"
        with open(f_path,"r",encoding='UTF-8') as fj:
            self.json_dict = json.load(fj)

    def set_id(self,id):
        self.json_dict["id"] = id

    def set_links(self,parents,childs):
        self.set_parents(parents)
        self.set_childs(childs)

    def set_parents(self,parents):
        self.json_dict["parents"] = parents

    def set_childs(self,childs):
        self.json_dict["childs"] = childs

    def set_elapsed_time(self,backlink_time,fwdlink_time):
        self.set_backlink_time(backlink_time)
        self.set_fwdlink_time(fwdlink_time)

    def set_backlink_time(self,backlink_time):
        self.json_dict["backlink_time"] = backlink_time

    def set_fwdlink_time(self,fwdlink_time):
        self.json_dict["fwdlink_time"] = fwdlink_time

    def write(self,pages_dir):
        try:
            with codecs.open(os.path.join(pages_dir,str(self.json_dict.get("id")))+".json","w","utf8") as fo:
                json.dump(self.json_dict,fo,indent=4,ensure_ascii=False)
        except Exception as e:
            print(e)
            pass

class Pages_Converter:
    def __init__(self):
        self.id_dict = {}#url→id変換用.

    """url→id変換＆登録"""
    def url_to_id(self,url):
        if url in self.id_dict.keys():
            return self.id_dict.get(url)
        else:
            new_id = len(self.id_dict.keys())
            self.id_dict[url] = new_id
            return new_id

    """id→url変換"""
    def id_to_url(self,id):
        try:
            ind = list(self.id_dict.values()).index(id)
            return list(self.id_dict.keys())[ind]
        except ValueError:
            return None

"""
htmlデータからタグ除去したテキストデータを抽出する
※scriptタグとstyleタグを無視
doc.textcontent()より性能がいい気がする．(タグの除去で性能ってのもよくわからないが)
Args:
	html: str, パースしたいhtmlデータ
Returns:
	text: str, タグ除去されたテキストデータ

############################################################
My_extract_content.py中のものとは違うことに注意．なぜ違うのかは不明．
こちらは画像に付与された説明文は取得せず，テキストデータのみを取得．
############################################################
"""
def strip_tags(html):
    et = lxml.html.fromstring(html)
    try:
        title = et.find(".//title").text
    except:
        title = ""

    xpath = r'//text()[name(..)!="script"][name(..)!="style"]'
    texts = []
    for text in et.xpath(xpath):
        # if text.strip() == None:
        #     continue
        texts.append(text)

    text = ''.join(texts)
    text = re.sub("\\n","",text)
    text = re.sub("\\r","",text)
    text = re.sub("\\t","",text)
    text = re.sub("\&#13;","",text)
    return title,text

class crawl_collector:
    def __init__(self,pages_dir,max_page=400,contain_word=None):
        self.pc = Pages_Converter()
        self.pages_dir = pages_dir
        self.max_page = max_page
        self.contain_word = contain_word
        self.urls = []
        self.searched_urlset = set()

        """UA偽装"""
        opener = urllib.request.build_opener()
        opener.addheaders=[ ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36")]
        urllib.request.install_opener(opener)

    """検索語の設定"""
    def set_search_word(self,search_word):
        self.search_word = search_word

    """yahoo検索によるURLの取得・Webページの取得と保存"""
    def get_urls(self):
        urls = yahoo.search(self.search_word,self.max_page)
        print("raw urls:" + str(len(urls)))
        urls = self.check_urls_and_get_page(urls)
        print("checked urls:" + str(len(urls)))
        self.urls.extend(urls)

    """url群の重複検知および，生死確認・またオプションでその他の条件を指定"""
    def check_urls_and_get_page(self,urls,emit_URLlist=[],omit_domain=True):
        """重複排除"""
        urls = list(set(urls))
        """死亡URL排除・Webページの取得"""
        new_urls = []
        for url in urls:
            if url in self.pc.id_dict:#既に探索済のurlは検査なしで合格
                new_urls.append(url)
                continue
            try:
                page = self.get_page(url)
            except Exception as e:
                print(e)
                continue

            """無事ページを取得できたらid登録＆jsonファイル保存"""
            id = self.pc.url_to_id(url)
            page.set_id(id)
            page.write(self.pages_dir)
            new_urls.append(url)
        return new_urls

    """Webページを取得"""
    def get_page(self,url):
        start_time = datetime.now()

        req = urllib.request.Request(url)#reqを介すとhttp以外もプロトコルに応じて対応できる？
        response = urllib.request.urlopen(req)
        html = response.read()

        """Webページのダウンロードにかかる時間"""
        downloaded_time = datetime.now()
        getpage_time = (downloaded_time-start_time)
        getpage_time = getpage_time.seconds + getpage_time.microseconds*0.000001

        html_ = yahoo.tounicode(html)
        doc = lxml.html.fromstring(html_)
        doc = lxml.html.make_links_absolute(doc,base_url=url)

        """テキストが日本語・テキストが一定量以上含まれているか"""
        title,text = strip_tags(lxml.html.tostring(doc))
        if(len(text)<100):
            raise Exception("Error:len(text)<100")
        if detect(text) != "ja":
            raise Exception("Error:text is not japanese")

        html_str = lxml.html.tostring(doc,encoding="utf-8").decode("utf-8")
        langdetect_time = (datetime.now()-downloaded_time)
        langdetect_time = langdetect_time.seconds + langdetect_time.microseconds*0.000001

        """特定単語を含んでいるか"""
        if(self.contain_word != None):
            if(re.search(self.contain_word,text) == None):
                raise Exception("Errot:keyword is not contained")

        """Extractcontentによる本文の抽出"""
        extractor = ExtractContent()
        opt = {"threshold":50}#thresholdは本文判定の基準となる句読点の数.ただ補正があるので数そのままではないはず．
        extractor.set_option(opt)#readmeとメソッド名が異なる
        extractor.analyse(html_)
        text2,title = extractor.as_text()
        return Page(url=url,title=title,text=text,text2=text2,html=html_str,getpage_time=getpage_time,langdetect_time=langdetect_time)

    """現在持っているurlリストに，それぞれの持つリンク先を追加"""
    def add_tolinks_to_urls(self,use_link_type="ext"):
        cur_urls = copy.deepcopy(self.urls)
        for url in cur_urls:
            if url in self.searched_urlset:
                continue
            t_urls,e_time = self.get_link_urls(url,use_link_type=use_link_type)
            checked_urls = self.check_urls_and_get_page(t_urls)
            self.urls.extend(checked_urls)
        self.searched_urlset = set(cur_urls)

    """urlに含まれる参照(href)をすべて取得"""
    def get_link_urls(self,tgt_url,use_link_type=None):
        """外部リンク・内部リンクの選別"""
        def separate_links(links,domain):
            int_links = []
            ext_links = []

            for link in links:
                if domain in link:
                    int_links.append(link)
                else:
                     ext_links.append(link)
            return int_links,ext_links

        start_time = datetime.now()

        """既にhtmlは取得済みなので，ファイルから読み込む"""
        id = self.pc.id_dict.get(tgt_url)
        with open(os.path.join(self.pages_dir,str(id)+".json"),"r",encoding='UTF-8') as fj:
            node = json.load(fj)

        html_ = node.get("html")
        doc = lxml.html.fromstring(html_)
        doc = lxml.html.make_links_absolute(doc,base_url=tgt_url)

        """hrefタグからurl取得&正規化"""
        urls = [a.attrib.get(u"href") for a in doc.xpath(u"//a")]
        urls = [url for url in urls if type(url) is str and re.match("https?",url) != None]#URL形式でないもの(エラー)を排除
        urls = [yahoo.url_normarization(url) for url in urls]#urlのパラメータ除去

        """外部リンク・内部リンクの指定"""
        if use_link_type != None:
            domain = tgt_url.split("/")[2]
            int_urls,ext_urls = separate_links(urls,domain)
            if use_link_type == "int":
                urls = int_urls
            elif use_link_type == "ext":
                urls = ext_urls

        elapsed_time = (datetime.now()-start_time)
        elapsed_time = elapsed_time.seconds + elapsed_time.microseconds*0.000001
        return urls,elapsed_time

    """現在持っているリンク集合それぞれからリンク先を取得し，リンク集合に存在しないページを排除してidに変換・jsonに追加．"""
    def cvt_tolinks_to_id(self,use_link_type="ext"):
        for url in self.urls:
            t_urls,e_time = self.get_link_urls(url,use_link_type=use_link_type)
            t_urls = list(set(t_urls).intersection(self.pc.id_dict.keys()))#積集合
            t_url_ids = [self.pc.url_to_id(t_url) for t_url in t_urls]
            self.add_to_json(t_url_ids,e_time,url)

    """jsonファイルにリンク情報の追加"""
    def add_to_json(self,ids,time,tgt_url):
        page = Page()
        page.load(self.pages_dir,self.pc.id_dict.get(tgt_url))
        page.set_childs(ids)
        page.set_fwdlink_time(time)
        page.write(self.pages_dir)

    """{url:id}の保存"""
    def save_id_dict(self):
        # pkl_name = os.path.join(os.path.split(self.pages_dir)[0],"id_dict.pkl")
        pkl_name = os.path.join(os.path.split(self.pages_dir)[0],"url_id_dict.pkl")
        with open(pkl_name,"wb") as fo:
            pickle.dump(self.pc,fo)

def main(search_word,root_dir,options):
    """関連フォルダ作成"""
    os.makedirs(root_dir,exist_ok=True)
    pages_dir = os.path.join(root_dir,"Pages")
    os.makedirs(pages_dir,exist_ok=True)

    """コメントログ保存"""
    # comment = options.get("comment")
    # comment_file = "。\n".join(comment.split("。"))
    # with open(os.path.join(root_dir,"comment.txt"),"w",encoding='UTF-8') as file_log:
    #     file_log.write(comment_file)

    """パラメーターの取得"""
    data_params = options.get("data_params")
    max_page = options["max_page"]
    add_childs = options.get("add_childs",False)

    """検索開始"""
    collector = crawl_collector(pages_dir=pages_dir,max_page=max_page)
    for word in search_word:
        """検索語の設定"""
        print("検索単語：" + word)
        collector.set_search_word(word)
        """収集&保存"""
        collector.get_urls()

    data_params["rootlen"] = len(collector.urls)

    """リンク先の収集&保存"""
    if add_childs == True:
        print("---リンク先の収集&保存---")
        collector.add_tolinks_to_urls(use_link_type="ext")#外部リンク

    """リンク関係の保存"""
    collector.cvt_tolinks_to_id()

    """URLとidの保存"""
    collector.save_id_dict()

    """収集データのオプション(設定)を保存"""
    with open(os.path.join(root_dir,"Collect_options.txt"),"w") as fc:
        for key,value in options.items():
            fc.write(f'{key} {value}\n')

"""検索語のリスト => ["iphone","iphone ケース"]"""
def search_words_generator(base_word,append_words):
	ret_list = [base_word]
	for append_word in append_words:
		ret_list.append(base_word+" "+append_word)
	return ret_list

"""保存名の決定"""
def suffix_generator(options):
	suffix = "_" + options["root_word"]
	suffix += "_" + str(options["max_page"])

	if options.get("add_childs",False):
		suffix += "_add_childs"
	if options.get("append",False):
		suffix += "_append"
	return suffix

"""url_id_dictの確認用"""
def check_url_id_dict(root_dir,dict_name="url_id_dict.pkl"):
    file = os.path.join(root_dir,dict_name)
    if os.path.exists(file):
        with open(file,'rb') as fi:
            pg = pickle.load(fi)
        print("取得URL数：" + str(len(pg.id_dict)))
    else:
        print("Not crawled.")

if  __name__ == "__main__":
    """設定ファイルの読み込み"""
    inifile = configparser.ConfigParser(allow_no_value = True,interpolation = configparser.ExtendedInterpolation())
    inifile.readfp(codecs.open("./Crawler.ini",'r','utf8'))

    """検索パラメータの設定"""
    search_word = inifile.get('options','search_word')
    max_page = int(inifile.get('options','max_page'))
    add_childs = strtobool(inifile.get('options','add_childs'))
    append = strtobool(inifile.get('options','append'))
    append_words = inifile.get('options','append_words')
    if append_words == "":
        append_words = []
    else:
        append_words = append_words.split(",")

    """設定"""
    options = {"search_word":search_word,
               "max_page":max_page,
               "add_childs":add_childs,
               "append":append,
               "data_params":dict()
               }

    options["root_word"] = options["search_word"]
    options["search_word"] = search_words_generator(options["root_word"],append_words)
    comment = "%sをキーワードとして検索。収集するページ上限は%d。使用するリンクは外部リンクのみ。"%(options["search_word"],options["max_page"])
    options["comment"] = comment

    """保存先"""
    save_dir = inifile.get('other_settings','save_dir')
    root_dir = save_dir + suffix_generator(options)

    main(search_word=options["search_word"],root_dir=root_dir,options=options)
    check_url_id_dict(root_dir)
