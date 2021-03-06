# Crawler

## Requirements

This implementation has been tested with the following versions.

```
python 3.6.2
urllib3 1.22
lxml 3.8.0
langdetect 1.0.7
extractcontent3 0.0.2
distutils 1.18.3
```

## How to Use
Crawler.py→My_extract_content.py→After_crawled_arrange.pyの順に回す．  

・Crawler.py  
Webページの収集と，リンク先の取得．（検索パラメータは設定ファイルで指定）  
※構造上，リンク先の情報は1段階のみ取得可能である点に注意．  

・My_extract_content.py  
本文の抽出と，不要なタグを取り除いたhtmlの保存.  

・After_crawled_arrange.py  
内部リンクと外部リンクの判定．  
Crawler.pyでも内部・外部リンクを指定して収集しているが，収集したページのドメインを定義しなおし，再選定を行う．  
※あくまで，収集した結果から減らしていくことしか出来ない点に注意．  

・Mymodule.py  
MyPythonModule/mymodule.pyをpython3に書き換えたもの．  

## 取得一覧
・Pages（jsonファイル） 
```
url:取得ページのURL
title:取得ページのタイトル
text:取得ページのテキスト文（Crawler.pyのstrip_tagsにより取得）
text2:取得ページのテキスト文（extractcontentにより取得）
html:取得ページのhtml
getpage_time:ページの取得に要する時間
langdetect_time:テキストのチェックに要する時間
id:取得ページのid
childs:リンク先のid（外部リンク・内部リンクは指定済み）
fwdlink_time:内部リンク・外部リンクの選定に要する時間
myexttext:取得ページのテキスト文（My_extract_content.pyのstrip_tagsにより取得）
to_int_links:リンク先のid（再選定後の内部リンク）
to_ext_links:リンク先のid（再選定後の外部リンク）
```

・url_id_dict.pkl  
取得したWebページのurlがキー，idが要素の辞書  

・Collect_options.txt  
検索条件  

・Myexttext_html（txtファイル）  
取得したWebページのテキスト文  

・Dropped_html（htmlファイル）  
取得したWebページのhtml  
