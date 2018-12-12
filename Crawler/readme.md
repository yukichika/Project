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
tqdm 4.28.1
configparser 3.5.0
```

## How to Use
Crawler.py→My_extract_content.py→After_crawled_arrange.pyの順に回す．  

Crawler.py  
Webページの収集と，リンク先の取得．（検索パラメータは設定ファイルで指定）  
構造上，リンク先の情報は1段階のみ取得可能である点に注意．  

My_extract_content.py  
本文の抽出と，不要なタグを取り除いたhtmlの保存.  

After_crawled_arrange.py  
内部リンクと外部リンクの判定．  
Crawler.pyでも内部・外部リンクを指定して収集しているが，収集したページのドメインを定義しなおし，再選定を行う．  
（childs→to_int_lins&to_ext_links）  

取得一覧（jsonファイル）  
```
url:取得ページのURL
title:取得ページのタイトル
text:取得ページのテキスト文（Crawler.pyのstrip_tagsにより取得）
text2:取得ページのテキスト文（extractcontentにより取得）
html:取得ページのhtml
getpage_time:ページの取得に要する時間
langdetect_time:テキストのチェックに要する時間
id:取得ページのid
childs:リンク先のid（外部リンク・内部リンクは指定）
fwdlink_time:
ｍｙexttext:取得ページのテキスト文（My_extract_content.pyのstrip_tagsにより取得）
to_int_links:リンク先のid（再選定後の内部リンク）
to_ext_links:リンク先のid（再選定後の外部リンク）
```



