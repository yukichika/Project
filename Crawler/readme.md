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


