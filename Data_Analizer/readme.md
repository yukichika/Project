# Data_Analizer

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
numpy 1.11.3
matplotlib 1.5.1
scikit-learn 0.18.1
xlsxwriter 0.9.6
scipy 0.19.0
```

## How to Use
collection_analizer2.py  
グラフを構築したデータを分析するために，エクセルに出力．  

取得一覧  
```
<collection_analizer>
id:LDAの通し番号
name_id:ファイルid
len(text):文字数
url:URL
domain:ドメイン名
len_parent:親リンク数
len_childs:子リンク数
to_int_links:内部リンク数
to_ext_links:外部リンク数
repTopic:代表トピック
pca:LDAトピック分布の第一主成分
pca_d2v:doc2vecベクトルの第一主成分
auth_score:オーソリティスコア
hub_score:ハブスコア

<topics>
トピック毎の単語分布
```

adjacent_analizer.py  
特定のノードに対して，隣接ノードの特性を解析．  
リンク元・リンク先のトピック分布の割合を取得．  

PCA_to_Topics.py  
PCAで次元圧縮した空間と元のトピック分布との対応づけ．  
トピック空間での主成分ベクトルを用いてカラーマップから単語分布を逆引き．  

