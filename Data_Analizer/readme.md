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

## How to Use(for LDA)
・collection_analizer_lda.py  
LDAにより構築したグラフデータを分析するために，エクセルに出力．  

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
pca_lda:LDAトピック分布の第一主成分
auth_score:オーソリティスコア
hub_score:ハブスコア

<topics>
トピック毎の単語分布
```

・adjacent_analizer.py  
特定のノードに対して，隣接ノードの特性を解析．  
リンク元・リンク先のトピック分布の割合を取得．  

・PCA_to_Topics.py  
PCAで次元圧縮した空間と元のトピック分布との対応づけ．  
トピック空間での主成分ベクトルを用いてカラーマップから単語分布を逆引き．  

## How to Use(for Doc2vec)
・collection_analizer_d2v.py  
Doc2vec（1次元）により構築したグラフデータを分析するために，エクセルに出力．  
collection_analizer_d2v_4.pyのみで十分．  

取得一覧  
```
<collection_analizer>
id:Doc2vecの通し番号
name_id:ファイルid
len(text):文字数
url:URL
domain:ドメイン名
len_parent:親リンク数
len_childs:子リンク数
to_int_links:内部リンク数
to_ext_links:外部リンク数
repTopic:代表トピック
pca_lda:LDAトピック分布の第一主成分
pca_d2v:Doc2vecベクトルの第一主成分
auth_score:オーソリティスコア
hub_score:ハブスコア
```

・collection_analizer_d2v_2.py   
collection_analizer_d2vの改良版．  

取得一覧  
```
<collection_analizer>
id:Doc2vecの通し番号
name_id:ファイルid
len(text):文字数
url:URL
domain:ドメイン名
len_parent:親リンク数
len_childs:子リンク数
to_int_links:内部リンク数
to_ext_links:外部リンク数
repTopic:代表トピック
pca_lda:LDAトピック分布の第一主成分
pca_d2v:Doc2vecベクトルの第一主成分
kmeans100:100次元ベクトルを用いたkmeansによるクラスタ数（カラーリストによる着色）
kmeans3:3次元ベクトルを用いたkmeansによるクラスタ数（カラーリストによる着色）
auth_score:オーソリティスコア
hub_score:ハブスコア
```

・collection_analizer_d2v_3.py   
collection_analizer_d2v_2の改良版．  

取得一覧  
```
<collection_analizer>
id:Doc2vecの通し番号
name_id:ファイルid
len(text):文字数
url:URL
domain:ドメイン名
len_parent:親リンク数
len_childs:子リンク数
to_int_links:内部リンク数
to_ext_links:外部リンク数
repTopic:代表トピック
pca_lda:LDAトピック分布の第一主成分
pca_d2v:Doc2vecベクトルの第一主成分
kmeans100_j:100次元ベクトルを用いたkmeansによるクラスタ数（カラーマップによる着色）
kmeans3_j:3次元ベクトルを用いたkmeansによるクラスタ数（カラーマップによる着色）
auth_score:オーソリティスコア
hub_score:ハブスコア
```

・collection_analizer_d2v_4.py   
collection_analizer_d2v_3の改良版．  

取得一覧  
```
<collection_analizer>
id:Doc2vecの通し番号
name_id:ファイルid
len(text):文字数
url:URL
domain:ドメイン名
len_parent:親リンク数
len_childs:子リンク数
to_int_links:内部リンク数
to_ext_links:外部リンク数
repTopic:代表トピック
pca_lda:LDAトピック分布の第一主成分
kmeans100_j:100次元ベクトルを用いたkmeansによるクラスタ数（並び替えた後）（カラーマップによる着色）
auth_score:オーソリティスコア
hub_score:ハブスコア
