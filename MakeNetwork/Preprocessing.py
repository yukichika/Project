#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import zenhan
import MeCab
import os
import codecs
from collections import Counter

def delete_pre(text):
    text = text.replace("　","")
    text = text.replace(" ","")

    text = re.sub(r'\(.*?\)',"",text)#半角()を中身ごと削除
    text = re.sub(r'\（.*?\）',"",text)#半角()を中身ごと削除
    return text

def simple_sentence(line):
    sentence_lists = re.findall(r'[^。]+(?:[。]|$)', line)
    return sentence_lists

def wakati(text):
    MECAB_MODE = "-Owakati"
    tagger = MeCab.Tagger(MECAB_MODE)
    result = tagger.parse(text)
    return result

def delete_aft(line):
    text = zenhan.z2h(line,mode=1)#アルファベット（全角→半角）
    text = zenhan.z2h(text,mode=2)#数字（全角→半角）
    text = zenhan.h2z(text,mode=4)#カタカナ（半角→全角）

    text = re.sub(r'[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u3004\u3007-\u303F\u3099-\u30A0\u30FB\u30FD-\u4DFF\uA000-\uE0FFF]',"",text)#その他文字列削除
    return text

"""ディレクトリ内の全てのファイルの絶対パスを取得"""
def get_all_paths(directory):
    corpus = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            corpus.append(os.path.join(root, file))
    return corpus

"""文書内の総単語をリストで取得"""
def words_list(text):
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(text).splitlines()
    allwords = []
    for line in lines:
        chunks = line.split('\t')
        if not chunks[0] == "\ufeff":#UTF-8の識別子\ufeff
            allwords.append(chunks[0])
    return allwords

def words_count(file):
    with codecs.open(file,'r','UTF-8','ignore') as file_in:
        sentence = file_in.read()
        words = words_list(sentence)#全単語
        # words = Wakati.words_list_select(sentence)#品詞選択
        word_count = len(words)
    return word_count

"""ディレクトリ内の全てのファイルの語彙数"""
def words_vocab(dir):
    filelists = get_all_paths(dir)
    print("ファイル数：" + str(len(filelists)))
    totalwords = []
    for file in filelists:
        with codecs.open(file,'r','UTF-8','ignore')as file_in:
            sentence = file_in.read()
        totalwords.extend(words_list(sentence))

    counter = Counter(totalwords)
    vocab_count = len(counter)
    return vocab_count

"""ディレクトリ内の全てのファイルの平均単語数"""
def words_average(dir):
    filelists = get_all_paths(dir)
    print("ファイル数：" + str(len(filelists)))
    if len(filelists) == 0:
        averagewords = 0
    else:
        totalwords = 0
        for file in filelists:
            totalwords += words_count(file)
        averagewords = totalwords / len(filelists)
    return averagewords

if __name__ == "__main__":
    text = "aaa    （aaa）　　　　　あ"
    print(delete_pre(text))
