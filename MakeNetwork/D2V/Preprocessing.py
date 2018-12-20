#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import zenhan
import MeCab

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

if __name__ == "__main__":
    text = "aaa    （aaa）　　　　　あ"
    print(delete_pre(text))
