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
Crawler.py��My_extract_content.py��After_crawled_arrange.py�̏��ɉ񂷁D  

�ECrawler.py  
Web�y�[�W�̎��W�ƁC�����N��̎擾�D�i�����p�����[�^�͐ݒ�t�@�C���Ŏw��j  
�\����C�����N��̏���1�i�K�̂ݎ擾�\�ł���_�ɒ��ӁD  

�EMy_extract_content.py  
�{���̒��o�ƁC�s�v�ȃ^�O����菜����html�̕ۑ�.  

�EAfter_crawled_arrange.py  
���������N�ƊO�������N�̔���D  
Crawler.py�ł������E�O�������N���w�肵�Ď��W���Ă��邪�C���W�����y�[�W�̃h���C�����`���Ȃ����C�đI����s���D  
�����܂ŁC���W�������ʂ��猸�炵�Ă������Ƃ����o���Ȃ��_�ɒ��ӁD  

## �擾�ꗗ
�EPages�ijson�t�@�C���j 
```
url:�擾�y�[�W��URL
title:�擾�y�[�W�̃^�C�g��
text:�擾�y�[�W�̃e�L�X�g���iCrawler.py��strip_tags�ɂ��擾�j
text2:�擾�y�[�W�̃e�L�X�g���iextractcontent�ɂ��擾�j
html:�擾�y�[�W��html
getpage_time:�y�[�W�̎擾�ɗv���鎞��
langdetect_time:�e�L�X�g�̃`�F�b�N�ɗv���鎞��
id:�擾�y�[�W��id
childs:�����N���id�i�O�������N�E���������N�͎w��ς݁j
fwdlink_time:���������N�E�O�������N�̑I��ɗv���鎞��
����exttext:�擾�y�[�W�̃e�L�X�g���iMy_extract_content.py��strip_tags�ɂ��擾�j
to_int_links:�����N���id�i�đI���̓��������N�j
to_ext_links:�����N���id�i�đI���̊O�������N�j
```

�Eurl_id_dict.pkl  
�擾����Web�y�[�W��url���L�[�Cid���v�f�̎���  

�ECollect_options.txt  
��������  

�EMyexttext_html�itxt�t�@�C���j  
�擾����Web�y�[�W�̃e�L�X�g��  

�EDropped_html�ihtml�t�@�C���j  
�擾����Web�y�[�W��html  
