# -*- encoding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import keras
import os
import cPickle as pkl

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP
import jieba
jieba.enable_parallel(8)

import sys
reload(sys)
sys.setdefaultencoding('utf8')

stopwords = stopwords.words('english')
english_punctuations = [',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%']
stopwords += english_punctuations


# nlp = StanfordCoreNLP(r'/home/linxuming/nltk_data/stanford/stanford-corenlp-full-2016-10-31/', memory='8g')


raw_data_path = 'data/UM-Corpus/data'


def data_preprocess():
	train_en_corpus = []
	train_ch_corpus = []
	test_en_corpus = []
	test_ch_corpus = []
	count = 0

	for dirname in ('Bilingual', 'Testing'):
		if dirname == 'Bilingual':
			sub_dir_names = ['Education', 'Laws', 'Microblog', 'News', 
								'Science', 'Spoken', 'Subtitles', 'Thesis']
			for filename in sub_dir_names:
				with open(os.path.join(raw_data_path, dirname, filename, ''.join(['Bi-', filename, '.txt']))) as fr:
					for i, line in enumerate(fr):
						line = line.strip().decode()
						count += 1
						if i % 2 == 0:
							train_en_corpus.append(line)
						else:
							train_ch_corpus.append(line)
					print('Finished {}'.format(count))
		else:
			with open(os.path.join(raw_data_path, dirname, 'Testing-Data.txt')) as fr:
				for i, line in enumerate(fr):
					line = line.strip().decode()
					count += 1
					if i % 2 == 0:
						test_en_corpus.append(line)
					else:
						test_ch_corpus.append(line)
				print('Finished {}'.format(count))
	train_en_corpus = '\n'.join(train_en_corpus)
	train_ch_corpus = '\n'.join(train_ch_corpus)
	test_en_corpus = '\n'.join(test_en_corpus)
	test_ch_corpus = '\n'.join(test_ch_corpus)
	return train_en_corpus, train_ch_corpus, test_en_corpus, test_ch_corpus


def segment(corpus, tokenizer, savepath=None):
	tokenized_corpus = []
	count = 0
	tokenized_corpus = ' '.join([_ for _ in tokenizer(corpus) if _.strip()])
	tokenized_corpus = tokenized_corpus.split(' \n ')
	# for sentence in corpus:
	# 	count += 1
	# 	tokenized_corpus.append(' '.join(tokenizer(sentence)))
	# 	if count % 1000 == 0:
	# 		print('Finished cutting {}'.format(count))
	if not savepath:
		return tokenized_corpus
	else:
		with open(savepath, 'w') as fw:
			pkl.dump(tokenized_corpus, fw)


if __name__ == '__main__':
	train_en_corpus, train_ch_corpus, test_en_corpus, test_ch_corpus = data_preprocess()
	segment(train_en_corpus, jieba.cut, 'data/preprocess/train_en_segment.pkl')
	segment(train_ch_corpus, lambda k: iter(k.strip()), 'data/preprocess/train_ch_segment.pkl')
	segment(test_en_corpus, jieba.cut, 'data/preprocess/test_en_segment.pkl')
	segment(test_ch_corpus, lambda k: iter(k.strip()), 'data/preprocess/test_ch_segment.pkl')









