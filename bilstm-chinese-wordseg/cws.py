#!/usr/bin/env python
# coding:utf8

import os
import os.path
import sys 
reload(sys)
sys.setdefaultencoding('utf8')

# 读取人民日报语料
data = []
for parent, dirnames, filenames in os.walk('../data/2014/'):
		for filename in filenames:
			fr = open(parent + '/' + filename, 'r')
			for line in fr:
				line = line.strip()