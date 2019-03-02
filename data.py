#-*- coding=utf-8 -*-

import csv
import sys
import pandas as pd

colnames = ['用户ID','会话ID'	,'用户问题','标准问题','问题答案','访问时间','问题类型','回答类型','相似度','模块','知识分类','平台','群体','客户端']

df = pd.read_csv('userLogs.csv')#names=colnames


    # if '建议问' in line:
    #     print line

df.sort_values('访问时间', inplace=True, ascending=True)
df.sort_values('用户ID', inplace=True, ascending=False)

flag = 0
count = 0
for index, row in df.iterrows():
    if count == 20:
        break
    if row['回答类型'] == '建议问':
        count += 1
        flag = 1
        print row['用户ID'], row['会话ID'], row['用户问题'],row['标准问题'],row['访问时间'],row['回答类型'],row['相似度']#,row['知识分类'],row['平台'],row['群体'],row['客户端']
    elif flag == 1:
        flag = 0
        print row['用户ID'], row['会话ID'], row['用户问题'],row['标准问题'],row['问题答案'],row['访问时间'],row['问题类型'],row['回答类型'],row['相似度'],row['模块'],row['知识分类'],row['平台'],row['群体'],row['客户端']
