#-*- coding=utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import json
import jieba
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import os.path
import random
import pandas as pd
import cPickle as pickle
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.layers import *
from keras.layers import LSTM
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import *
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.backend import stack
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pprint import pprint


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def create_dic_id():
    with open('userLogs_large.json', 'r') as file:
        cur_id = 0
        dic_id = {}
        for line in file:
            if cur_id == 100000:
                break
            j_line = json.loads(line)
            dic_id[cur_id] = j_line['session_id']
            cur_id += 1
        dic_id[-1] = 0
        dic_id[cur_id] = 0
        dic_id[cur_id+1] = -1
    return dic_id


def create_dic_id_d(data_cleaned):
    cur_id = 0
    dic_id_d = {}
    for j_line in data_cleaned:
        if cur_id == 100000:
            break
        dic_id_d[cur_id] = j_line['session_id']
        cur_id += 1
    dic_id_d[-1] = 0
    dic_id_d[cur_id] = 0
    dic_id_d[cur_id+1] = -1
    return dic_id_d


def load_data(word_vectors, dic_id, label_dic, params):
    X = []
    Y = []
    emt_list = []
    dic_y = {}
    data_cleaned = []
    y_id = 0
    cur_id = 0
    with open('userLogs_large.json', 'r') as file:
        count_total = 0
        for line in file:
            j_line = json.loads(line)
            if cur_id == 100000:
                break
            # print '-------------------------------------------------------------------------'
            # for key in j_line:
            #     print key, j_line[key]
            #     if key == 'a_suggestion':
            #         for item in j_line['a_suggestion']:
            #             print item
            ###########
            # if j_line['session_id'] == 'aa2c970a6c674049a7c67c3ee6043f81':
            #     print '-------------------------------------------------'
            #     print cur_id
            #     for key in j_line:
            #         print key, j_line[key], (dic_id[cur_id] == dic_id[cur_id+1] or dic_id[cur_id] == dic_id[cur_id-1]) and 'a_suggestion' in j_line

            if (dic_id[cur_id] == dic_id[cur_id+1] or dic_id[cur_id] == dic_id[cur_id-1]) and 'a_suggestion' in j_line:
                if len(j_line['a_suggestion']) == 0:
                    emt_list.append(cur_id)
                    cur_id += 1
                    continue
                else:
                    seged_list = jieba.lcut(j_line['q_user'])
                    temp_d = []
                    for i in seged_list:
                        # print i
                        if i in word_vectors:
                            temp_d.append(word_vectors[i])
                    temp_d = np.array(temp_d)
                    if temp_d.size == 0:
                        # print j_line['q_user']
                        # print seged_list
                        emt_list.append(cur_id)
                        cur_id += 1
                        continue
                    else:
                        if temp_d.shape[0] < params['seq_max_len']:
                            tmp_res_d = np.zeros((params['seq_max_len'],200))
                            tmp_res_d[:temp_d.shape[0],:temp_d.shape[1]] = temp_d
                        else:
                            tmp_res_d = temp_d[:params['seq_max_len'],:]
                        data_cleaned.append(j_line)
                        X.append(tmp_res_d)
                        Y.append(label_dic[j_line['topic']])
                        # Y.append(label_dic[j_line['topic']])
                        if j_line['q_user'] not in dic_y:
                            dic_y[j_line['q_user']] = y_id
                            last = y_id
                        y_id += 1
                    cur_id += 1
            else:
                emt_list.append(cur_id)
                cur_id += 1
                continue

        # for i in dic_id:
        #     print i, dic_id[i]

        # for key, value in sorted(dic_y.iteritems(), key=lambda (k,v): (v,k)):
        #     print "dic_y: %s: %s" % (key, value)

        print emt_list
        print len(X), len(Y), len(emt_list), len(dic_id), len(dic_y), last
        return X, Y, emt_list, dic_y, data_cleaned


def load_data_2(X, emt_list, dic_id, dic_y, data_cleaned, params):
    X_D = []
    Y_D = []
    X_A = []
    Y_A = []
    rewards = []
    dic_tmp_a = {}
    dic_d_a = {}
    dic_a_acount = {}

    cur_id = 0
    print emt_list
    for j_line in data_cleaned:
        if cur_id == 100000:
            break
        if cur_id in emt_list:
            cur_id += 1
            continue
        else:
            seged_list = jieba.lcut(j_line['q_user'])
            temp_m = []
            for i in seged_list:
                if i in word_vectors:
                    temp_m.append(word_vectors[i])
            temp_m = np.array(temp_m)
            if temp_m.size == 0:
                cur_id += 1
                continue
            else:
                for a in j_line['a_suggestion']:
                    if a not in dic_a_acount:
                        dic_a_acount[a] = 1
                    else:
                        dic_a_acount[a] += 1
            cur_id += 1

    # for a in dic_a_acount:
    #     print '##############################'
    #     print a, dic_a_acount[a]
    count_total = 0
    cur_id = 0
    d_id = 0
    flag = 0
    prev = 0
    rewards = []
    dic_d_a = {}
    for j_line in data_cleaned:
        if cur_id == 100000:
            break
        if flag == 1:
            if 'a_suggestion' in j_line:
                tmp_arm = []
                tmp_d_id = []
                for sug in j_line['a_suggestion']:
                    tmp_sample = []
                    if sug in dic_y and sug in dic_a_acount:
                        if dic_a_acount[sug] >= 40:
                            #print sug, dic_y[sug], len(X)
                            tmp_sample.append(X[cur_id+1])
                            tmp_sample.append(X[cur_id])
                            tmp_sample.append(X[dic_y[sug]])
                            #print np.array(tmp_sample).shape
                            X_D.append(np.array(tmp_sample))
                            Y_D.append(dic_y[sug])
                            tmp_arm.append(dic_y[sug])
                            tmp_d_id.append(d_id)
                            d_id += 1
                            if prev == sug:
                                rewards.append(1)
                            else:
                                rewards.append(0)
                for idx in tmp_d_id:
                    dic_d_a[idx] = tmp_arm

        #if j_line['q_user'] in dic_a_acount:
        #print dic_id[cur_id] == dic_id[cur_id+1] == dic_id[cur_id+2], cur_id

        if dic_id[cur_id] == dic_id[cur_id+1] == dic_id[cur_id+2]:
            if j_line['q_user'] not in dic_a_acount:
                cur_id += 1
                flag = 0
                prev = 0
                continue
            else:
                #print dic_a_acount[j_line['q_user']]
                if dic_a_acount[j_line['q_user']] >= 5:
                    flag = 1
                    prev = j_line['q_user']
                    #print cur_id, j_line['q_user'], dic_a_acount[j_line['q_user']]
                else:
                    flag = 0
                    prev = 0
                cur_id += 1
        else:
            flag = 0
            prev = 0
            cur_id += 1

    print len(X_D), len(Y_D), rewards
    return X_D, Y_D, rewards, dic_d_a


def modelTrain(X, Y, params, size):
    train_data = np.array(X)
    Y = np_utils.to_categorical(Y, params['n_classes'])
    print train_data[0].shape[1]
    sub_input = Input(shape=(params['seq_max_len']*size, train_data[0].shape[1]))
    mask_input = Masking(mask_value = 0, input_shape = (params['seq_max_len'], train_data[0].shape[1]))(sub_input)
    sub_output, state_h, state_c = LSTM(output_dim=params['n_hidden'], return_state=True, return_sequences=False, consume_less = 'mem')(mask_input)
    drop_output = Dropout(params['dropout'])(sub_output)
    latentx = Dense(params['n_classes'] * params['n_latent'], activation = 'relu', bias = True if params['bias'] else False)(drop_output)
    y = Dense(params['n_classes'], bias = False)(latentx)#params['n_classes']
    y_act = Activation('softmax')(y) if params['is_clf'] else Activation('linear')(y)
    objective = params['loss'] if params['is_clf'] else 'mean_squared_error'
    metric = [acc] if params['is_clf'] else [rmse]
    model = Model(input = sub_input, output = y_act)
    model.compile(loss = objective, optimizer = RMSprop(lr = params['lr']), metrics = metric)

    model_E = Model(input = sub_input, output = [sub_output, state_h, state_c])


    checkpointer = ModelCheckpoint(filepath='test.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('test.csv')
    hist = model.fit(x = np.array(X), y = np.array(Y), batch_size = params['batch_size'], verbose = 2, \
        nb_epoch = params['n_epochs'], validation_split=params['validation_split'],shuffle=True,callbacks=[csv_logger,checkpointer])

    return model, model_E


def modelContext(X, Y, model, size):
    context = np.array(model.predict(np.array(X).reshape((len(X),params['seq_max_len']*size, X[0].shape[1]))))[2]
    print context.shape
    return context


def onlineLearning(contextD, contextA, rewards, dic_d_a):
    print rewards
    n_a=len(contextA)  # number of actions
    D= contextD
    n=len(D)  # number of data points
    k=len(D[0])   # number of features
    print k, n_a
    # our data
    th=np.random.random( (n_a,k) ) - 0.5      # our real theta, what we will try to guess/


    P=D.dot(th.T)
    print P[0]
    import matplotlib.pyplot as plt
    optimal=np.array(P.argmax(axis=1), dtype=int)
    plt.title("Distribution of ideal arm choices")
    plt.hist(optimal,bins=range(0,n_a))

    eps=0.1

    choices=np.zeros(n)
    explore=np.zeros(n)
    norms  =np.zeros(n)
    b      =np.zeros_like(th)
    A      =np.zeros( (n_a, k,k)  )
    #A = contextA
    for a in range (0,n_a):
        A[a]=np.identity(k)
    th_hat =np.zeros_like(th) # our temporary feature vectors, our best current guesses
    p      =np.zeros(n_a)
    alph   =0.2


    j = 0
    # LinUCB, using a disjoint model
    # This is all from Algorithm 1, p 664, "A contextual bandit approach..." Li, Langford
    tmp_index = -1
    prev = dic_d_a[0]
    regret = []
    for i in range(0,n):

        x_i = D[i]   # the current context vector
        print i, dic_d_a[i], rewards[i]

        if np.array_equal(prev, dic_d_a[i]) and tmp_index < len(dic_d_a[i]) - 1:
            tmp_index += 1
        else:
            prev = dic_d_a[i]
            tmp_index = 0

        #for a in range (0,n_a):
        for a in dic_d_a[i]:
            A_inv      = np.linalg.inv(A[a])        # we use it twice so cache it
            th_hat[a]  = A_inv.dot(b[a])            # Line 5
            ta         = x_i.dot(A_inv).dot(x_i)    # how informative is this ?
            a_upper_ci = alph * np.sqrt(ta)         # upper part of variance interval
            a_mean     = th_hat[a].dot(x_i)         # current estimate of mean
            p[a]       = a_mean + a_upper_ci        # top CI

        #norms[i]       = np.linalg.norm(th_hat - th,'fro')    # diagnostic, are we converging ?

        # Let's not be biased with tiebreaks, but add in some random noise
        p= p + ( np.random.random(len(p)) * 0.000001)
        p_tmp =[]
        for idx in dic_d_a[i]:
            p_tmp.append(p[idx])
        choices[i] = dic_d_a[i][np.array(p_tmp).argmax()]   # choose the highest, line 11
        a = int(choices[i])
        print a, rewards[i], tmp_index, dic_d_a[i][tmp_index]
        if rewards[i] == 1:
            if a != dic_d_a[i][tmp_index]:
                regret.append(1)
            else:
                regret.append(0)

        # see what kind of result we get
        #rewards[i] = th[a].dot(x_i)  # using actual theta to figure out reward

        # update the input vector
        A[a]      += np.outer(x_i,x_i)
        b[a]      += rewards[i] * x_i

    print len(regret), regret.count(1), regret.count(0)

    # print P.max(axis=1).shape
    # regret=(P.max(axis=1) - rewards)
    # plt.subplot(122)
    # plt.plot(np.array(regret).cumsum())
    # plt.title("Cumulative Regret")
    # plt.show()

#Main
count = 0
flag = 0
count_t = 0
question = {}
select = {}
topic = []

word_vectors = KeyedVectors.load_word2vec_format('userLogs_vectors.txt', binary=False)
labels = [u'\u57fa\u7840\u77e5\u8bc6', u'\u624b\u673a', u'7\u6d88\u8d39\u8005\u4e91\u670d\u52a1', u'1\u624b\u673afaq', u'\u8def\u7531\u5668', u'\u8d2d\u7269\u77e5\u8bc6', u'3\u8def\u7531\u5668faq', u'\u6d3b\u52a8\u77e5\u8bc6', u'\u624b\u73af', u'\u4ea7\u54c1\u8d2d\u4e70', u'\u5bb6\u5ead\u4ea7\u54c1\u8865\u5145', u'6\u5bb6\u5ead\u5a92\u4f53\u7ec8\u7aeffaq', u'2\u5e73\u677ffaq', u'\u670d\u52a1', u'8\u4ea7\u54c1\u901a\u7528\u77e5\u8bc6', u'\u9000\u6362\u8d27', u'\u624b\u8868', u'\u4ee3\u9500\u5546', u'5\u7a7f\u6234\u7c7b\u4ea7\u54c1faq', u'\u5e73\u677f', u'\u914d\u4ef6', u'\u5b98\u7f51', u'', u'4\u79fb\u52a8\u5bbd\u5e26\u4ea7\u54c1faq']
label_dic = {}
for i in range(len(labels)):
    label_dic[labels[i]] = i

params = {'t': 5,
          'seq_max_len': 20,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.1,
          'n_epochs': 2,
          'n_hidden': 8,
          'n_latent': 8,
          'bias': 1,
          'is_clf': 1,
          'validation_split': 0.2,
          'data_size': 10,
          'loss': 'categorical_crossentropy'}

dic_id = create_dic_id()
X, Y, count_id, dic_y, data_cleaned = load_data(word_vectors, dic_id, label_dic, params)
params['n_classes'] = len(labels)
modelX, modelX_E = modelTrain(X, Y, params, 1)
contextX = modelContext(X, Y,modelX_E, 1)

print len(contextX)
print modelX_E

dic_id_d = create_dic_id_d(data_cleaned)
X_D, Y_D, rewards, dic_d_a = load_data_2(contextX, count_id, dic_id_d, dic_y, data_cleaned, params)

print np.array(X_D).shape
params['n_classes'] = len(Y)
params['seq_max_len'] = 3
modelD, modelD_E = modelTrain(X_D, Y_D, params, 1)
# modelA, modelA_E = modelTrain(X_A, Y_A, params, 1)

contextD = modelContext(X_D, Y_D, modelD_E, 1)
# contextA = modelContext(X_A, Y_A, modelA_E, 1)

onlineLearning(contextD, data_cleaned, rewards, dic_d_a)

# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1])))).shape
# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1]))))[1]
# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1]))))[2]