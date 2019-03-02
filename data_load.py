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


def load_data(word_vectors, label_dic, params):
    X_q = []
    Y_q = []
    X_a = []
    Y_a = []
    count_a = 0
    count_q = -1
    dic_q_a_r = {}
    dic_q_a = {}
    dic_y = {}
    with open('userLogs.json', 'r') as file:
        y_id = 0
        for line in file:
            j_line = json.loads(line)
            if j_line['a_type'] != unicode('建议问'):
                temp_q = j_line['q_user']
                temp_id = j_line['user_id']
                temp_label = j_line['topic']
            if j_line['a_type'] == unicode('建议问'):
                if temp_q in j_line['a_suggestion']:
                    seged_list = jieba.lcut(j_line['q_user'])
                    temp_m = []
                    for i in seged_list:
                        if i in word_vectors:
                            temp_m.append(word_vectors[i])
                    temp_m = np.array(temp_m)
                    if temp_m.size == 0:
                        continue
                    else:
                        print j_line['q_user']
                        for a in j_line['a_suggestion']:
                            if a == temp_q and j_line['q_user'] not in dic_y:
                                    dic_y[j_line['q_user']] = y_id
                            if a not in dic_y:
                                dic_y[a] = y_id
                                y_id += 1
                    count_a += 1
                    if count_a == 100:
                        break
    # for i in dic_y:
    #     print i, dic_y[i]
    count_a = 0
    with open('userLogs.json', 'r') as file1:
        for line in file1:
            j_line = json.loads(line)
            if j_line['a_type'] != unicode('建议问'):
                temp_q = j_line['q_user']
                temp_id = j_line['user_id']
                temp_label = j_line['topic']
            if j_line['a_type'] == unicode('建议问'):
                if temp_q in j_line['a_suggestion']:
                    # if temp_lable not in topic:
                    #     topic.append(temp_lable)
                    # jieba
                    # print j_line['q_user']
                    seged_list = jieba.lcut(j_line['q_user'])
                    temp_m = []
                    for i in seged_list:
                        # print i
                        if i in word_vectors:
                            temp_m.append(word_vectors[i])
                    temp_m = np.array(temp_m)
                    if temp_m.size == 0:
                        # print j_line['q_user']
                        # print seged_list
                        continue
                    else:
                        if temp_m.shape[0] < params['seq_max_len']:
                            tmp_res = np.zeros((params['seq_max_len'],200))
                            tmp_res[:temp_m.shape[0],:temp_m.shape[1]] = temp_m
                        else:
                            tmp_res = temp_m[:params['seq_max_len'],:]
                        X_q.append(np.array(tmp_res))
                        print j_line['q_user']
                        Y_q.append(dic_y[j_line['q_user']])
                        #print np.array(temp_m).shape
                        # jianyiwen
                        print count_a
                        id_a_array = []
                        for a in j_line['a_suggestion']:
                            count_q += 1
                            id_a_array.append(count_q)
                            # print a
                            if a == temp_q:
                                dic_q_a_r[count_a] = count_q
                                # print temp_q
                            seged_list = jieba.lcut(a)
                            temp_m = []
                            for i in seged_list:
                                # print i
                                if i in word_vectors:
                                    temp_m.append(word_vectors[i])
                            temp_m = np.array(temp_m)
                            if temp_m.shape[0] < params['seq_max_len']:
                                tmp_res = np.zeros((params['seq_max_len'],200))
                                tmp_res[:temp_m.shape[0],:temp_m.shape[1]] = temp_m
                            else:
                                tmp_res = temp_m[:params['seq_max_len'],:]
                            X_a.append(np.array(tmp_res))
                            Y_a.append(dic_y[a])
                    dic_q_a[count_a] = id_a_array
                    count_a += 1
                    if count_a == 100:
                        break
    # print len(X), len(Y)
    #
    # for i in range(len(X)):
    #     print np.array(X[i]).shape, Y[i]
    for i in dic_q_a_r:
        print i, dic_q_a_r[i]
    print len(X_q), len(Y_q), len(X_a), len(Y_a)
    return X_q, Y_q, X_a, Y_a, dic_q_a_r, dic_q_a, len(dic_y)


def modelTrain(X, Y, params):
    train_data = np.array(X)
    Y = np_utils.to_categorical(Y, params['n_classes'])
    print train_data[0].shape[1]
    sub_input = Input(shape=(params['seq_max_len'], train_data[0].shape[1]))
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

def modelContext(X, Y, model):
    context = np.array(model.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1]))))[2]
    print context.shape
    return context

def onlineLearning(contextQ, contextA, dic_q_a_r, dic_q_a):

    q=len(contextQ)  # number of data points
    n_a=len(contextA)  # number of actions

    D=[]
    for i in range(0,q):
        #for a in range (0,n_a):
        for j in range (dic_q_a[i][0],dic_q_a[i][0]+len(dic_q_a[i])):
            D.append(np.concatenate((contextQ[i],contextA[j]), axis=0))
    D = np.array(D)
    n=len(D)  # number of data points
    k=len(D[0])   # number of features
    print k, n_a
        # our data
    dic_rewards = dic_q_a_r
    th=np.random.random( (n_a,k) ) - 0.5      # our real theta, what we will try to guess/


    P=D.dot(th.T)
    print P[0]
    import matplotlib.pyplot as plt
    optimal=np.array(P.argmax(axis=1), dtype=int)
    plt.title("Distribution of ideal arm choices")
    plt.hist(optimal,bins=range(0,n_a))

    eps=0.1

    choices=np.zeros(n)
    rewards=np.zeros(n)
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
    for i in range(0,n):

        x_i = D[i]   # the current context vector

        print dic_q_a[j][0],dic_q_a[j][0]+len(dic_q_a[j])
        #for a in range (0,n_a):
        for a in range (dic_q_a[j][0],dic_q_a[j][0]+len(dic_q_a[j])):
            A_inv      = np.linalg.inv(A[a])        # we use it twice so cache it
            print A_inv.shape
            th_hat[a]  = A_inv.dot(b[a])            # Line 5
            ta         = x_i.dot(A_inv).dot(x_i)    # how informative is this ?
            a_upper_ci = alph * np.sqrt(ta)         # upper part of variance interval
            a_mean     = th_hat[a].dot(x_i)         # current estimate of mean
            p[a]       = a_mean + a_upper_ci        # top CI

        #norms[i]       = np.linalg.norm(th_hat - th,'fro')    # diagnostic, are we converging ?

        # Let's not be biased with tiebreaks, but add in some random noise
        p= p + ( np.random.random(len(p)) * 0.000001)
        print p.shape, p
        p_choice = []
        print len(dic_q_a), j
        print dic_q_a[j]
        for c in dic_q_a[j]:
            p_choice.append(p[c])
        print p_choice, dic_rewards[j]
        #choices[i] = p.argmax()   # choose the highest, line 11
        choices[i] = np.array(p_choice).argmax()   # choose the highest, line 11
        a = int(choices[i])
        print a, dic_rewards[j]-dic_q_a[j][0]
        # see what kind of result we get
        print 'reward:', j, i, dic_rewards[j], dic_q_a[j][0], a
        if dic_rewards[j]-dic_q_a[j][0] == a:
            print 'reward:', j, i
            rewards[i] = 1
        else:
            rewards[i] = 0
            #rewards[i] = th[a].dot(x_i)  # using actual theta to figure out reward

        # update the input vector
        A[a]      += np.outer(x_i,x_i)
        b[a]      += rewards[i] * x_i

        if i == dic_q_a[j][0] + len(dic_q_a[j]) - 1:
            j += 1

    plt.figure(1,figsize=(10,5))
    plt.subplot(121)
    plt.plot(norms);
    plt.title("Frobenius norm of estimated theta vs actual")
    plt.show()

    print P.max(axis=1).shape
    print rewards.shape
    regret=(P.max(axis=1) - rewards)
    plt.subplot(122)
    plt.plot(regret.cumsum())
    plt.title("Cumulative Regret")
    plt.show()

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
          'idx': 1, # 1: dnn, 2: dfm, 3: dmvm
          'loss': 'categorical_crossentropy'}

X_q, Y_q, X_a, Y_a, dic_q_a_r, dic_q_a, len_Y = load_data(word_vectors, label_dic, params)
params['n_classes'] = len_Y
modelQ, modelQ_E = modelTrain(X_q, Y_q, params)
modelA, modelA_E = modelTrain(X_a, Y_a, params)

contextQ = modelContext(X_q, Y_q,modelQ_E)
contextA = modelContext(X_a, Y_a, modelA_E)

onlineLearning(contextQ, contextA, dic_q_a_r, dic_q_a)

# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1])))).shape
# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1]))))[1]
# print np.array(model2.predict(np.array(X).reshape((len(X),params['seq_max_len'], X[0].shape[1]))))[2]