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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pprint import pprint
import math


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def sequence2vec(qustion, word2vec, params):
    tmp = {}
    tmp['matrix'] = []
    question_words = jieba.lcut(qustion)
    for i in question_words:
        # print i
        if i in word2vec:
            tmp['matrix'].append(word2vec[i])
    tmp['matrix'] = np.array(tmp['matrix'])
    if tmp['matrix'].size == 0:
        tmp['result_matrix'] = []
    else:
        if tmp['matrix'].shape[0] < params['seq_max_len']:
            tmp['result_matrix'] = np.zeros((params['seq_max_len'], 200))
            tmp['result_matrix'][:tmp['matrix'].shape[0], :tmp['matrix'].shape[1]] = tmp['matrix']
        else:
            tmp['result_matrix'] = tmp['matrix'][:params['seq_max_len'], :]
    return tmp['result_matrix']


def load_data(word2vec, label_dic, params):
    X_D = []
    Y_D = []
    X_A = []
    Y_A = []
    rewards = []
    dic = {}
    temp = {}
    dic['count_a_suggestion'] = {}
    dic['result_suggestion_matrix'] = {}
    dic['y'] = {}
    dic_d_a = {}
    f = open('./data/single-round_' + str(params['count']) + '.txt','w')

    # Build suggestion dictionary and count numbers
    with open('userLogs_large.json', 'r') as file:
        for line in file:
            j_line = json.loads(line)
            # If record is not suggest question, skip and save temp
            if j_line['a_type'] != unicode('建议问'):
                temp['q_user'] = j_line['q_user']
                temp['user_id'] = j_line['user_id']
                temp['topic'] = j_line['topic']
            # If record is a suggest question, check and save valid result
            if j_line['a_type'] == unicode('建议问'):
                if temp['q_user'] in j_line['a_suggestion']:
                    temp['matrix'] = []
                    question_words = jieba.lcut(j_line['q_user'])
                    for word in question_words:
                        if word in word2vec:
                            temp['matrix'].append(word2vec[word])
                    temp['matrix'] = np.array(temp['matrix'])
                    if temp['matrix'].size == 0:
                        continue
                    else:
                        for suggestion in j_line['a_suggestion']:
                            if suggestion not in dic['count_a_suggestion']:
                                    dic['count_a_suggestion'][suggestion] = 1
                            else:
                                if suggestion == temp['q_user']:
                                    dic['count_a_suggestion'][suggestion] += 1
                    # count_total += 1
                    # if count_total == 1000:
                    #     break

    count_total = 0
    with open('userLogs_large.json', 'r') as file1:
        y_id = 0
        d_id = 0
        for line in file1:
            j_line = json.loads(line)
            if j_line['a_type'] != unicode('建议问'):
                temp['q_user'] = j_line['q_user']
                temp['user_id'] = j_line['user_id']
                temp['topic'] = j_line['topic']
            if j_line['a_type'] == unicode('建议问'):
                if temp['q_user'] in j_line['a_suggestion']:
                    # if temp_lable not in topic:
                    #     topic.append(temp_lable)
                    # jieba
                    # print j_line['q_user']
                    temp['result_question_matrix'] = sequence2vec(j_line['q_user'], word2vec, params)
                    if len(temp['result_question_matrix']) == 0:
                        continue
                    else:
                        # jianyiwen
                        d_id_array = []
                        a_id_array = []
                        for suggestion in j_line['a_suggestion']:
                            if dic['count_a_suggestion'][suggestion] < params['count']:
                                continue
                            # print a
                            # print temp_q
                            if suggestion not in dic['y']:
                                temp['result_suggestion_matrix'] = sequence2vec(suggestion, word2vec, params)
                                dic['y'][suggestion] = y_id
                                y_id += 1
                                dic['result_suggestion_matrix'][suggestion] = temp['result_suggestion_matrix']
                                X_A.append(temp['result_suggestion_matrix'])
                                Y_A.append(dic['y'][suggestion])
                            else:
                                temp['result_suggestion_matrix'] = dic['result_suggestion_matrix'][suggestion]

                            if suggestion == temp['q_user']:
                                rewards.append(1)
                            else:
                                rewards.append(0)
                            X_D.append(np.concatenate((temp['result_question_matrix'], temp['result_suggestion_matrix']), axis=0))
                            Y_D.append(dic['y'][suggestion])
                            q_words = jieba.lcut(j_line['q_user'])
                            s_words = jieba.lcut(suggestion)
                            str_saver = ''
                            str_saver = str_saver + str(dic['y'][suggestion]) + '\t'
                            for word in q_words:
                                str_saver += word
                                str_saver += ' '
                            for word in s_words:
                                str_saver += word
                                str_saver += ' '
                            str_saver += '\n'
                            f.write(str_saver)
                            d_id_array.append(d_id)
                            d_id += 1
                            a_id_array.append(dic['y'][suggestion])
                            #print j_line['q_user'], temp_q, a, dic_y[a] #tmp_res_d[0], tmp_res_a[0]
                        # print d_id_array, a_id_array
                        for idx in d_id_array:
                            dic_d_a[idx] = a_id_array

                    count_total += 1
                    # if count_total == 1000:
                    #     break
    # print len(X), len(Y)
    #
    # for i in range(len(X)):
    #     print np.array(X[i]).shape, Y[i]
    f.close()
    print rewards
    print len(X_D), len(Y_D), len(X_A), len(Y_A)
    return X_D, Y_D, X_A, Y_A, rewards, dic_d_a, len(X_A)


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
    model.fit(x = np.array(X), y = np.array(Y), batch_size = params['batch_size'], verbose=2,
                nb_epoch = params['n_epochs'], validation_split=params['validation_split'], shuffle=True,
                callbacks=[csv_logger,checkpointer])

    return model, model_E


def modelContext(X, Y, model, size):
    context = np.array(model.predict(np.array(X).reshape((len(X),params['seq_max_len']*size, X[0].shape[1]))))[2]
    print context.shape
    return context


def onlineLearning(contextD, arm_num, rewards, dic_d_a, params):
    print rewards
    n_a= arm_num#len(contextA)  # number of actions
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

    eps=params['eps']
    choices=np.zeros(n)
    b      =np.zeros_like(th)
    A      =np.zeros( (n_a, k,k)  )
    for a in range (0,n_a):
        A[a]=np.identity(k)
    th_hat =np.zeros_like(th) # our temporary feature vectors, our best current guesses
    p      =np.zeros(n_a)
    alph   =0.2

    j = 0.1
    # LinUCB, using a disjoint model
    # This is all from Algorithm 1, p 664, "A contextual bandit approach..." Li, Langford
    tmp_index = -1
    prev = dic_d_a[0]
    regret = []
    for i in range(0,n):
        x_i = D[i]   # the current context vector
        # print i, dic_d_a[i], rewards[i]
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

        # Let's not be biased with tiebreaks, but add in some random noise
        p= p + ( np.random.random(len(p)) * 0.000001)
        p_tmp =[]
        for idx in dic_d_a[i]:
            p_tmp.append(p[idx])
        if random.uniform(0, 1) > eps:
            choices[i] = dic_d_a[i][np.array(p_tmp).argmax()]
        else:
            choices[i] = random.choice(dic_d_a[i]) # choose the highest, line 11
        a = int(choices[i])
        # print a, dic_d_a[i][tmp_index], rewards[i], tmp_index
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

    # print P.max(axis=1).shape
    # regret=(P.max(axis=1) - rewards)
    print len(regret), regret.count(1), regret.count(0)
    # plt.subplot(122)
    # plt.plot(np.array(regret).cumsum())
    # plt.title("Cumulative Regret")
    # plt.show()

def UCB(rewards, dic_d_a, params):
    sample_num = len(rewards)
    score = {}
    count = {}
    regret = []
    choices=np.zeros(sample_num)
    for i in range(sample_num):
        for item in dic_d_a[i]:
            score[item] = 1
            count[item] = 1

    n = 1
    eps = params['eps']
    prev = dic_d_a[0]
    tmp_index = -1
    for i in range(sample_num):
        if np.array_equal(prev, dic_d_a[i]) and tmp_index < len(dic_d_a[i]) - 1:
            tmp_index += 1
        else:
            prev = dic_d_a[i]
            tmp_index = 0

        tmp_scores = []
        for item in dic_d_a[i]:
            tmp_scores.append(score[item] + math.sqrt(2*math.log(n))/count[item])

        if random.uniform(0, 1) > eps:
            choices[i] = dic_d_a[i][np.array(tmp_scores).argmax()]
        else:
            choices[i] = random.choice(dic_d_a[i]) # choose the highest, line 11

        a = int(choices[i])
        # print a, dic_d_a[i][tmp_index], rewards[i], tmp_index
        if rewards[i] == 1:
            if a != dic_d_a[i][tmp_index]:
                regret.append(1)
                count[a] += 1
            else:
                score[a] += 1
                count[a] += 1
                regret.append(0)
            n += 1

    print len(regret), regret.count(1), regret.count(0)

# Main()
count = 0
flag = 0
count_t = 0
question = {}
select = {}
topic = []

WORD2VEC = KeyedVectors.load_word2vec_format('userLogs_vectors.txt', binary=False)
labels = [u'\u57fa\u7840\u77e5\u8bc6', u'\u624b\u673a', u'7\u6d88\u8d39\u8005\u4e91\u670d\u52a1', u'1\u624b\u673afaq', u'\u8def\u7531\u5668', u'\u8d2d\u7269\u77e5\u8bc6', u'3\u8def\u7531\u5668faq', u'\u6d3b\u52a8\u77e5\u8bc6', u'\u624b\u73af', u'\u4ea7\u54c1\u8d2d\u4e70', u'\u5bb6\u5ead\u4ea7\u54c1\u8865\u5145', u'6\u5bb6\u5ead\u5a92\u4f53\u7ec8\u7aeffaq', u'2\u5e73\u677ffaq', u'\u670d\u52a1', u'8\u4ea7\u54c1\u901a\u7528\u77e5\u8bc6', u'\u9000\u6362\u8d27', u'\u624b\u8868', u'\u4ee3\u9500\u5546', u'5\u7a7f\u6234\u7c7b\u4ea7\u54c1faq', u'\u5e73\u677f', u'\u914d\u4ef6', u'\u5b98\u7f51', u'', u'4\u79fb\u52a8\u5bbd\u5e26\u4ea7\u54c1faq']
label_dic = {}
for i in range(len(labels)):
    label_dic[labels[i]] = i

params = {'count': 40,
          'eps': 0,
          't': 5,
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

print("------------------read datasets begin-------------------")
X_D, Y_D, X_A, Y_A, rewards, dic_d_a, len_Y = load_data(WORD2VEC, label_dic, params)
# print("------------------UCB begin-------------------")
# UCB(rewards, dic_d_a, params)
# print("------------------UCB End-------------------")

# params['n_classes'] = len_Y
# print("------------------Train Embedding begin-----------------")
# modelD, modelD_E = modelTrain(X_D, Y_D, params, 2)
# modelA, modelA_E = modelTrain(X_A, Y_A, params, 1)
#
# contextD = modelContext(X_D, Y_D,modelD_E, 2)
# contextA = modelContext(X_A, Y_A, modelA_E, 1)
#
# print contextD.shape
# print contextA.shape
# print("------------------Online Learning begin-----------------")
# onlineLearning(contextD, len(contextA), rewards, dic_d_a, params)

# print("------------------Capsule Learning begin-----------------")
# contextD = np.load('./data/single_capsule_' + str(params['count']) + '.npy')
# onlineLearning(contextD, 53, rewards, dic_d_a, params)

