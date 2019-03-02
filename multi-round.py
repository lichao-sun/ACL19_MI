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
    dic, cur, last = {}, {}, {}
    dic['count_a_suggestion'] = {}
    dic['result_matrix'] = {}
    dic['question_label_id'] = {}
    with open('userLogs_large.json', 'r') as file:
        last['l0'], last['l1'], last['l2'], last_session = '', '', '', ''
        sum = 0
        session_skip = 0
        for line in file:
            j_line = json.loads(line)
            current_session = j_line['session_id']
            cur['l0'], cur['l1'], cur['l2'] = j_line, last['l0'], last['l1']
            last['l0'], last['l1'], last['l2'] = cur['l0'], cur['l1'], cur['l2']

            # current_session != last_session and
            if current_session == last_session:
                if session_skip < params['skip']:
                    session_skip += 1
                else:
                    session_skip = 0
                    last_session = current_session
                    continue
            else:
                session_skip = 0
            if 'a_suggestion' in cur['l0'] and 'a_suggestion' in cur['l1'] \
                and cur['l1']['q_user'] in cur['l0']['a_suggestion'] and cur['l2']['q_user'] in cur['l1']['a_suggestion']:
                # print last['l1']['q_user']
                sum += 1
                # if last['l1']['q_user'] != '':
                # if cur['l1']['q_user'] not in dic['count_a_suggestion']:
                #     dic['count_a_suggestion'][cur['l1']['q_user']] = 1
                # else:
                #     dic['count_a_suggestion'][cur['l1']['q_user']] += 1
                # if last['l2']['q_user'] != '':
                if cur['l2']['q_user'] not in dic['count_a_suggestion']:
                    dic['count_a_suggestion'][cur['l2']['q_user']] = 1
                else:
                    dic['count_a_suggestion'][cur['l2']['q_user']] += 1

                # for item in cur['l1']['a_suggestion']:
                #     if item not in dic['count_a_suggestion']:
                #         dic['count_a_suggestion'][item] = 1
                #     else:
                #         dic['count_a_suggestion'][item] += 1
            last_session = current_session
        print 'sum:', sum
        # count = 0
        # for item in dic['count_a_suggestion']:
        #     if dic['count_a_suggestion'][item] >= params['count']:
        #         print item, dic['count_a_suggestion'][item]
        #         count += 1
        # print count

    with open('userLogs_large.json', 'r') as file:
        y_id = 0
        d_id = 0
        X = []
        Y = []
        dic['question_data_id'] = {}
        session_skip = 0
        for line in file:
            j_line = json.loads(line)
            current_session = j_line['session_id']
            cur['l0'], cur['l1'], cur['l2'] = j_line, last['l0'], last['l1']
            last['l0'], last['l1'], last['l2'] = cur['l0'], cur['l1'], cur['l2']

            sample = []
            # if current_session == last_session:
            #     count += 1
            # current_session != last_session and
            if current_session == last_session:
                if session_skip < params['skip']:
                    session_skip += 1
                else:
                    session_skip = 0
                    last_session = current_session
                    continue
            else:
                session_skip = 0
            if 'a_suggestion' in cur['l0'] and 'a_suggestion' in cur['l1'] \
                and cur['l1']['q_user'] in cur['l0']['a_suggestion'] and cur['l2']['q_user'] in cur['l1']['a_suggestion']:
                    if dic['count_a_suggestion'][cur['l2']['q_user']] >= params['count']:
                        dic['result_matrix']['l0'] = sequence2vec(cur['l0']['q_user'], word2vec, params)
                        dic['result_matrix']['l1'] = sequence2vec(cur['l1']['q_user'], word2vec, params)
                        dic['result_matrix']['l2'] = sequence2vec(cur['l2']['q_user'], word2vec, params)
                        # print len(dic['result_matrix']['l0']), len(dic['result_matrix']['l1']), len(dic['result_matrix']['l2'])
                        if len(dic['result_matrix']['l0']) * len(dic['result_matrix']['l1']) \
                                * len(dic['result_matrix']['l2']) > 0:
                            if cur['l2']['q_user'] not in dic['question_label_id']:
                                dic['question_label_id'][cur['l2']['q_user']] = y_id
                                X.append(dic['result_matrix']['l2'])
                                Y.append(y_id)
                                dic['question_data_id'][cur['l2']['q_user']] = d_id
                                d_id += 1
                                y_id += 1
                                # dic['question_label_id'][cur['l0']['q_user']] = y_id
                                X.append(dic['result_matrix']['l0'])
                                Y.append(y_id)
                                dic['question_data_id'][cur['l0']['q_user']] = d_id
                                d_id += 1
                                # y_id += 1
                                # dic['question_label_id'][cur['l1']['q_user']] = y_id
                                X.append(dic['result_matrix']['l1'])
                                Y.append(y_id)
                                dic['question_data_id'][cur['l1']['q_user']] = d_id
                                d_id += 1
                                # y_id += 1
                            else:
                                y_tmp_id = dic['question_label_id'][cur['l2']['q_user']]
                                # dic['question_label_id'][cur['l0']['q_user']] = y_id
                                X.append(dic['result_matrix']['l0'])
                                Y.append(y_tmp_id)
                                dic['question_data_id'][cur['l0']['q_user']] = d_id
                                d_id += 1
                                # y_id += 1
                                # dic['question_label_id'][cur['l1']['q_user']] = y_id
                                X.append(dic['result_matrix']['l1'])
                                Y.append(y_tmp_id)
                                dic['question_data_id'][cur['l1']['q_user']] = d_id
                                d_id += 1

                            for item in cur['l1']['a_suggestion']:
                                if item in dic['count_a_suggestion']:
                                    if item not in dic['question_label_id'] and cur['l2']['q_user'] != item \
                                            and dic['count_a_suggestion'][item] >= params['count']:
                                        dic['result_matrix']['item'] = sequence2vec(item, word2vec, params)
                                        if len(dic['result_matrix']['item']) > 0:
                                                dic['question_label_id'][item] = y_id
                                                X.append(dic['result_matrix']['item'])
                                                Y.append(y_id)
                                                dic['question_data_id'][item] = d_id
                                                d_id += 1
                                                y_id += 1
            last_session = current_session
        print len(X), len(Y), max(Y), y_id, len(dic['question_data_id'])
    return X, Y, dic['count_a_suggestion'], dic['question_label_id'], dic['question_data_id'], y_id + 1


def create_input(contextX, dic_count_question, dic_question_label_id, dic_question_data_id, params):
    dic, cur, last = {}, {}, {}
    dic['count_a_suggestion'] = dic_count_question
    dic['result_matrix'] = {}
    dic['question_label_id'] = dic_question_label_id
    dic['question_data_id'] = dic_question_data_id
    last['l0'], last['l1'], last['l2'], last_session = '', '', '', ''

    with open('userLogs_large.json', 'r') as file:
        y_id = 0
        d_id = 0
        X = []
        Y = []
        rewards = []
        dic_d_a = {}
        session_skip = 0
        for line in file:
            j_line = json.loads(line)
            current_session = j_line['session_id']
            cur['l0'], cur['l1'], cur['l2'] = j_line, last['l0'], last['l1']
            last['l0'], last['l1'], last['l2'] = cur['l0'], cur['l1'], cur['l2']

            d_id_array = []
            a_id_array = []
            if current_session == last_session:
                if session_skip < params['skip']:
                    session_skip += 1
                else:
                    session_skip = 0
                    last_session = current_session
                    continue
            else:
                session_skip = 0
            if 'a_suggestion' in cur['l0'] and 'a_suggestion' in cur['l1'] \
                and cur['l1']['q_user'] in cur['l0']['a_suggestion'] \
                and cur['l2']['q_user'] in cur['l1']['a_suggestion'] \
                and cur['l0']['q_user'] in dic['question_data_id'] and \
                cur['l1']['q_user'] in dic['question_data_id'] and \
                cur['l2']['q_user'] in dic['question_data_id']:
                    for item in cur['l1']['a_suggestion']:
                        if item in dic['count_a_suggestion'] and item in dic['question_data_id'] and \
                                        item in dic['question_label_id'] and \
                                        dic['count_a_suggestion'][item] >= params['count']:
                            if cur['l2']['q_user'] == item:
                                rewards.append(1)
                            else:
                                rewards.append(0)
                            sample = []
                            sample.append(contextX[dic['question_data_id'][cur['l0']['q_user']]])
                            sample.append(contextX[dic['question_data_id'][cur['l1']['q_user']]])
                            sample.append(contextX[dic['question_data_id'][item]])
                            X.append(sample)
                            Y.append(dic['question_label_id'][item])
                            d_id_array.append(d_id)
                            d_id += 1
                            a_id_array.append(dic['question_label_id'][item])
                    for idx in d_id_array:
                        dic_d_a[idx] = a_id_array
            last_session = current_session
    return X, Y, rewards, dic_d_a


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


def onlineLearning(contextD, y_id, rewards, dic_d_a, para):#contextA
    print rewards
    n_a=y_id #len(contextA)  # number of actions
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

        #norms[i]       = np.linalg.norm(th_hat - th,'fro')    # diagnostic, are we converging ?

        # Let's not be biased with tiebreaks, but add in some random noise
        p = p + ( np.random.random(len(p)) * 0.000001)
        p_tmp =[]
        for idx in dic_d_a[i]:
            p_tmp.append(p[idx])
        if random.uniform(0, 1) > eps:
            choices[i] = dic_d_a[i][np.array(p_tmp).argmax()]
        else:
            choices[i] = random.choice(dic_d_a[i]) # choose the highest, line 11
        a = int(choices[i])
        # print a, rewards[i], tmp_index, dic_d_a[i][tmp_index]
        # if rewards[i] == 1:
        #     if a != dic_d_a[i][tmp_index]:
        #         regret.append(1)
        #     else:
        #         regret.append(0)
        if len(dic_d_a[i]) > 1:
            if rewards[i] == 1:
                if a != dic_d_a[i][tmp_index]:
                    regret.append(1)
                else:
                    regret.append(0)
        elif len(dic_d_a[i]) == 1:
            if rewards[i] == 1:
                regret.append(0)
            else:
                regret.append(1)

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
    prev = dic_d_a[0]
    eps = params['eps']
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
        if len(dic_d_a[i]) > 1:
            if rewards[i] == 1:
                if a != dic_d_a[i][tmp_index]:
                    regret.append(1)
                else:
                    regret.append(0)
                    score[a] += 1
        elif len(dic_d_a[i]) == 1:
            if rewards[i] == 1:
                regret.append(0)
                score[a] += 1
            else:
                regret.append(1)
        count[a] += 1
        n += 1

    print len(regret), regret.count(1), regret.count(0)

if __name__ == "__main__":
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

    params = {'eps': 0.3,
              'count': 5,
              'skip': 1,
              't': 5,
              'seq_max_len': 20,
              'batch_size': 128,
              'lr': 0.001,
              'dropout': 0.1,
              'n_epochs': 5,
              'n_hidden': 8,
              'n_latent': 8,
              'bias': 1,
              'is_clf': 1,
              'validation_split': 0.2,
              'data_size': 10,
              'loss': 'categorical_crossentropy'}

    X, Y, dic_count_question, dic_question_label_id, dic_question_data_id, y_id = load_data(word_vectors, label_dic, params)
    params['n_classes'] = y_id
    modelX, modelX_E = modelTrain(X, Y, params, 1)
    contextX = modelContext(X, Y,modelX_E, 1)

    print len(contextX)
    print modelX_E

    print("------------------read datasets begin-------------------")
    X_D, Y_D, rewards, dic_d_a = create_input(contextX, dic_count_question, dic_question_label_id, dic_question_data_id, params)
    # print("------------------UCB begin-------------------")
    # UCB(rewards, dic_d_a, params)
    # print("------------------UCB End---------------------")
    print np.array(X_D).shape
    params['n_classes'] = y_id
    params['seq_max_len'] = 3
    modelD, modelD_E = modelTrain(X_D, Y_D, params, 1)
    # modelA, modelA_E = modelTrain(X_A, Y_A, params, 1)

    contextD = modelContext(np.array(X_D), Y_D, modelD_E, 1)
    # contextA = modelContext(X_A, Y_A, modelA_E, 1)
    print("------------------Contextual begin-------------------")
    onlineLearning(contextD, y_id, rewards, dic_d_a, params)
    print("------------------Contextual End---------------------")
