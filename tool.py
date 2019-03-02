import numpy as np
import scipy.spatial.distance as ds
from collections import Counter
from sklearn.preprocessing import normalize as norm
from sklearn.metrics import roc_auc_score

def ave_auc(y, scores, class_num):
    average_auc = 0.0
    for i in range(class_num):
        label = np.zeros_like(y, dtype=np.int32)
        label[y == i] = 1
        average_auc += roc_auc_score(label, scores[:, i])
    return average_auc / class_num

''' get the max number index for a matrix '''
def argmax(x):
    dimension = len(x.shape)
    if dimension == 2:
         d1, d2= x.shape
         max_idx = np.argmax(acc)
         max_d1 = max_idx / d2
         max_d2 = max_idx % d2
         return max_d1, max_d2

    if dimension == 3:
         d1, d2, d3 = x.shape
         max_idx = np.argmax(acc)
         max_d1 = max_idx / (d2 * d3)
         max_d2 = max_idx % (d2 * d3) / d3
         max_d3 = max_idx % (d2 * d3) % d3
         return max_d1, max_d2, max_d3


''' store matrix into file '''
def dump(x, file_name):
    dimension = len(x.shape)
    f = open(file_name, 'w')

    if dimension == 2:
        for i in x:
            f.write('\t'.join([str(a) for a in i]))
            f.write('\n')
    if dimension == 3:
        for i in x:
            for j in i:
                f.write('\t'.join([str(a) for a in j]))
                f.write('\n')
            f.write('\n')

def power(m_list):
    return [2**x for x in m_list]


"""Compute softmax values for each sets of scores in x."""
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

''' replace nan and inf to 0 '''
def replace_nan(X):
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X

''' normalize matrix '''
def norm_matrix(x):
    replace_nan(x)
    for i in range(x.shape[0]):
        x[i] = norm(x[i], 'l2', 1)
    return x

''' normalize '''
def normalize(features):
    features = replace_nan(features)
    s = np.sqrt(np.sum(np.square(features), axis=1))
    return replace_nan(features / s[:,None])

def rank_hashtable(k):
    hashtable = {}
    hashtable[0] = 1
    for i in range(1, k):
        hashtable[i] = hashtable[i-1]+1/float(i+1)
    return hashtable

''' compute class label similarity '''
def compute_label_sim(sig_y1, sig_y2, sim_scale):
    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim

''' compute class label similarity '''
def Compute_Sim(Sig_Y, idx1, idx2, sim_scale):
    sig_y1 = Sig_Y[np.unique(idx1-1)]
    sig_y2 = Sig_Y[np.unique(idx2-1)]

    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim

''' normalize label embedding '''
def get_class_signatures(label, norm_method='L2'):
    if (norm_method == 'L2'):
        s = np.sqrt(np.sum(np.square(label), axis=1))
        Sig_Y = replace_nan(label / s[:,None])

    Dist = ds.cdist(Sig_Y, Sig_Y, 'euclidean')
    median_Dist = np.median(Dist[Dist>0])
    Sig_Y = replace_nan(Sig_Y / median_Dist)
    return Sig_Y

''' get accuracy '''
def evaluate_easy(predict_label, true_label):
    labels = np.unique(true_label)
    class_num = labels.shape[0]
    acc_arr = np.zeros(class_num)
    for i in range(class_num):
        idx = (true_label == labels[i])
        acc_arr[i] = np.sum(predict_label[idx] == labels[i]) / float(np.sum(idx))
        #print "class ", labels[i], np.sum(predict_label[idx] == labels[i]), np.sum(idx), acc_arr[i]

    return np.mean(acc_arr)

def evaluate_easy_2(Ypred, Ytrue):
    labels = np.unique(Ytrue)
    L = labels.shape[0]
    confusion = np.zeros((L, 1))
    for i in range(L):
        confusion[i] = float(np.sum(np.logical_and(Ytrue == labels[i], Ypred == labels[i]))) / np.sum(Ytrue == labels[i])
    acc = np.mean(confusion)
    acc2 = np.mean(Ypred == Ytrue)
    return acc2

def boost_acc(pred_list, true):
    pred = []
    length = len(true)
    for i in range(length):
        all_preds = [pred_list[0][i], pred_list[1][i], pred_list[2][i]]
        counts = Counter(all_preds)
        #print all_preds, counts.most_common(1)[0][0]
        pred.append(int(counts.most_common(1)[0][0]))
    pred = np.array(pred)
    acc = evaluate_easy(pred, true)
    return acc

def write_acc(f, acc_record):
    output = str(acc_record[0])+'\t'+str(acc_record[1])+'\t'+str(acc_record[2])+'\t'+str(sum(acc_record)/len(acc_record))+'\n'
    f.write(output)

def dump_result(config, acc_record, acc_test_unseen_record, acc_test_seen_record, unseen_ypred_record, seen_ypred_record, Ytest_unseen, Ytest_seen):
    method = config.get('model', 'method')
    lr = config.getfloat('model', 'learning_rate')
    file_name = './results/'+method+'/lr_'+str(lr)
    f = open(file_name, 'w')

    f.write("validation_acc\n")
    write_acc(f, acc_record)

    f.write("test_unseen_acc\n")
    write_acc(f, acc_test_unseen_record)

    f.write("test_seen_acc\n")
    write_acc(f, acc_test_seen_record)

    f.write("test_unseen acc\n"+str(boost_acc(unseen_ypred_record, Ytest_unseen))+'\n')
    f.write("test_seen acc\n"+str(boost_acc(seen_ypred_record, Ytest_seen))+'\n')
