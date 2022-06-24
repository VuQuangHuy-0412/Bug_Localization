from math import ceil
import numpy as np
from random import shuffle
from datas import DATASET
from util import csv2dict, topk_accuarcy
#from joblib import Parallel, delayed
import pickle
#from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
import tensorflow as tf

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Conv2D, Conv1D, Flatten, MaxPool2D
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn import model_selection
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.svm import SVC


def oversample(samples):
    """ Oversamples the features for label "1" 
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    samples_ = []

    # oversample features of buggy files
    for i, sample in enumerate(samples):
        samples_.append(sample)
        if i % 301 == 0:
            for _ in range(99):
                samples_.append(sample)

    return samples_


def focal_loss(gamma=2., alpha=.9):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def features_and_labels(samples):
    """ Returns features and labels for the given list of samples
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    features = np.zeros((len(samples), 6))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        features[i][5] = float(sample['semantic_similarity'])
        labels[i]      = float(sample["match"])

    return features, labels


def kfold_split_indexes(k, len_samples):
    """ Returns list of tuples for split start(inclusive) and 
        finish(exclusive) indexes.
    
    Arguments:
        k {integer} -- the number of folds
        len_samples {interger} -- the length of the sample list
    """
    step = ceil(len_samples / k)
    ret_list = [(start, start + step) for start in range(0, len_samples, step)]

    return ret_list


def kfold_split(bug_reports, i, k):
    """ Returns train samples and bug regports for test
    
    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
    """
    total_list = list(range(0, i)) + list(range(i + 1, k))
    test_path = str(DATASET.results / DATASET.name) + 'test' + str(i) + '.csv'
    test_samples = csv2dict(test_path)
    total_1 = total_0 = 0
    topk_wrong = 300
    train_data_1, train_data_0 = [], []
    index_fold = 0
    if DATASET.name == "aspectj":
      fold_train = 2
    else:
      fold_train = 9

    for j, train_id in enumerate(total_list):
        if index_fold == fold_train:
            break
        else:
            train_path = str(DATASET.results / DATASET.name) + 'train' + str(train_id) + '.csv'
            train_sample = csv2dict(train_path)
            for sample in train_sample:
                if int(sample["match"]) == 1:
                    total_1 += 1
                    train_data_1.append(sample)
                else:
                    total_0 += 1
                    train_data_0.append(sample)
            index_fold = index_fold + 1

    N = total_1
    Sn = total_0 // N
    batch = Sn + 1
    train_data = []
    for i, value in enumerate(train_data_1):
        train_data.append(value)
        for j in range(i * Sn, i * Sn + Sn):
            train_data.append(train_data_0[j])

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports.values() if br.bug_id in test_br_ids]

    return train_data, test_bug_reports, batch


def train_model(i, k, bug_reports, only_rvsm=False):
    """ Trains the dnn model and calculates top-k accuarcies
    
    Arguments:
        i {interger} -- current fold number for printing information
        num_folds {integer} -- total fold number for printing information
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        bug_reports {list of dictionaries} -- list of all bug reports
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv" pairs
    """
    train_samples, test_bug_reports, batch = kfold_split(bug_reports, i, k)
    shuffle(train_samples)
    shuffle(test_bug_reports)
    X_train, y_train = features_and_labels(train_samples)

    if not only_rvsm:
      # define model keras model
      clf = Sequential()
      clf.add(Dense(300, activation='relu', input_dim=6))
      clf.add(Dropout(0.2))
      clf.add(Dense(150, activation='relu'))
      clf.add(Dropout(0.2))
      # clf.add(Dense(50, activation='relu'))
      clf.add(Dense(1, activation='sigmoid'))
      clf.summary()

      # compline the keras model
      #clf.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
      # clf.compile(loss=BinaryFocalLoss(gamma=2), optimizer='sgd', metrics=['accuracy'])
      clf.compile(loss=focal_loss(), optimizer='sgd', metrics=['accuracy'])

      # fit model
      clf.fit(X_train, y_train.ravel(), epochs=30, batch_size=batch, verbose=1)


      # clf = MLPRegressor(
      #     solver="sgd",
      #     alpha=1e-5,
      #     hidden_layer_sizes=(300,),
      #     random_state=1,
      #     max_iter=10000,
      #     n_iter_no_change=30,
      # )
      # clf.fit(X_train, y_train.ravel())


      #clf = MLPRegressor(
      #    solver="sgd",
      #    alpha=1e-5,
      #    hidden_layer_sizes=(300,),
      #    random_state=1,
      #    max_iter=10000,
      #    n_iter_no_change=30,
      #)


      ## MLP Classifier
      # clf = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter=300,activation = 'relu',solver='adam',random_state=1)


      # ANN
      # Initialising the ANN
      #clf = Sequential()

      # Adding the input layer and the first hidden layer
      #clf.add(Dense(15, kernel_initializer='glorot_uniform', activation='relu', input_dim=len(original_X.columns)))
      # Adding the second hidden layer
      #clf.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
      #clf.add(Dense(5, kernel_initializer='glorot_uniform', activation='relu'))
      # Adding the output layer
      #clf.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
      # Compiling the ANN

      #clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


      # Naive_Bayes
      #clf = GaussianNB()


      # KNN
      # clf = KNeighborsClassifier(n_neighbors=100)


      # Decision Tree
      # clf = DecisionTreeClassifier()


      # Random Forest
      # clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)


      # SVM
      # clf = SVC(gamma='auto')


      # Perceptron
      # clf = Perceptron()
    else:
      clf = None

    total_list = topk_accuarcy(test_bug_reports, i, clf=clf)
    print("Fold ", i + 1)
    print("==========")
    print(total_list[0])
    print(total_list[1])
    print(total_list[2])

    return total_list


def evaluate_model_kfold(only_rvsm=False):
    """ Run kfold cross validation in parallel
    
    Keyword Arguments:
        k {integer} -- the number of folds (default: {10})
    """
    #samples = csv2dict(DATASET.results / "features.csv")

    # These collections are speed up the process while calculating top-k accuracy
    #sample_dict, bug_reports, br2files_dict = helper_collections(samples)

    #np.random.shuffle(samples)

    # K-fold Cross Validation in parallel
    #acc_dicts = Parallel(n_jobs=-2)(  # Uses all cores but one
    #    delayed(train_model)(
    #        i, k, samples, start, step, sample_dict, bug_reports, br2files_dict
    #    )
    #    for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    #)
    if DATASET.name == "aspectj":
      k = 3
    else:
      k = 10

    k_ids = list(range(0, k))
    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    total_lists = []
    for i in k_ids:
        total_list = train_model(i, k, bug_reports, only_rvsm)
        total_lists.append(total_list)
    print(total_lists)

    # Calculating the average accuracy from all folds
    mrrs = []
    maps = []
    acc_dicts = []
    for i in range(0, len(total_lists)):
        mrrs.append(total_lists[i][0])
        maps.append(total_lists[i][1])
        acc_dicts.append(total_lists[i][2])

    avg_mrr = np.mean(mrrs)
    avg_map = np.mean(maps)
    avg_acc_dict = {}
    for key in acc_dicts[0].keys():
        avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)

    print('Top K accuracy total: ', avg_acc_dict)
    print('MRR_total: ', avg_mrr)
    print('MAP_total: ', avg_map)

    return avg_acc_dict, avg_mrr, avg_map