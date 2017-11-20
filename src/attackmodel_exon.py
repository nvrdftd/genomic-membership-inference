import numpy as np
import tensorflow as tf
import csv
import random
import re
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Bidirectional, Conv1D, MaxPooling1D
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc

training_exon_file = 'highconfidencetargetexon012'
training_intron_file = 'highconfidencetargetexon1_randomized'

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]
    return (accuracy * 100)

# Plot data
def generate_results(y_true, y_predict):
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='#3261a1', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('eval/auc.png')
    plt.close()

def padding(mylist):
	listlength = len(mylist)
	iteration = 60 - listlength
	for a in range(iteration):
		mylist.append(np.zeros(4))
	return mylist

def randomize_except(c):
    chars = ['A', 'C', 'G', 'T']
    new_c = random.sample(chars, 1)
    while c == new_c:
        new_c = random.sample(chars, 1)
    return new_c

def preprocess_modify(filename):
    f = open(filename, 'r')
    training_x = list()
    mod_k = 20
    training_size = 0
    strlist = list()
    chars = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

    for l in f.readlines():
        searched = re.search('^>', l)
        if searched != None and searched.group(0) == '>':
            continue
        else:
            tmp = list()
            l = l.strip()
            start_index = random.randint(0, len(l) - mod_k)
            end_index = start_index + mod_k - 1

            while i < len(l):
                c = l[i]
                if i >= start_index and i <= end_index:
                    c = randomize_except(c)
                if c in chars:
                    tmp.append(chars[c])
            if len(tmp) == 60:
                training_size += 1
                tmp = padding(tmp)
                strlist.append(l)
                training_x.append(tmp)

    f.close()
    return training_x, training_size, strlist


def preprocess_fa(filename, sample_size=None):
    f = open(filename, 'r')
    training_x = list()
    training_size = 0
    strlist = list()
    chars = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

    for l in f.readlines():
        searched = re.search('^>', l)
        if searched != None and searched.group(0) == '>':
            continue
        else:
            tmp = list()
            for c in l.strip():
                if c in chars:
                    tmp.append(chars[c])
            if len(tmp) == 60:
                training_size += 1
                strlist.append(l)
                training_x.append(tmp)
                if training_size == sample_size:
                    break
    f.close()
    return training_x, training_size, strlist

def csvwriter(filename, data):
    with open(filename, 'w') as f:
        #fieldnames = ['y1', 'y2']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer.writeheader()
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)

def rawwriter(filename, data):
    with open(filename, 'w') as f:
        for e in data:
            f.write(e)


exon_x_train, training_exon_size, strtmp = preprocess_fa(training_exon_file)
intron_x_train, training_intron_size, strtmp = preprocess_fa(training_intron_file)
exon_y_train = [[1, 0] for _ in range(training_exon_size)]
intron_y_train = [[0, 1] for _ in range(training_intron_size)]

print('# of exons: %d' % training_exon_size)
print('# of introns: %d' % training_intron_size)

training_all_x = np.concatenate((exon_x_train, intron_x_train), axis=0).astype(dtype=np.float32)
training_all_y = np.concatenate((exon_y_train, intron_y_train), axis=0).astype(dtype=np.float32)


index = np.random.permutation(len(training_all_x))
train_data, train_labels = training_all_x[index], training_all_y[index]

metrics = list()
all_target_prediction = list()
all_test_prediction = list()

def contains(input_str):
    for s in target_exon_str:
        if s == input_str:
            return True
    return False

def attack_eval(target_prediction, test_prediction, threshold):
    target_size = len(target_prediction)
    test_size = len(test_prediction)
    precision = 0
    accuracy = 0
    recall = 0
    tp = 0
    tn = 0
    i = target_size - 1
    while i >= 0:
        target_val = target_prediction[i][0]
        test_val = test_prediction[i][0]
        if target_val >= threshold:
            tp += 1
        if test_val < threshold:
            tn += 1
        i -= 1
    total_count = float(tp + target_size - tp)
    if total_count <= 0:
        precision = 0.0
    else:
        precision = tp / total_count
    recall = tp / float(tp + test_size - tn)
    accuracy = (tp + tn) / float(target_size + test_size)
    return accuracy, precision, recall

def plot_figs(target_prediction, test_prediction):
    target_size = len(target_prediction)
    test_size = len(test_prediction)
    titles = [['5kmer Teset', '10kmer Test', '20kmer Test'], ['30kmer Test', '40kmer Test', 'Randomized Test']]
    f1, axes1 = plt.subplots(2, 3, sharex='all', sharey='all', tight_layout=True, figsize=(15, 10))
    f2, axes2 = plt.subplots(2, 3, sharex='all', sharey='all', tight_layout=True, figsize=(15, 10))
    pred_i = 0
    for i in range(2):
        for j in range(3):
            tar_pred = target_prediction[pred_i]
            tes_pred = test_prediction[pred_i]
            target_size = len(tar_pred)
            test_size = len(tes_pred)
            cur_fig = axes1[i, j]
            cur_fig.plot(np.linspace(1, target_size, num=target_size), tar_pred, marker='+', color='#909090', linestyle='None', label='Member')
            cur_fig.plot(np.linspace(1, test_size, num=test_size), tes_pred, marker='+', color='#3261a1', linestyle='None', label='Non-member')
            cur_fig.legend(prop={'size': 6})
            cur_fig.set_ylabel('Prediction')
            cur_fig.set_ylim(-0.05, 1.05)
            cur_fig.set_title(titles[i][j])

            cur_fig = axes2[i, j]
            cur_fig.hist([tar_pred, tes_pred], bins=20, color=['#909090', '#3261a1'], label=['Member', 'Non-member'])
            cur_fig.legend(prop={'size': 6})
            cur_fig.set_xlabel('Prediction')
            cur_fig.set_ylabel('Frequency')
            cur_fig.set_title(titles[i][j])
            pred_i += 1

    f1.savefig('figures/cmp_exon.png')
    f2.savefig('figures/hist_exon.png')

def test_eval(target_file, test_file):
    test_x, test_size, test_str = preprocess_fa(test_file)
    test_y = [[0, 1] for _ in range(test_size)]
    target_x, target_size, target_str = preprocess_fa(target_file, test_size)
    target_y = [[1, 0] for _ in range(target_size)]
    print('start tesing %s' % test_file)
    print('# of target: %d' % target_size)
    print('# of test: %d' % test_size)

    target_prediction = model.predict(np.asarray(target_x, dtype=np.float32))
    test_prediction = model.predict(np.asarray(test_x, dtype=np.float32))

    all_eval = list()
    threshold = 0.99
    interval = 0.01
    num = 9
    while num >= 0:
        print("testing selected threshold: %.2f" % threshold)
        acc, pre, rec = attack_eval(target_prediction, test_prediction, threshold)
        all_eval.append(acc)
        all_eval.append(pre)
        all_eval.append(rec)
        threshold -= interval
        num -= 1
    metrics.append(all_eval)
    all_target_prediction.append(target_prediction[:, 0])
    all_test_prediction.append(test_prediction[:, 0])

model = Sequential()
model.add(
    Conv1D(128, activation='relu', input_shape=(60, 4), strides=2, kernel_size=10))
model.add(MaxPooling1D(strides=3, pool_size=10))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128, input_shape=(128,), return_sequences=True)))
model.add(Bidirectional(LSTM(128, input_shape=(128,), return_sequences=True)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

print('Compiling model......')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
history = model.fit(train_data, train_labels, batch_size=128, epochs=20, validation_split=0.1, shuffle=False)

# summarize history for accuracy
plt.plot(history.history['acc'], marker='+', color='#909090', label='Training')
plt.plot(history.history['val_acc'], marker='+', color='#3261a1', label='Validation')
plt.title('Model Accuracy')
plt.ylim(0, 1.05)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower left', prop={'size': 6}, borderpad=1)
plt.savefig('figures/model_accuracy.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'], marker='+', color='#909090', label='Training')
plt.plot(history.history['val_loss'], marker='+', color='#3261a1', label='Validation')
plt.title('Model Loss')
plt.ylim(0, 1.0)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right', prop={'size': 6}, borderpad=1)
plt.savefig('figures/model_loss.png')
plt.close()

print("Attack evaluating...")

target_file = '../../datasets/exon/exon-target.fa'

test_file = 'highconfexon5mod'
test_eval(target_file, test_file)

test_file = 'highconfexon10mod'
test_eval(target_file, test_file)

test_file = 'highconfexon20mod'
test_eval(target_file, test_file)

test_file = 'highconfexon30mod'
test_eval(target_file, test_file)

test_file = 'highconfexon40mod'
test_eval(target_file, test_file)

test_file = 'highconfidencetargetexon1_randomized'
test_eval(target_file, test_file)

csvwriter('eval/metrics_all.csv', np.transpose(metrics))
plot_figs(all_target_prediction, all_test_prediction)
csvwriter('eval/predict_target.csv', all_target_prediction)
csvwriter('eval/predict_test.csv', all_test_prediction)

score_trainingall = model.predict(train_data)

generate_results(train_labels[:, 0], score_trainingall[:, 0])
