from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D,Convolution1D,MaxPooling2D,MaxPooling1D
from keras.optimizers import Adadelta,RMSprop
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from random import randint
from sklearn.cross_validation import train_test_split
from keras.layers import LSTM, Dense 
from keras.layers.wrappers import Bidirectional
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
import tensorflow as tf
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.feature_extraction import DictVectorizer as DV
tf.logging.set_verbosity(tf.logging.INFO)
from keras import utils as np_utils
from keras.layers import Merge

mpl.use('Agg')

training_exon_file = 'train-shadow-exon'
training_intron_file = 'train-shadow-intron'

test_10kmerexon_file = 'shadow-10kmer-exon'
test_10kmerintron_file = 'shadow-10kmer-intron'

test_2kmerexon_file= 'shadow-2kmer-exon'
test_2kmerintron_file  = 'shadow-2kmer-intron'

test_5kmerexon_file= 'shadow-5kmer-exon'
test_5kmerintron_file  = 'shadow-5kmer-intron'

test_8kmerexon_file = 'shadow-8kmer-exon'
test_8kmerintron_file = 'shadow-8kmer-intron'

test_20kmerexon_file = 'shadow-20kmer-exon'
test_20kmerintron_file = 'shadow-20kmer-intron'

test_30kmerexon_file = 'shadow-30kmer-exon'
test_30kmerintron_file = 'shadow-30kmer-intron'

test_40kmerexon_file = 'shadow-40kmer-exon'
test_40kmerintron_file = 'shadow-40kmer-intron'

test_50kmerexon_file = 'shadow-50kmer-exon'
test_50kmerintron_file = 'shadow-50kmer-intron'

test_exon_shuffle = 'randomized-shadow-exon'
test_intron_shuffle = 'randomized-shadow-intron'

#calculate accuracy
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


# Plot ROC curve
def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(15,5))
    plt.plot(fpr, tpr, label='ROC curve of Training Data(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.savefig('auc.png')
    plt.close()
    print('AUC: %f' % roc_auc)

def padding(mylist):
	listlength = len(mylist)
	iteration = 60 - listlength
	for a in range(iteration):
		mylist.append(np.zeros(4))
	return mylist

#preprocess the DNA sequence to one-hot encoding
def preprocess_fa(filename):
    f = open(filename, 'r')
    training_x = list()
    training_size = 0
    strlist = list()

    for l in f.readlines():
        searched = re.search('^>', l)
        if searched != None and searched.group(0) == '>':
            continue
        else:
            tmp = list()
            for c in l.strip():
                if c in "A":
                    tmp.append(np.array([1, 0, 0, 0]))
                if c in "T":
                    tmp.append(np.array([0, 0, 0, 1]))
                if c in "G":
                    tmp.append(np.array([0, 0, 1, 0]))
                if c in "C":
                    tmp.append(np.array([0, 1, 0, 0]))
            if len(tmp) == 60:
                training_size += 1
                tmp = padding(tmp)
                strlist.append(l)
                training_x.append(tmp)

    f.close()
    return training_x, training_size, strlist

#retrieve the 60bp DNA sequence with encoded one hot vector
exon_x_train, training_exon_size, strtmp = preprocess_fa(training_exon_file)
intron_x_train, training_intron_size, strtmp = preprocess_fa(training_intron_file)

test_2kmerintron, intron2kmersize, strtmp = preprocess_fa(test_2kmerintron_file)
test_2kmerexon, exon2kmersize, strtmp = preprocess_fa(test_2kmerexon_file)

test_5kmerintron, intron5kmersize, strtmp = preprocess_fa(test_5kmerintron_file)
test_5kmerexon, exon5kmersize, strtmp = preprocess_fa(test_5kmerexon_file)

test_8kmerintron, intron8kmersize, strtmp = preprocess_fa(test_8kmerintron_file)
test_8kmerexon, exon8kmersize, strtmp = preprocess_fa(test_8kmerexon_file)

test_10kmerintron, intron10kmersize, strtmp = preprocess_fa(test_10kmerintron_file)
test_10kmerexon, exon10kmersize, strtmp = preprocess_fa(test_10kmerexon_file)

test_20kmerintron, intron20kmersize, strtmp = preprocess_fa(test_20kmerintron_file)
test_20kmerexon, exon20kmersize, strtmp = preprocess_fa(test_20kmerexon_file)

test_30kmerintron, intron30kmersize, strtmp = preprocess_fa(test_30kmerintron_file)
test_30kmerexon, exon30kmersize, strtmp = preprocess_fa(test_30kmerexon_file)

test_40kmerintron, intron40kmersize, strtmp = preprocess_fa(test_40kmerintron_file)
test_40kmerexon, exon40kmersize, strtmp = preprocess_fa(test_40kmerexon_file)

test_50kmerintron, intron50kmersize, strtmp = preprocess_fa(test_50kmerintron_file)
test_50kmerexon, exon50kmersize, strtmp = preprocess_fa(test_50kmerexon_file)

exon_shuffle_test, exon_shuff_size, strtmp = preprocess_fa(test_exon_shuffle)
intron_shuffle_test, intron_shuff_size, strtmp = preprocess_fa(test_intron_shuffle)

#give the train label class for exon and intron separately
exon_y_train = [[1, 0] for _ in range(training_exon_size)]
intron_y_train = [[0, 1]for _ in range(training_intron_size)]

#concatenate the training datasets
training_all_x = np.concatenate((exon_x_train,intron_x_train),axis=0)
training_all_y = np.concatenate((exon_y_train,intron_y_train),axis=0)

#concatenate the test datasets
test_all_x = np.concatenate((exon_x_train,intron_x_train,test_2kmerexon,test_2kmerintron), axis=0)

#convert all training and test data into np array value float 32
train_data = np.asarray(training_all_x,dtype=np.float32) # Returns np.array
train_labels = np.asarray(training_all_y, dtype=np.float32)
eval_data = np.asarray(test_all_x, dtype=np.float32) # Returns np.array

model = Sequential()
#first layer CNN 1D with 320 nb_filters, shape(60,4), filter length=26, activation relu with subsample output=1
model.add(Convolution1D(320, input_shape=(60,4),border_mode='valid', filter_length=26, activation="relu",subsample_length=1))
#max pooling with stride=4
model.add(MaxPooling1D(pool_length=26, stride=4))
#drop out 20%
model.add(Dropout(0.2))
#Bi-directional LSTM both backward and forward
model.add(Bidirectional(LSTM(input_dim=320, output_dim=320, return_sequences=True,go_backwards= False)))
model.add(Bidirectional(LSTM(input_dim=320, output_dim=320, return_sequences=True,go_backwards= True)))
#drop out 50%
model.add(Dropout(0.5))
model.add(Flatten())
#activation relu
model.add(Dense(320, activation='relu'))
#2 output with activation sigmoid
model.add(Dense(2, activation='sigmoid'))

print ('compiling model')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history=model.fit(train_data, train_labels, batch_size=100, nb_epoch=20,verbose=1, validation_split=0.3)
#print ("Accuracy on train data is:")
#print(accuracy(train_data, train_labels, model))

print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(15,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('cnnaccuracy.png')  
plt.close()

# summarize history for loss
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('cnnloss.png')  
plt.close()

#predict original training dataset
score_trainingall = model.predict(train_data)
#predict all the test data from 2kmer, 5kmer, 8kmer, 10kmer, 20kmer, 30kmer,40kmer and randomized sequence
score_alltest = model.predict(eval_data)
#write the results into a file
filename = 'results_2kmer.txt'
with open(filename, 'w') as f:
 for a in range(len(score_alltest)):
    f.write("%s\n" %  str(score_alltest[a]))

#generate AUC  plot for training model
print('Generating AUC results training')
generate_results(train_labels[:, 0], score_trainingall[:, 0])

