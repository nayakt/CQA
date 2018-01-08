import os
os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"

import numpy as np
np.random.seed(1000)
from scipy.spatial.distance import cdist
from collections import namedtuple
from keras.models import Model
from keras.layers import Dense, Activation, Input, merge, Lambda, Dropout
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
import sys
import re
import metrics
from collections import defaultdict
from sklearn import linear_model
from keras.models import load_model
from operator import itemgetter

vec_dim = 300
sent_vec_dim = 300
sent_len_cut_off = 80
word_vecs = {}
stop_words=[]
data_folder = ""


def load_word2vec(fname, data):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    file_writer = open ("../data/EmbeddingsNotFound.txt", 'w')
    vocab = load_vocab(data)
    print "vocab size:", len(vocab)
    NoWordVec = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            vec = np.fromstring(f.read(binary_len), dtype='float32')
            if vocab.has_key(word.lower()):
                word_vecs[word.lower()] = vec
            else:
                NoWordVec[word.lower()] = 1
    for word in NoWordVec.keys():
        file_writer.write(word + "\n")
    file_writer.close()
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
    print "Embedding size:", len(word_vecs.keys())
    return word_vecs


def load_vocab(data):
    vocab = {}
    for data_type in data:
        for data in data_type:
            for word in data.Sent_1_Words:
                if vocab.has_key(word):
                    vocab[word] += 1
                else:
                    vocab[word] = 1

            for word in data.Sent_2_Words:
                if vocab.has_key(word):
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    return vocab


def get_sample(task, line):
    line = line.strip()
    line = line.lower()
    parts = line.split('\t')
    if task == 'A':
        qs_id = parts[0]
        qs_subj = parts[1]
        qs = parts[2]
        com_id = parts[3]
        comment = parts[4]
        uid = parts[5][1:len(parts[5])]
        rank = parts[6]
        label = int(parts[7])
        if qs_subj != "Nothing":
            qs_subj_words = qs_subj.split()
        if qs != "Nothing":
            qs_words = qs.split()
        comment_words = comment.split()

        sent_1_words = list()
        for word in qs_subj_words:
            sent_1_words.append(word)
        for word in qs_words:
            sent_1_words.append(word)

        sent_2_words = list()
        for word in comment_words:
            sent_2_words.append(word)

        sample = QASample(Sent_1_Words=sent_1_words, Sent_2_Words=sent_2_words, Label=label)
        return sample
    else:
        org_qs_id = parts[0]
        org_qs_subj = parts[1]
        org_qs = parts[2]
        rel_qs_id = parts[3]
        rel_qs_subj = parts[4]
        rel_qs = parts[5]
        label = int(parts[6])
        if org_qs_subj != "Nothing":
            org_qs_subj_words = org_qs_subj.split()
        if org_qs != 'Nothing':
            org_qs_words = org_qs.split()
        if rel_qs_subj != "Nothing":
            rel_qs_subj_words = rel_qs_subj.split()
        if rel_qs != 'Nothing':
            rel_qs_words = rel_qs.split()

        sent_1_words = list()
        for word in org_qs_subj_words:
            sent_1_words.append(word)
        for word in org_qs_words:
            sent_1_words.append(word)

        sent_2_words = list()
        for word in rel_qs_subj_words:
            sent_2_words.append(word)
        for word in rel_qs_words:
            sent_2_words.append(word)

        sample = QASample(Sent_1_Words=sent_1_words, Sent_2_Words=sent_2_words, Label=label)
        return sample


def load_samples(task, file_name):
    file_reader = open(file_name)
    lines = file_reader.readlines()
    file_reader.close()
    samples = []
    for line in lines:
        sample = get_sample(task, line)
        samples.append(sample)
    return samples


def load_stop_words(stop_file):
    file_reader=open(stop_file)
    lines=file_reader.readlines()
    for line in lines:
        line=line.replace('\n','')
        stop_words.append(line)
    return stop_words


def get_max_len(train_samples, dev_samples, test_samples):
    max_l = len(train_samples[0].Sent_1_Words)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].Sent_1_Words) > max_l:
            max_l = len(train_samples[i].Sent_1_Words)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].Sent_1_Words) > max_l:
            max_l = len(dev_samples[i].Sent_1_Words)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].Sent_1_Words) > max_l:
            max_l = len(test_samples[i].Sent_1_Words)

    for i in range(0, len(train_samples)):
        if len(train_samples[i].Sent_2_Words) > max_l:
            max_l = len(train_samples[i].Sent_2_Words)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].Sent_2_Words) > max_l:
            max_l = len(dev_samples[i].Sent_2_Words)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].Sent_2_Words) > max_l:
            max_l = len(test_samples[i].Sent_2_Words)
    return max_l


def cal_score(ref_lines, probs):
    reranking_th = 0.0
    line_count = 0
    pred_lines = defaultdict(list)
    for ref_line in ref_lines:
        qid, aid, lbl = ref_line[0], ref_line[1], ref_line[2]
        pred_lines[qid].append((lbl, probs[line_count][0], aid))
        line_count += 1
    # for qid in pred_lines.keys():
    #     candidates = pred_lines[qid]
    #     if all(relevant == "false" for relevant, _, _ in candidates):
    #         del pred_lines[qid]
    for qid in pred_lines.keys():
        pred_sorted = pred_lines[qid]
        max_score = max([score for rel, score, aid in pred_sorted])
        if max_score >= reranking_th:
            pred_sorted = sorted(pred_sorted, key=itemgetter(1), reverse=True)
        pred_lines[qid] = [rel for rel, score, aid in pred_sorted]
    MAP = metrics.map(pred_lines, 10)
    MRR = metrics.mrr(pred_lines, 10)
    return MAP, MRR


def compose_decompose(qmatrix, amatrix):
    qhatmatrix, ahatmatrix = f_match(qmatrix, amatrix, window_size=3)
    qplus, qminus = f_decompose(qmatrix, qhatmatrix)
    aplus, aminus = f_decompose(amatrix, ahatmatrix)
    return qplus, qminus, aplus, aminus


def f_match(qmatrix, amatrix, window_size=3):
    A = 1 - cdist(qmatrix, amatrix, metric='cosine')  # Similarity matrix
    Atranspose = np.transpose(A)
    qa_max_indices = np.argmax(A,
                               axis=1)  # 1-d array: for each question word, the index of the answer word which is most similar
    # Selecting answer word vectors in a window surrounding the most closest answer word
    qa_window = [range(max(0, max_idx - window_size), min(amatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 qa_max_indices]
    # Selecting question word vectors in a window surrounding the most closest answer word
    # Finding weights and its sum (for normalization) to find f_match for question for the corresponding window of answer words
    qa_weights = [(np.sum(A[qword_idx][aword_indices]), A[qword_idx][aword_indices]) for qword_idx, aword_indices in
                  enumerate(qa_window)]
    # Then multiply each vector in the window with the weights, sum up the vectors and normalize it with the sum of weights
    # This will give the local-w vecotrs for the Question sentence words and Answer sentence words.
    qhatmatrix = np.array([np.sum(weights.reshape(-1, 1) * amatrix[aword_indices], axis=0) / weight_sum for
                           ((qword_idx, aword_indices), (weight_sum, weights)) in
                           zip(enumerate(qa_window), qa_weights)])

    # Doing similar stuff for answer words
    aq_max_indices = np.argmax(A,
                               axis=0)  # 1-d array: for each   answer word, the index of the question word which is most similar
    aq_window = [range(max(0, max_idx - window_size), min(qmatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 aq_max_indices]
    aq_weights = [(np.sum(Atranspose[aword_idx][qword_indices]), Atranspose[aword_idx][qword_indices]) for
                  aword_idx, qword_indices in enumerate(aq_window)]
    ahatmatrix = np.array([np.sum(weights.reshape(-1, 1) * qmatrix[qword_indices], axis=0) / weight_sum for
                           ((aword_idx, qword_indices), (weight_sum, weights)) in
                           zip(enumerate(aq_window), aq_weights)])
    return qhatmatrix, ahatmatrix


def f_decompose(matrix, hatmatrix):
    # finding magnitude of parallel vector
    mag = np.sum(hatmatrix * matrix, axis=1) / np.sum(hatmatrix * hatmatrix, axis=1)
    # multiplying magnitude with hatmatrix vector
    plus = mag.reshape(-1, 1) * hatmatrix
    minus = matrix - plus
    return plus, minus


def get_wang_conv_model_input(sent_1_matrix, sent_2_matrix, max_len):
    token = np.zeros((2, vec_dim), dtype='float')
    #for qmatrix, amatrix in zip(qsamples, asamples):
    sent_1_plus, sent_1_minus, sent_2_plus, sent_2_minus = compose_decompose(sent_1_matrix, sent_2_matrix)
    # Padding questions
    sent_1_pad_width = ((0, max_len - sent_1_plus.shape[0]), (0, 0))
    sent_1_plus_pad = np.pad(sent_1_plus, pad_width=sent_1_pad_width, mode='constant', constant_values=0.0)
    sent_1_minus_pad = np.pad(sent_1_minus, pad_width=sent_1_pad_width, mode='constant', constant_values=0.0)
    # Padding answers
    sent_2_pad_width = ((0, max_len - sent_2_plus.shape[0]), (0, 0))
    sent_2_plus_pad = np.pad(sent_2_plus, pad_width=sent_2_pad_width, mode='constant', constant_values=0.0)
    sent_2_minus_pad = np.pad(sent_2_minus, pad_width=sent_2_pad_width, mode='constant', constant_values=0.0)
    sent_1_plusminus = np.concatenate((sent_1_plus_pad, token, sent_1_minus_pad))
    sent_2_plusminus = np.concatenate((sent_2_plus_pad, token, sent_2_minus_pad))

    return sent_1_plusminus, sent_2_plusminus


def get_wang_model_input(samples, max_len):
    """
    Returns the training samples and labels as numpy array
    """

    sent_1_list = np.zeros((len(samples), (max_len * 2) + 2, sent_vec_dim))
    sent_2_list = np.zeros((len(samples), (max_len * 2) + 2, sent_vec_dim))
    labels_list = []
    counter = 0
    for sample in samples:
        sent_1_len = len(sample.Sent_1_Words)
        if sent_1_len > max_len:
            sent_1_len = max_len
        sent_2_len = len(sample.Sent_2_Words)
        if sent_2_len > max_len:
            sent_2_len = max_len

        sent_1_matrix = get_sent_matrix(sample.Sent_1_Words[0:sent_1_len])
        sent_2_matrix = get_sent_matrix(sample.Sent_2_Words[0:sent_2_len])
        sent_1_plusminus, sent_2_plusminus = get_wang_conv_model_input(sent_1_matrix, sent_2_matrix, max_len)
        sent_1_list[counter] = sent_1_plusminus
        sent_2_list[counter] = sent_2_plusminus
        labels_list.append(sample.Label)
        counter += 1

    return sent_1_list, sent_2_list, np.array(labels_list)


def get_sent_matrix(words):
    """
    Given a sentence, gets the input in the required format.
    """
    vecs = []
    vec = np.zeros(vec_dim, dtype='float32')
    for word in words:
        if word_vecs.has_key(word):
            vec = word_vecs[word]
        else:
            vec = word_vecs["<unk>"]
        vecs.append(np.array(vec))
    return np.array(vecs)


def run_wang_cnn_model(task, max_len, train_samples, dev_samples, dev_ref, test_samples):
    model_file = 'task_' + task + '_wang_sim_disim_dropout_model.h5'
    best_model_file = os.path.join(data_folder, model_file)
    batch_size = 10
    epoch = 10
    best_MAP = -10.0
    drop_p = 0.5
    print "pre-processing starts......"

    if max_len > sent_len_cut_off:
        max_len = sent_len_cut_off

    dev_q_tensor, dev_a_tensor, dev_labels_np = get_wang_model_input(dev_samples, max_len)
    print "dev input processing completed......"

    test_q_tensor, test_a_tensor, test_labels_np = get_wang_model_input(test_samples, max_len)
    print "test input processing completed......"

    train_q_tensor, train_a_tensor, train_labels_np = get_wang_model_input(train_samples, max_len)
    print "train input processing completed......"

    reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))

    nb_filter = 500
    l = 2 * max_len + 2
    qs_input = Input(shape=(l, vec_dim,), dtype='float32', name='qs_input')
    qs_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_3 = Dropout(drop_p)(qs_convmodel_3)
    qs_convmodel_3 = MaxPooling1D(pool_length=l - 2)(qs_convmodel_3)
    qs_convmodel_3 = reduce(qs_convmodel_3)
    qs_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_2 = Dropout(drop_p)(qs_convmodel_2)
    qs_convmodel_2 = MaxPooling1D(pool_length=l - 1)(qs_convmodel_2)
    qs_convmodel_2 = reduce(qs_convmodel_2)
    qs_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_1 = Dropout(drop_p)(qs_convmodel_1)
    qs_convmodel_1 = MaxPooling1D(pool_length=l)(qs_convmodel_1)
    qs_convmodel_1 = reduce(qs_convmodel_1)
    qs_concat = merge([qs_convmodel_1, qs_convmodel_2, qs_convmodel_3], mode='concat', concat_axis=-1)

    ans_input = Input(shape=(l, vec_dim,), dtype='float32', name='ans_input')
    ans_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_3 = Dropout(drop_p)(ans_convmodel_3)
    ans_convmodel_3 = MaxPooling1D(pool_length=l - 2)(ans_convmodel_3)
    ans_convmodel_3 = reduce(ans_convmodel_3)
    ans_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_2 = Dropout(drop_p)(ans_convmodel_2)
    ans_convmodel_2 = MaxPooling1D(pool_length=l - 1)(ans_convmodel_2)
    ans_convmodel_2 = reduce(ans_convmodel_2)
    ans_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_1 = Dropout(drop_p)(ans_convmodel_1)
    ans_convmodel_1 = MaxPooling1D(pool_length=l)(ans_convmodel_1)
    ans_convmodel_1 = reduce(ans_convmodel_1)
    ans_concat = merge([ans_convmodel_1, ans_convmodel_2, ans_convmodel_3], mode='concat', concat_axis=-1)

    q_a_model=merge([qs_concat, ans_concat], mode='concat', concat_axis=-1)
    sim_model = Dense(output_dim=1, activation='linear')(q_a_model)
    labels = Activation('sigmoid', name='labels')(sim_model)

    wang_model = Model(input=[qs_input, ans_input], output=[labels])

    #model.summary()

    wang_model.compile(loss={'labels': 'binary_crossentropy'},
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    for epoch_count in range(0, epoch):
        wang_model.fit({'qs_input': train_q_tensor, 'ans_input': train_a_tensor}, {'labels': train_labels_np}, nb_epoch=1, batch_size=batch_size, verbose=2)
        dev_probs = wang_model.predict([dev_q_tensor, dev_a_tensor], batch_size=batch_size)
        MAP, MRR = cal_score(dev_ref, dev_probs)
        print "Dev MAP:", MAP
        if MAP > best_MAP:
            best_MAP = MAP
            wang_model.save(best_model_file)

    best_wang_model = load_model(best_model_file)
    #train_probs = best_wang_model.predict([train_q_tensor, train_a_tensor], batch_size=batch_size)
    #dev_probs = best_wang_model.predict([dev_q_tensor, dev_a_tensor], batch_size=batch_size)
    test_probs = best_wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    #
    # reg_train_data_np = get_lr_data(train_samples, train_probs)
    # reg_dev_data_np = get_lr_data(dev_samples, dev_probs)
    # reg_test_data_np = get_lr_data(test_samples, test_probs)
    #
    # LR_Dense_MAP = train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np, train_labels_np, dev_ref, test_ref)
    return test_probs


def get_ref_res(task, file_name):
    ref_res = []
    if task == 'A':
        with open(file_name) as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')
                lbl = 'true'
                if int(parts[7]) == 0:
                    lbl = 'false'
                ref_res.append((parts[0], parts[3], lbl))
    else:
        with open(file_name) as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')
                lbl = 'true'
                if int(parts[6]) == 0:
                    lbl = 'false'
                ref_res.append((parts[0], parts[3], lbl))

    return ref_res


def process_relevance_file(file_name):
    file_reader = open(file_name)
    lines = file_reader.readlines()
    file_reader.close()
    ref_res = []
    for line in lines:
        line = line.strip()
        parts = line.split("\t")
        label = parts[4]
        ref_res.append((parts[0], parts[1], label))
    return ref_res


if __name__ == "__main__":

    task_name = sys.argv[1]
    data_folder = os.path.join("../data")

    word_vec_file = os.path.join(data_folder, sys.argv[2])
    stop_words_file = os.path.join(data_folder, sys.argv[3])

    data_folder = os.path.join(data_folder, "Task_" + task_name)

    train_file = os.path.join(data_folder, sys.argv[4])
    dev_file = os.path.join(data_folder, sys.argv[5])
    test_file = os.path.join(data_folder, sys.argv[6])
    test_ref_file = os.path.join(data_folder, sys.argv[7])
    pred_file = os.path.join(data_folder, "preds.txt")

    QASample=namedtuple("QASample", "Sent_1_Words Sent_2_Words Label")

    print "loading data......"

    train_data = load_samples(task_name, train_file)
    test_data = load_samples(task_name, test_file)
    dev_data = load_samples(task_name, dev_file)
    max_l = get_max_len(train_data, dev_data, test_data)

    print "Max. len:", max_l

    print "loading word vectors......"
    word_vecs = load_word2vec(word_vec_file, [train_data, dev_data, test_data])
    print "word vectors loaded......"

    stop_words = load_stop_words(stop_file=stop_words_file)

    dev_ref = get_ref_res(task_name, dev_file)
    test_ref = process_relevance_file(test_ref_file)

    print "Decomposition and Composition based CNN model started......"
    # test_probs = list()
    # for i in range(0, len(test_data)):
    #     test_probs.append([0.0])
    #
    # for j in range(0, 10):
    #     print 'Model Count:', (j+1)
    #     probs = run_wang_cnn_model(task_name, max_l, train_data, dev_data, dev_ref, test_data)
    #     MAP, MRR = cal_score(test_ref, probs)
    #     print "MAP:", MAP
    #     print "MRR:", MRR
    #     for i in range(0, len(test_data)):
    #         test_probs[i][0] += probs[i][0]
    test_probs = run_wang_cnn_model(task_name, max_l, train_data, dev_data, dev_ref, test_data)
    MAP, MRR = cal_score(test_ref, test_probs)
    file_writer = open(pred_file, 'w')
    for index in range(0, len(test_ref)):
        file_writer.write(test_ref[index][0] + "\t" + test_ref[index][1] + "\t" + "0" + "\t" + str(
            test_probs[index][0]) + "\t" + "false")
        file_writer.write("\n")
    file_writer.close()
    print "Decomp Comp CNN"
    print "MAP:", MAP
    print "MRR:", MRR




















































