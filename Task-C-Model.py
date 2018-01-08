import os
os.environ["THEANO_FLAGS"] = "device=cuda1,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"

import numpy as np
np.random.seed(1000)
from scipy.spatial.distance import cdist
from collections import namedtuple
from keras.models import Model
from keras.layers import Dense, Activation, Input, merge, Lambda
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


def get_max_len(test_samples):
    max_l = len(test_samples[0].Sent_1_Words)

    for i in range(1, len(test_samples)):
        if len(test_samples[i].Sent_1_Words) > max_l:
            max_l = len(test_samples[i].Sent_1_Words)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].Sent_2_Words) > max_l:
            max_l = len(test_samples[i].Sent_2_Words)
    return max_l


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


def cal_score(ref_lines, probs):
    reranking_th = 0.0
    line_count = 0
    pred_lines = defaultdict(list)
    for ref_line in ref_lines:
        qid, aid, lbl = ref_line[0], ref_line[1], ref_line[2]
        pred_lines[qid].append((lbl, probs[line_count], aid))
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


def get_wang_conv_sim_model_input(sent_1_matrix, sent_2_matrix, max_len):
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

    #return sent_1_plusminus, sent_2_plusminus
    return  sent_1_plus_pad, sent_2_plus_pad


def get_wang_model_input(task, samples, max_len):
    """
    Returns the training samples and labels as numpy array
    """
    if task == 'A':
        sent_1_list = np.zeros((len(samples), (max_len * 2) + 2, sent_vec_dim))
        sent_2_list = np.zeros((len(samples), (max_len * 2) + 2, sent_vec_dim))
    else:
        sent_1_list = np.zeros((len(samples), max_len, sent_vec_dim))
        sent_2_list = np.zeros((len(samples), max_len, sent_vec_dim))
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
        if task == 'A':
            sent_1_plusminus, sent_2_plusminus = get_wang_conv_model_input(sent_1_matrix, sent_2_matrix, max_len)
        else:
            sent_1_plusminus, sent_2_plusminus = get_wang_conv_sim_model_input(sent_1_matrix, sent_2_matrix, max_len)
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


def run_pre_trained_model(task, model_file, test_data, max_len):
    batch_size = 32
    wang_model = load_model(model_file)
    if max_len > sent_len_cut_off:
        max_len = sent_len_cut_off
    test_q_tensor, test_a_tensor, test_labels_np = get_wang_model_input(task, test_data, max_len)
    test_probs = wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    return test_probs


def process_relevance_file(file_name, label_type):
    file_reader = open(file_name)
    lines = file_reader.readlines()
    file_reader.close()
    ref_res = []
    for line in lines:
        line = line.strip()
        parts = line.split("\t")
        label = parts[4]
        if label_type == 1:
            if label == '1':
                label = 'true'
            elif label == '0':
                label = 'false'
        ref_res.append((parts[0], parts[1], label))
    return ref_res

if __name__ == "__main__":
    word_vec_file = sys.argv[1]
    stop_words_file = sys.argv[2]
    data_folder = "../data/SemEval-2017/Task_C"
    task_A_model_file = sys.argv[3]
    task_A_data_file = sys.argv[4]
    task_B_model_file = sys.argv[5]
    task_B_data_file = sys.argv[6]

    QASample = namedtuple("QASample", "Sent_1_Words Sent_2_Words Label")

    print "loading data......"

    task_A_data = load_samples('A', task_A_data_file)
    task_B_data = load_samples('B', task_B_data_file)

    print "loading word vectors......"
    word_vecs = load_word2vec(word_vec_file, [task_A_data, task_B_data])
    print "word vectors loaded......"

    stop_words = load_stop_words(stop_file=stop_words_file)

    max_l = get_max_len(task_A_data)
    task_A_probs = run_pre_trained_model('A', task_A_model_file, task_A_data, max_l)
    prob_file = os.path.join(data_folder, 'probs_A.txt')
    writer = open(prob_file, 'w')
    for i in range(0, len(task_A_probs)):
        writer.write(str(task_A_probs[i][0]) + '\n')
    writer.close()

    prob_file = os.path.join(data_folder, 'probs_A_normalized.txt')
    writer = open(prob_file, 'w')
    for i in range(0, len(task_A_probs) / 10):
        prob_sum = 0.0
        for j in range(0, 10):
            prob_sum += task_A_probs[i * 10 + j][0]
        for j in range(0, 10):
            writer.write(str(task_A_probs[i * 10 + j][0] / prob_sum) + '\n')
    writer.close()

    max_l = get_max_len(task_B_data)
    task_B_probs = run_pre_trained_model('A', task_B_model_file, task_B_data, max_l)
    prob_file = os.path.join(data_folder, 'probs_B.txt')
    writer = open(prob_file, 'w')
    for i in range(0, len(task_B_probs)):
        writer.write(str(task_B_probs[i][0]) + '\n')
    writer.close()

    prob_file = os.path.join(data_folder, 'probs_B_normalized.txt')
    writer = open(prob_file, 'w')
    for i in range(0, len(task_B_probs) / 10):
        prob_sum = 0.0
        for j in range(0, 10):
            prob_sum += task_B_probs[i * 10 + j][0]
        for j in range(0, 10):
            writer.write(str(task_B_probs[i * 10 + j][0] / prob_sum) + '\n')
    writer.close()





















































