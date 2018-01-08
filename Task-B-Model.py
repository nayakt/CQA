import os
import sys
import re
import nltk
import math
from collections import defaultdict
from operator import itemgetter
import metrics
reload(sys)
sys.setdefaultencoding('utf-8')


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


def load_stop_words():
    s_words = list()
    file_reader = open('../data/stopwords.txt', 'r')
    lines = file_reader.readlines()
    for line in lines:
        s_words.append(line.replace('\n', '').lower())
    file_reader.close()
    return s_words


def get_stem(word):
    try:
        word_stem = stemmer.stem(word)
    except:
        word_stem = word
    return word_stem


# clean string if necessary
def clean(sent):
    sent = sent.lower()
    tokens = nltk.word_tokenize(sent)
    return tokens


def get_sent_pairs(lines, sent_1_ind, sent_2_ind):
    pairs = list()
    for line in lines:
        parts = line.strip().split('\t')
        sent_1 = ''
        for ind in sent_1_ind:
            sent_1 += parts[ind] + ' '
        sent_2 = ''
        for ind in sent_2_ind:
            sent_2 += parts[ind] + ' '
        pairs.append((sent_1, sent_2))
    return pairs


def is_valid_token(token):
    match_obj = re.match(r'^[a-z][a-z]*-?[a-z]*[a-z]$', token)
    if match_obj is not None:
        return True
    return False


def get_score(pair):
    s1 = pair[0].decode('utf8')
    s2 = pair[1].decode('utf8')
    tokens = clean(s1)
    s1_tokens = list()
    for token in tokens:
        if token not in stop_words and is_valid_token(token):
            s1_tokens.append(get_stem(token))
    tokens = clean(s2)
    s2_tokens = list()
    for token in tokens:
        if token not in stop_words and is_valid_token(token):
            s2_tokens.append(get_stem(token))
    tf = 0
    for token in s1_tokens:
        tf += s2_tokens.count(token)
    return tf  # / math.log(math.e + len(s2_tokens))


def cal_tf_sim_score(pairs):
    score = list()
    for pair in pairs:
        score.append(get_score(pair))
    norm_score = list()
    for i in range(0, len(score) - 9, 10):
        prob_sum = 0.0
        for j in range(i, i + 10):
            prob_sum += score[j]
        if prob_sum > 0:
            for j in range(i, i + 10):
                norm_score.append(score[j] * 1.0 / prob_sum)
        else:
            for j in range(i, i + 10):
                norm_score.append(score[j])
    return norm_score


if __name__ == "__main__":
    task = sys.argv[1]
    src_file = sys.argv[2]
    ref_file = sys.argv[3]
    pred_file = sys.argv[4]
    data_folder = '../data'
    stop_words = load_stop_words()
    reader = open(src_file)
    data_lines = reader.readlines()
    reader.close()
    task_A_Sent_1 = [1, 2]
    task_A_Sent_2 = [4]
    task_B_Sent_1 = [1, 2]
    task_B_Sent_2 = [4, 5]
    task_C_Sent_1 = [1, 2]
    task_C_Sent_2 = [4]
    sent_pairs = list()
    if task == 'A':
        sent_pairs = get_sent_pairs(data_lines, task_A_Sent_1, task_A_Sent_2)
    elif task == 'B':
        sent_pairs = get_sent_pairs(data_lines, task_B_Sent_1, task_B_Sent_2)
    stemmer = nltk.PorterStemmer()
    sim_score = cal_tf_sim_score(sent_pairs)
    writer = open(pred_file, 'w')
    for s in sim_score:
        writer.write(str(s) + '\n')
    writer.close()
    if ref_file != '':
        ref_lines = process_relevance_file(ref_file, 1)
        test_map, test_mrr = cal_score(ref_lines, sim_score)
        print test_map
        print test_mrr




















































