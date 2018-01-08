import os
import sys
import xml.etree.ElementTree as ET
import nltk


def get_tokenized_sent(sent):
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    tokenized_sent = " ".join(tokens)
    return tokenized_sent.strip()

if __name__=="__main__":
    #file_name = sys.argv[1]
    reload(sys)
    sys.setdefaultencoding('UTF8')
    data_dir = "/home/nayak/Work/SemEval2017/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train"
    src_files = list()
    src_files.append("/home/nayak/Work/CQA/semeval2017_task3_test_input_abcd/SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-input.xml")
    # src_files.append("SemEval2016-Task3-CQA-QL-train-part2.xml")

    task_B_file_name = os.path.join('/home/nayak/Work/CQA/data/SemEval-2017/Task_C', 'Task_B_test.txt')
    task_B_writer = open(task_B_file_name, 'w')
    task_A_file_name = os.path.join('/home/nayak/Work/CQA/data/SemEval-2017/Task_C', 'Task_A_test.txt')
    task_A_writer = open(task_A_file_name, 'w')
    # ref_file_name = os.path.join(data_dir, 'train.relevancy')
    # ref_file_writer = open(ref_file_name, 'w')

    qs_map = dict()
    for src_file_name in src_files:
        tree = ET.parse(src_file_name)
        root = tree.getroot()

        for org_q_node in root:
            org_q_id = org_q_node.get('ORGQ_ID')
            org_q_subj_node = org_q_node.find('OrgQSubject')
            org_q_body_node = org_q_node.find('OrgQBody')
            org_q_subj = 'Nothing'
            if org_q_subj_node.text is not None:
                org_q_subj = get_tokenized_sent(org_q_subj_node.text)
            org_q_body = 'Nothing'
            if org_q_body_node.text is not None:
                org_q_body = get_tokenized_sent(org_q_body_node.text)

            thread_node = org_q_node.find("Thread")

            rel_q_node = thread_node.find("RelQuestion")
            rel_q_id = rel_q_node.get('RELQ_ID')
            rel_q_subj_node = rel_q_node.find('RelQSubject')
            rel_q_body_node = rel_q_node.find('RelQBody')
            rel_q_subj = 'Nothing'
            if rel_q_subj_node.text is not None:
                rel_q_subj = get_tokenized_sent(rel_q_subj_node.text)
            rel_q_body = 'Nothing'
            if rel_q_body_node.text is not None:
                rel_q_body = get_tokenized_sent(rel_q_body_node.text)

            val = '0'
            if rel_q_node.get("RELQ_RELEVANCE2ORGQ") == "PerfectMatch" or rel_q_node.get("RELQ_RELEVANCE2ORGQ") == "Relevant":
                val = '1'

            task_B_writer.write(org_q_id + "\t" + org_q_subj + "\t" + org_q_body + "\t" + rel_q_id + "\t" + rel_q_subj + "\t" + rel_q_body + "\t" + val + "\n")

            com_count = 0
            for com_node in thread_node.findall("RelComment"):
                uid = com_node.get("RELC_USERID")
                val = com_node.get("RELC_RELEVANCE2RELQ")
                val1 = com_node.get("RELC_RELEVANCE2ORGQ")
                if val == "Good":
                    lbl = "1"
                else:
                    lbl = "0"
                if val1 == "Good":
                    lbl1 = "1"
                else:
                    lbl1 = "0"
                comment = com_node.find("RelCText").text
                comment = get_tokenized_sent(comment)
                com_id = rel_q_id + "_C" + str(com_count + 1)

                task_A_writer.write(rel_q_id + "\t")
                task_A_writer.write(rel_q_subj + "\t")
                task_A_writer.write(rel_q_body + "\t")
                #
                task_A_writer.write(com_id + "\t")
                task_A_writer.write(comment + "\t")
                task_A_writer.write(uid + "\t")
                task_A_writer.write(str(com_count + 1) + "\t")
                task_A_writer.write(lbl + "\t")
                task_A_writer.write("\n")
                # ref_file_writer.write(org_q_id + '\t' + com_id + '\t0\t0\t' + lbl1 + '\n')
                com_count += 1

    task_B_writer.close()
    task_A_writer.close()
    #ref_file_writer.close()