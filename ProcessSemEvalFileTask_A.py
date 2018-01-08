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
    data_dir = "/home/nayak/Work/CQA/data/Task_A"
    src_files = list()
    # src_files.append("/home/nayak/Work/CQA/semeval2017_task3_test_input_abcd/SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-subtaskA-input.xml")
    src_files.append('/home/nayak/Work/CQA/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml')
    src_files.append("/home/nayak/Work/CQA/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml")
    src_files.append("/home/nayak/Work/CQA/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml")
    src_files.append("/home/nayak/Work/CQA/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml")
    src_files.append("/home/nayak/Work/CQA/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml")

    task_A_file_name = os.path.join('/home/nayak/Work/CQA/data/SemEval-2017/Task_A', 'train_big.txt')
    task_A_writer = open(task_A_file_name, 'w')

    for src_file_name in src_files:
        # src_file_name = os.path.join(data_dir, src_file_name)
        tree = ET.parse(src_file_name)
        root = tree.getroot()

        for thread_node in root:

            qs_node = thread_node.find("RelQuestion")
            qs_id = qs_node.get("RELQ_ID")

            qs_subj_node = qs_node.find("RelQSubject")
            qs_body_node = qs_node.find("RelQBody")
            qs_subj = "Nothing"
            if not qs_subj_node.text is None:
                qs_subj = qs_subj_node.text
                qs_subj = get_tokenized_sent(qs_subj)
            qs_body = "Nothing"
            if not qs_body_node.text is None:
                qs_body = qs_body_node.text
                qs_body = get_tokenized_sent(qs_body)
            qs = qs_subj + "\t" + qs_body

            com_count = 0
            for com_node in thread_node.findall("RelComment"):
                uid = com_node.get("RELC_USERID")
                val = com_node.get("RELC_RELEVANCE2RELQ")
                if val == "Good":
                    lbl = "1"
                else:
                    lbl = "0"
                comment = com_node.find("RelCText").text
                comment = get_tokenized_sent(comment)
                com_id = qs_id + "_C" + str(com_count + 1)

                task_A_writer.write(qs_id + "\t")
                task_A_writer.write(qs + "\t")

                task_A_writer.write(com_id + "\t")
                task_A_writer.write(comment + "\t")
                task_A_writer.write(uid + "\t")
                task_A_writer.write(str(com_count + 1) + "\t")
                task_A_writer.write(lbl + "\t")
                task_A_writer.write("\n")

                com_count += 1

    task_A_writer.close()