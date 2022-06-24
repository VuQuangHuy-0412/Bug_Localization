import csv
import os
import random
import timeit
import numpy as np
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datas import DATASET


def tsv2dict(tsv_path):
    """ Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
        tsv_path {string} -- path of the tsv file
    """
    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        line["files"] = [
            os.path.normpath(f)
            for f in line["files"].strip().split()
            if f.endswith(".java")
        ]
        line["raw_text"] = line["summary"] + line["description"]
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )

        dict_list.append(line)
    return dict_list


def csv2dict(csv_path):
    """ Converts a comma separated values (csv) file into a dictionary

    Arguments:
        csv_path {string} -- path to csv file
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict


def top_k_wrong_files(br, bug_reports, java_src_dict, all_similarities_i, all_semantics_i, k=300):
    """ Randomly samples 2*k from all wrong files and returns metrics
        for top k files according to rvsm similarity.

    Arguments:
      br {BugReport} -- the current bug_report
      bug_reports {OrderDict of BugReport} -- all bug reports
      java_src_dict {OrderDict of SourceFile} -- all source files
      all_similarities_i {2D list} -- vsm_similarity all bug reports and all source files
      all_similarities_i {2D list} -- semantic_similarity of all bug reports and all source files

    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    """
    right_files = br.files
    br_text = ' '.join(br.summary['stemmed'] + br.description['stemmed'])
    br_date = br.report_time
    br_raw_text = br.raw_summary + br.raw_description
    # Randomly sample 2*k files
    randomly_sampled = random.sample(set(java_src_dict) - set(right_files), 2*k)

    all_files = []
    for java_file in randomly_sampled:
        java_file = os.path.normpath(java_file)

        try:
            # Source code of the java file
            src = java_src_dict[java_file]
            src_keys = list(java_src_dict.keys())
            index = src_keys.index(java_file)

            # rVSM Text Similarity
            rvsm = all_similarities_i[index]

            # Class Name Similarity
            src_name = src.exact_file_name
            cns = class_name_similarity(br_raw_text, src_name)

            # Previous Reports
            prev_reports = previous_reports(java_file, br_date, bug_reports)

            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_text, prev_reports)

            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)

            # Bug Fixing Frequency
            bff = len(prev_reports)

            # Semantic similarity
            ss = all_semantics_i[index]

            all_files.append([java_file, rvsm, cfs, cns, bfr, bff, ss])
        except:
            pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = ((tfidf * tfidf.T).A)[0, 1]

    return sim


def get_months_between(d1, d2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months


def most_recent_report(bug_reports):
    """ Returns the most recently submitted previous report 

    Arguments:
        bug_reports {OrderDict of BugReport} -- all bug reports
    """

    if len(bug_reports) > 0:
        return max(bug_reports, key=lambda x: x.report_time)

    return None


def previous_reports(filename, until, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        until {datetime} -- until date
        bug_reports {OrderDict of BugReport} -- all bug reports
    """
    return [
        bug_reports[br]
        for br in bug_reports
        if (filename in bug_reports[br].files and bug_reports[br].report_time < until)
    ]


def bug_fixing_recency(br, prev_reports):
    """ Calculates the Bug Fixing Recency

    Arguments:
        br {BugReport} -- current bug report
        prev_reports {list of BugReport} -- list of previous reports
    """
    mrr = most_recent_report(prev_reports)

    if br and mrr:
        return 1 / float(
            get_months_between(br.report_time, mrr.report_time) + 1
        )

    return 0


def collaborative_filtering_score(raw_text, prev_reports):
    """ Calculates the Collaborative Filtering Score

    Arguments:
        raw_text {string} -- raw text of the bug report 
        prev_reports {list of BugReport} -- list of previous reports
    """

    prev_reports_merged_text = ""
    for report in prev_reports:
        prev_reports_merged_text += ' '.join(report.summary['stemmed'] + report.description['stemmed'])

    cfs = cosine_sim(raw_text, prev_reports_merged_text)

    return cfs


def class_name_similarity(br_raw_text, src_name):
    """ Calculates the Class Name Similarity

    Arguments:
        br_raw_text {string} -- raw text of the bug report 
        src_name {string} -- name of the source file 
    """
    #classes = source_code.split(" class ")[1:]
    #class_names = [c[: c.find(" ")] for c in classes]
    #class_names_text = " ".join(class_names)
    #class_name_sim = cosine_sim(raw_text, class_names_text)
    if src_name in br_raw_text:
        class_name_sim = len(src_name)
    else:
        class_name_sim = 0
    return class_name_sim


def all_vsm_similarity(bug_reports, java_src_dict):
    """ Calculates the VSM Similarity between all bug reports and all source files

    Arguments:
        bug_reports {OrderDict of BugReport} -- all bug reports
        java_src_dict {OrderDict of SourceFile} -- all source files
    """

    src_texts = [' '.join(src.file_name['stemmed'] + src.class_names['stemmed']
                        + src.method_names['stemmed']
                        + src.variables['stemmed']
                        + src.pos_tagged_comments['stemmed']
                        + src.attributes['stemmed'])
                for src in java_src_dict.values()]

    br_texts = [' '.join(br.summary['stemmed'] + br.description['stemmed'])
                for br in bug_reports.values()]

    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    src_tfidf = tfidf.fit_transform(src_texts)
    reports_tfidf = tfidf.transform(br_texts)

    # normalizing the length of sources files
    src_lenghts = np.array([float(len(src_text.split())) for src_text in src_texts]).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    normalized_src_len = min_max_scaler.fit_transform(src_lenghts)

    # Applying logistic length function
    src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))

    all_simis = []
    for report_tfidf in reports_tfidf:
        s = cosine_similarity(src_tfidf, report_tfidf)
        # revised VSM score caculation
        rvsm_score = s * src_len_score
        normalized_score = np.concatenate(min_max_scaler.fit_transform(rvsm_score))
        all_simis.append(normalized_score.tolist())

    return all_simis


def all_semantic_similarity(bug_reports, java_src_dict):
    """ Calculates the Semantic Similarity between all bug reports and all source files

    Arguments:
        bug_reports {OrderDict of BugReport} -- all bug reports
        java_src_dict {OrderDict of SourceFile} -- all source files
    """

    nlp = spacy.load('en_core_web_lg')
    min_max_scaler = MinMaxScaler()

    src_docs = [nlp(' '.join(src.file_name['unstemmed'] + src.class_names['unstemmed']
                             + src.attributes['unstemmed']
                             + src.comments['unstemmed']
                             + src.method_names['unstemmed']))
                for src in java_src_dict.values()]

    all_simis = []
    for report in bug_reports.values():
        report_doc = nlp(' '.join(report.summary['unstemmed']
                                  + report.pos_tagged_description['unstemmed']))
        scores = []
        for src_doc in src_docs:
            simi = report_doc.similarity(src_doc)
            scores.append(simi)

        scores = np.array([float(count) for count in scores]).reshape(-1, 1)
        normalized_scores = np.concatenate(
            min_max_scaler.fit_transform(scores)
        )

        all_simis.append(normalized_scores.tolist())

    return all_simis


def helper_collections(samples, only_rvsm=False):
    """ Generates helper function for calculations

    Arguments:
        samples {list} -- samples from features.csv

    Keyword Arguments:
        only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False})
    """
    sample_dict = {}
    for s in samples:
        sample_dict[s["report_id"]] = []

    for s in samples:
        temp_dict = {}

        values = [float(s["rVSM_similarity"])]
        if not only_rvsm:
            values += [
                float(s["collab_filter"]),
                float(s["classname_similarity"]),
                float(s["bug_recency"]),
                float(s["bug_frequency"]),
                float(s["semantic_similarity"])
            ]
        temp_dict[os.path.normpath(s["file"])] = values

        sample_dict[s["report_id"]].append(temp_dict)

    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
    br2files_dict = {}

    for bug_report in bug_reports.values():
        br2files_dict[bug_report.bug_id] = bug_report.files

    return sample_dict, bug_reports, br2files_dict


def topk_accuarcy(test_bug_reports, i, clf=None):
    """ Calculates top-k accuracies

    Arguments:
        test_bug_reports {list of dictionaries} -- list of all bug reports
        i {int} -- the current fold

    Keyword Arguments:
        clf {object} -- A classifier with 'predict()' function. If None, rvsm relevancy is used. (default: {None})
    """
    test_path = str(DATASET.results / DATASET.name) + 'test' + str(i) + '.csv'
    test_sample = csv2dict(test_path)
    if clf is None:
        sample_dict, bug_reports, br2files_dict = helper_collections(test_sample, True)
    else:
        sample_dict, bug_reports, br2files_dict = helper_collections(test_sample)
    topk_counters = [0] * 20
    negative_total = 0
    mrr = []
    mean_avgp = []
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = bug_report.bug_id

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except:
            negative_total += 1
            continue

        # Calculate relevancy for all files related to the bug report in features.csv
        # Remember that, in features.csv, there are 300 wrong(randomly chosen) files for each right(buggy)
        relevancy_list = []
        if clf:  # dnn classifier
            relevancy_list = clf.predict(dnn_input)
            relevancy_list_new = []
            for y in relevancy_list:
                relevancy_list_new.append(y[0])
        else:  # rvsm
            relevancy_list = np.array(dnn_input).ravel()
            relevancy_list_new = relevancy_list
        x = list(np.argsort(relevancy_list, axis=0))
        x.reverse()

        temp = []
        for y in x:
            if clf is None:
                t = y
            else:
                t = y[0]
            temp.append(corresponding_files[t])
        # getting the ranks of reported fixed files
        relevant_ranks = sorted(temp.index(fixed) + 1
                                for fixed in br2files_dict[bug_id] if fixed in temp)
        if (len(relevant_ranks) == 0):
            continue
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                  for j, rank in enumerate(relevant_ranks)]))

        # Top-1, top-2 ... top-20 accuracy
        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list_new, -i)[-i:]
            # print(max_indices)
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    break
    acc_dict = {}
    print('negative_total: ', negative_total)
    mrr1 = np.mean(mrr)
    mean_avgp1 = np.mean(mean_avgp)

    for i, counter in enumerate(topk_counters):
        acc = counter / (len(test_bug_reports) - negative_total)
        acc_dict[i + 1] = round(acc, 3)
        
    total_list = []
    total_list.append(mrr1)
    total_list.append(mean_avgp1)
    total_list.append(acc_dict)
    return total_list


class CodeTimer:
    """ Keeps time from the initalization, and print the elapsed time at the end.

        Example:

        with CodeTimer("Message"):
            foo()
    """

    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs.".format(self.took))