from models import kfold_split_indexes
from util import *
import numpy as np
from joblib import Parallel, delayed, cpu_count
import csv
import os
import pickle
from datas import DATASET
from sklearn.preprocessing import MinMaxScaler


def extract(i, br, bug_reports, java_src_dict, all_similarities_i, all_semantics_i):
    """ Extracts features for 50 wrong(randomly chosen) files for each
        right(buggy) file for the given bug report.

    Arguments:
        i {integer} -- Index for printing information
        br {dictionary} -- Given bug report
        bug_reports {list of dictionaries} -- All bug reports
        java_src_dict {dictionary} -- A dictionary of java source codes
    """
    print("Bug report : {} / {}".format(i + 1, len(bug_reports)), end="\r")

    br_id = br.bug_id
    br_date = br.report_time
    br_files = br.files
    br_text = ' '.join(br.summary['stemmed'] + br.description['stemmed'])
    br_raw_text = br.raw_summary + br.raw_description

    features = []

    for java_file in br_files:
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

            features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, ss, 1])

            for java_file, rvsm, cfs, cns, bfr, bff, ss in top_k_wrong_files(br, bug_reports, java_src_dict, all_similarities_i, all_semantics_i):
                features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, ss, 0])

        except:
            pass

    return features


def extract_features():
    """ Parallelizes the feature extraction process """

    # Read bug reports from tab separated file
    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    # Read all java source files
    with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
        java_src_dict = pickle.load(file)

    all_similarities = all_vsm_similarity(bug_reports, java_src_dict)
    all_semantics = all_semantic_similarity(bug_reports, java_src_dict)
    total_bugs = len(bug_reports)
    if DATASET.name == "aspectj":
      k = 3
    else:
      k = 10
    topk_wrong = 300

    for i, (start_bug, finish_bug) in enumerate(kfold_split_indexes(k, total_bugs)):
        # Use all CPUs except one to speed up extraction and avoid computer lagging
        batches = Parallel(n_jobs=cpu_count() - 1)(
            delayed(extract)(j, br, bug_reports, java_src_dict, all_similarities[j], all_semantics[j])
            for j, br in enumerate(list(bug_reports.values())[start_bug: finish_bug])
        )

        # Flatten features
        features = [row for batch in batches for row in batch]

        # Scale features
        features_rVSM = []
        features_collab_filter = []
        features_classname_similarity = []
        features_bug_recency = []
        features_bug_frequency = []
        features_semantic_similarity = []

        for m, value in enumerate(features):
            features_rVSM.append(value[2])
            features_collab_filter.append(value[3])
            features_classname_similarity.append(value[4])
            features_bug_recency.append(value[5])
            features_bug_frequency.append(value[6])
            features_semantic_similarity.append(value[7])

        features_rVSM = np.array(features_rVSM).reshape(-1, 1)
        features_collab_filter = np.array(features_collab_filter).reshape(-1, 1)
        features_classname_similarity = np.array(features_classname_similarity).reshape(-1, 1)
        features_bug_recency = np.array(features_bug_recency).reshape(-1, 1)
        features_bug_frequency = np.array(features_bug_frequency).reshape(-1, 1)
        features_semantic_similarity = np.array(features_semantic_similarity).reshape(-1, 1)
        
        min_max_scaler = MinMaxScaler()

        normalized_rVSM = np.concatenate(min_max_scaler.fit_transform(features_rVSM))
        normalized_collab_filter = np.concatenate(min_max_scaler.fit_transform(features_collab_filter))
        normalized_classname_similarity = np.concatenate(min_max_scaler.fit_transform(features_classname_similarity))
        normalized_bug_recency = np.concatenate(min_max_scaler.fit_transform(features_bug_recency))
        normalized_bug_frequency = np.concatenate(min_max_scaler.fit_transform(features_bug_frequency))
        normalized_semantic_similarity = np.concatenate(min_max_scaler.fit_transform(features_semantic_similarity))

        normalized_features = []
        for t, value in enumerate(features):
            normalized_features.append([value[0], value[1], normalized_rVSM[t], normalized_collab_filter[t],
                                        normalized_classname_similarity[t], normalized_bug_recency[t],
                                        normalized_bug_frequency[t], normalized_semantic_similarity[t], value[8]])

        '''features_train = []
        br_ids = set([s[0] for s in features])
        new_br_ids = list(br_ids)
        datas = {k: [] for k in new_br_ids}
        
        for sample in normalized_features:
            if int(sample[8]) == 1:
                features_train.append(sample)
            else:
                index = sample[0]
                datas[index].append(sample)
                
        for key, value in datas.items():
            top_k_files = sorted(value, key=lambda x: x[2])[:topk_wrong]
            for temp in top_k_files:
                features_train.append(temp)'''

        # Save features to a csv file
        features_path_train = os.path.normpath(
            str(DATASET.results / DATASET.name) + 'train' + str(i) + '.csv')
        with open(features_path_train, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "report_id",
                    "file",
                    "rVSM_similarity",
                    "collab_filter",
                    "classname_similarity",
                    "bug_recency",
                    "bug_frequency",
                    "semantic_similarity",
                    "match"
                ]
            )
            for row in normalized_features:
                writer.writerow(row)
                
        features_path_test = os.path.normpath(
            str(DATASET.results / DATASET.name) + 'test' + str(i) + '.csv')
        with open(features_path_test, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "report_id",
                    "file",
                    "rVSM_similarity",
                    "collab_filter",
                    "classname_similarity",
                    "bug_recency",
                    "bug_frequency",
                    "semantic_similarity",
                    "match"
                ]
            )
            for row in normalized_features:
                writer.writerow(row)

# Keep time while extracting features
with CodeTimer("Feature extraction"):
    extract_features()