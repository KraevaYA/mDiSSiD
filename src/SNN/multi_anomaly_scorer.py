import numpy as np
import pandas as pd
import math
import os
import csv
#from sklearn.externals import joblib
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools

import utils
import config
import plots
import accuracy


def main():

    args = utils.parse_predict_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(e)
    
    dataset_dir = os.path.join(config.SNN_DATASETS_DIR, args.dataset)
    results_dir = os.path.join(config.RESULTS_DIR, args.dataset)

    dataset_params = utils.read_json_file(os.path.join(dataset_dir, '0', 'input_params.json'))


    #create combinations
    ts_indices = list(range(0, dataset_params['d']))
    combinations_indices = []

    for i in range(1, dataset_params['d']+1):
        combinations_indices.append(list(itertools.combinations(ts_indices, i)))

    # calculate thresholds
    test_predictions_for_threshold = []
    for i in range(dataset_params['d']):
        
        SNN_results_dir = os.path.join(results_dir, str(i), args.nn_type)
        
        pred = pd.read_csv(os.path.join(SNN_results_dir, 'test_predictions_for_threshold.csv'), header=None).to_numpy()
        
        #pred = np.subtract([1]*len(pred), pred)
        
        if (i == 0):
            test_predictions_for_threshold = pred
        else:
            test_predictions_for_threshold = np.concatenate((test_predictions_for_threshold, pred), axis=1)

                
    n_test = test_predictions_for_threshold.shape[0]
    all_max_N_comb = {}
    all_N_thresholds = {}
    
    print(f"Length of the original test time series = {n_test}")
    
    # calculate threshold for each N
    for i in range(dataset_params['d']):
        indexes = combinations_indices[i]
        
        # find min  indexes = [(0, 1), (0, 2), (1, 2)] or indexes = [(0, 1, 2)]
        max_N_comb = np.array([0]*n_test)
        for j in range(len(indexes)):
            min_comb = test_predictions_for_threshold[:,combinations_indices[i][j]].min(axis=1)
            max_N_comb = np.max((max_N_comb, min_comb), axis=0)
        all_max_N_comb[f"{i}"] = max_N_comb

        threshold = np.sort(max_N_comb)[int(np.ceil(config.N_PERCENTILE/100*n_test)-1)]
        threshold += threshold*0.5
        all_N_thresholds[f"{i}"] = threshold

        print(f"Threshold for {i}N = {threshold}")

    
    # create annotation for original test ts
    test_original_predictions = []
    for i in range(dataset_params['d']):
        SNN_results_dir = os.path.join(results_dir, str(i), args.nn_type)
        original_pred = pd.read_csv(os.path.join(SNN_results_dir, 'original_test_predictions.csv'), header=None).to_numpy()
        
        #original_pred = np.subtract([1]*len(original_pred), original_pred)
        
        if (i == 0):
            test_original_predictions = original_pred
        else:
            test_original_predictions = np.concatenate((test_original_predictions, original_pred), axis=1)


    n_original_test = test_original_predictions.shape[0]
    annotation_results = [0]*n_original_test

    all_max_N_comb_test = {}
    for i in range(dataset_params['d']):
        indexes = combinations_indices[i]
    
        max_N_comb_test = np.array([0]*n_original_test)
        for j in range(len(indexes)):
            min_comb_test = test_original_predictions[:,combinations_indices[i][j]].min(axis=1)
            max_N_comb_test = np.max((max_N_comb_test, min_comb_test), axis=0)
        all_max_N_comb_test[f"{i}"] = max_N_comb_test
            

    for i in range(n_original_test):
        label = 0
        for j in range(dataset_params['d']):
            if (all_max_N_comb_test[f"{j}"][i] >= all_N_thresholds[f"{j}"]):
                label = 1
                break
        annotation_results[i] = label

    SNN_results_dir = os.path.join(results_dir, args.nn_type)
    utils.create_directory(SNN_results_dir)
    utils.write_dataset(np.array(annotation_results).reshape(-1,1), SNN_results_dir, 'annotation_results.csv')

    # build comparison plot

    # read multivariate time series
    multi_test_ts = []
    for i in range(dataset_params['d']):
        test_original_ts_path = os.path.join(dataset_dir, str(i), 'test_original_ts.csv')
        test_ts =  pd.read_csv(test_original_ts_path, header=None).to_numpy()
        if (i == 0):
            multi_test_ts = test_ts
        else:
            multi_test_ts = np.concatenate((multi_test_ts, test_ts), axis=1)

    n_test = multi_test_ts.shape[0] # length of multivariate time series

    print(multi_test_ts)
    print(n_test)

    # read true labels for multivariate time series
    test_label_path = os.path.join(dataset_dir, '0', 'test_label.csv')
    true_label = utils.load_test_original_ts_from_csv(test_label_path)

    plots.plot_comparison_similarity_scores(multi_test_ts, annotation_results, true_label.reshape(1,-1)[0], n_test, len(annotation_results), dataset_params['d'], os.path.join(SNN_results_dir, 'similarity_scores.png'))


    # calculate final scores
    pred_final_scores = np.array([0]*n_original_test)
    for i in range(dataset_params['d']):
        pred_final_scores = np.min((pred_final_scores, all_max_N_comb_test[f"{i}"]), axis=0)

    # calculate accuracy matrics with VUS package
    if (config.ACCURACY_METHOD == 'VUS'):
        #np_predicted_scores = pd.DataFrame(final_scores).to_numpy().reshape(1,-1)[0]
        pred_final_scores = pd.DataFrame(annotation_results).to_numpy().reshape(1,-1)[0]
        np_true_label = pd.DataFrame(true_label).to_numpy().reshape(1,-1)[0]
        accuracy_metrics = accuracy.scoring(pred_final_scores, np_true_label, slidingWindow=dataset_params['m'])
        utils.save_metrics(accuracy_metrics, os.path.join(SNN_results_dir, 'accuracy_metrics.csv'))


if __name__ == '__main__':
    main()
