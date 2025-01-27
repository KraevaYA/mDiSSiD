import numpy as np
import pandas as pd
import math
import os
import csv
import tensorflow as tf
#from sklearn.externals import joblib
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import multi_siamese_nn
import utils
import config
import plots
import accuracy

#import tensorflow as tf

def main():
    
    args = utils.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(e)
    
    dataset_dir = os.path.join(config.SNN_DATASETS_DIR, args.dataset, str(args.dimension))
    results_dir = os.path.join(config.RESULTS_DIR, args.dataset, str(args.dimension))
    utils.create_directory(results_dir)

    dataset_params = utils.read_json_file(os.path.join(dataset_dir, 'input_params.json'))

    num_subs_per_pair = 2
    
    if (args.act == 'fit'):

        # 1. read and load the training set
        print('1. Start to read and transform the training set for fitting the SNN')
        train_val_set_path = os.path.join(dataset_dir, 'train_set.csv')
        x_train_val, y_train_val = utils.load_dataset(train_val_set_path)
        x_train_val = utils.normalize_dataset(x_train_val)
        x_train, y_train, x_val, y_val = utils.train_val_split(x_train_val, y_train_val, args.val_size)
        
        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)
        
        # make the train samples
        X_train, Y_train = utils.make_train_set_samples_with_snippets(x_train, y_train, x_snippets, y_snippets)
        # make the validation samples
        X_val, Y_val = utils.make_train_set_samples(x_val, y_val)
    
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3], 1))

        ready_X_train = []
        ready_X_val = []

        for i in range(dataset_params['snippets_number']):
            for j in range(num_subs_per_pair):
                ready_X_train.append(X_train[:,i,j])
                ready_X_val.append(X_val[:,i,j])

        print('The training and validation sets are ready to fit the SNN\n')
        
        # -----------------------------------------------

        # 2. init SNN, fit and predict the similarity score
        print('2. Start to fit SNN')
        SNN_results_dir = os.path.join(results_dir, args.nn_type)
        utils.create_directory(SNN_results_dir)

        mDiSSiD_model = multi_siamese_nn.build_mDiSSiD_model(snippets_num=dataset_params['snippets_number'], input_shape=(dataset_params['m'], 1), base_type=args.nn_type)

        mDiSSiD_model.compile(loss=multi_siamese_nn.multi_loss(margin=args.margin), optimizer=args.optimizer)
        mDiSSiD_model.summary()

        history = mDiSSiD_model.fit(ready_X_train, Y_train,
                                    validation_data=(ready_X_val, Y_val),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    shuffle=True,
                                    #callbacks=callbacks,
                                    )
        model_dir = os.path.join(SNN_results_dir, 'models')
        utils.create_directory(model_dir)
        mDiSSiD_model.save_weights(os.path.join(model_dir, 'weights.hdf5'))

        print('mDiSSiD model is fitted\n')

        # --------------------------------------------------------------

        # 3. read and load the test and snippets sets
        print("3. Start to read and transform the test set for finding the anomaly threshold")
        test_set_path = os.path.join(dataset_dir, 'test_set.csv')
        x_test, y_test = utils.load_dataset(test_set_path)
        x_test = utils.normalize_dataset(x_test)

        X_test, Y_test = utils.make_test_set_samples(x_test, y_test, x_snippets, y_snippets)

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1))

        ready_X_test = []
        for i in range(dataset_params['snippets_number']):
            for j in range(num_subs_per_pair):
                ready_X_test.append(X_test[:,i,j])

        predictions = mDiSSiD_model.predict(ready_X_test)
        min_predictions = predictions.min(axis=1)

        utils.write_dataset(min_predictions.reshape(-1,1), SNN_results_dir, 'test_predictions_for_threshold.csv')

        print('The Siamese neural network finished to detect anomalies in the test set\n')

        utils.save_snn_params(args.nn_type, args.epochs, args.batch_size, args.margin, args.optimizer, os.path.join(SNN_results_dir, 'snn_params.json'))

        print('Fitting the SNN and the finding the anomaly threshold are done\n')

    else:
        
        snn_params = utils.read_json_file(os.path.join(results_dir, args.nn_type+'/snn_params.json'))
        
        test_original_ts_path = os.path.join(dataset_dir, 'test_original_ts.csv')
        test_ts = utils.load_test_original_ts_from_csv(test_original_ts_path)
        
        test_label_path = os.path.join(dataset_dir, 'test_label.csv')
        true_label = utils.load_test_original_ts_from_csv(test_label_path)

        N = len(test_ts) - dataset_params['m'] + 1
        x_test = utils.split_ts_to_subs(test_ts, N, dataset_params['m'])
        x_test = utils.normalize_dataset(x_test)
    
        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)

        X_test = utils.make_original_test_ts_samples(x_test, x_snippets)
    
        # split the test pairs
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1))
        
        ready_X_test = []
        for i in range(dataset_params['snippets_number']):
            for j in range(num_subs_per_pair):
                ready_X_test.append(X_test[:,i,j])

        SNN_results_dir = os.path.join(results_dir, args.nn_type)
    
        mDiSSiD_model = multi_siamese_nn.build_mDiSSiD_model(snippets_num=dataset_params['snippets_number'], input_shape=(dataset_params['m'], 1), base_type=args.nn_type)
        
        model_path = os.path.join(SNN_results_dir, 'models', 'weights.hdf5')
        if os.path.exists(model_path):
            tf.compat.v1.reset_default_graph()
            mDiSSiD_model.load_weights(model_path)
        else:
            print("The model doesn't exist")
        
        predictions = mDiSSiD_model.predict(ready_X_test)
        
        min_predictions = predictions.min(axis=1)
        
        utils.write_dataset(min_predictions.reshape(-1,1), SNN_results_dir, 'original_test_predictions.csv')

        print(f"DiSSiD accuracy on dataset: {dataset_params['input_files']}")


if __name__ == '__main__':
    main()
