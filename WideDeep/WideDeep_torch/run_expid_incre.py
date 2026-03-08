# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append('../../../fuxictr')
sys.path.append('../../../')

import logging
import pickle
import copy
import fuxictr_version
import numpy as np
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import importlib
import glob
import pandas as pd
import glob

def get_feature_indices(experiment_id, df):
    """
    get the feature indices for the experiment
    Args:
        experiment_id (str): the experiment id
        df (pd.DataFrame): the dataframe
    Returns:
        list: the feature indices
    """
    experiment_id = experiment_id.lower()  # 

    if "criteo" in experiment_id:
        incre = 1
        start = 29 + incre - 1
    elif "avazu" in experiment_id:
        incre = 1
        start = 9 + incre - 1
    elif "iflychu" in experiment_id:
        incre = 10
        start = incre - 1
    else:
        raise ValueError(f"Unsupported experiment_id: {experiment_id}")

    return list(range(start, df.shape[0] + incre - 1, incre))


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='WideDeep_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--feat_file', type=str, help='The feature importance file.')
    parser.add_argument('--K', type=int, default=None, help='If set, only evaluate with this number of top features; overrides i_list logic.')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params['expid'] = experiment_id
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    if params.get('spe_processor'):
        module_name = f"fuxictr.datasets.{params['spe_processor']}"
        fp_module = importlib.import_module(module_name)
        assert hasattr(fp_module, 'FeatureProcessor')
        FeatureProcessor = getattr(fp_module, 'FeatureProcessor')
    else:
        from fuxictr.preprocess import FeatureProcessor

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)

    # here specify feature numbers
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Get checkpoint dir from a temporary model to find feature importance files
    temp_model_class = getattr(model_zoo, params['model'])
    temp_model = temp_model_class(feature_map, **params)
    dir_name = os.path.dirname(temp_model.checkpoint)
    output_log = os.path.join(dir_name, 'incre_eval_results.log')
    if args.get('feat_file'):
        files = [args['feat_file']]
    else:
        file_pattern = os.path.join(dir_name, 'feature_importance_result_k*.csv')
        files = glob.glob(file_pattern)
        
        del temp_model
        
        logging.info(f"Found {len(files)} feature importance files: {files}")

    # Outer loop: iterate over feature importance files
    for file in files:
        file_basename = os.path.basename(file)
        logging.info(f"===== Processing file: {file_basename} =====")
        
        # get importance
        df = pd.read_csv(file)

        # sort df by feature_weight
        df = df.sort_values(by='feature_weight', ascending=False)

        logging.info(f"Feature importance file loaded: {file_basename}, total features: {df.shape[0]}")

        topk_column_name = []
        topk_logloss_result = []
        topk_auc_result = []
        time_consumption = []
        inf_times = []

        if args.get('K') is not None:
            num_features = args['K']
            if num_features < 1:
                raise ValueError(f"--K must be >= 1, got {num_features}")
            # --K is num_features directly; use K-1 as index to match loop (num_features = i + 1)
            i_list = [num_features - 1]
        else:
            i_list = get_feature_indices(experiment_id=experiment_id, df=df)
        
        for i in i_list:
            i = min(i, df.shape[0] - 1)
            num_features = i + 1

            search_str = f"{file_basename}, num_features={num_features}, "
            logging.info(f"Searching for {search_str} in {output_log}")
            if os.path.exists(output_log):
                with open(output_log, "r", encoding="utf-8") as f:
                    file_content = f.read()
            
                if search_str in file_content:
                    logging.info(f"Record already exists for {file_basename}, num_features={num_features}, skipping...")
                    continue

            seed_everything(seed=params['seed'])
            topk_params = copy.deepcopy(params)
            topk_params['use_features'] = df['feature_name'].values.tolist()[:num_features]
            logging.info('--- Used Features: {} (totally {} features)'.format(topk_params['use_features'], num_features))
            topk_feature_map = FeatureMap(topk_params['dataset_id'], data_dir)
            topk_feature_map.load(feature_map_json, topk_params)

            model_class = getattr(model_zoo, topk_params['model'])
            topk_model = model_class(topk_feature_map, **topk_params)
            topk_model.count_parameters()  # print number of parameters used in model

            train_gen,valid_gen,test_gen = H5DataLoader(topk_feature_map, stage='both', **topk_params).make_iterator()

            start_time = datetime.now()
            topk_model.fit(train_gen, validation_data=valid_gen, **params)
            time_consumption.append((datetime.now() - start_time).seconds)

            start_time = datetime.now()
            valid_result = topk_model.evaluate(test_gen)
            inf_times.append((datetime.now() - start_time).seconds)

            topk_column_name.append('topk_{}_with_{}'.format(num_features, topk_params['use_features']))
            topk_logloss_result.append(valid_result['logloss'])
            topk_auc_result.append(valid_result['AUC'])
            cur_AUC = valid_result['AUC']

            # Save results incrementally
            with open(output_log, 'a') as f:
                f.write(f"{file_basename}, num_features={num_features}, "
                        f"logloss={valid_result['logloss']}, AUC={valid_result['AUC']}, "
                        f"train_time={time_consumption[-1]}s, inf_time={inf_times[-1]}s\n")

            del train_gen, valid_gen, test_gen
            gc.collect()
        
        logging.info(f"===== Completed file: {file_basename} =====")
