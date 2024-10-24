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
# print(sys.path)

import logging
import pickle
import fuxictr
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import build_dataset
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import importlib
import torch
import numpy as np
import logging
from infomer import email

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--cp', type=str, help='checkpoint path')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    # seed_everything(seed=params['seed'])

    if params.get('spe_processor',None) != None:
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
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    model_class = getattr(model_zoo, params['model'])
    print('params[model]', params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model

    train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()

    # email.common_send('WD_PFI_select.py - Build Data',"")

    if args.get('cp',None) != None:
        print('load model from checkpoint')
        model.load_state_dict(torch.load(args['cp']))
    else:
        model.fit(train_gen, validation_data=valid_gen, **params)

    # email.common_send('WD_PFI_select.py - Training', "")

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    # --- update for pfi start ---
    file = open('exp_metric', 'wb')
    pickle.dump(valid_result, file)
    print('valid_result', valid_result)
    file.close()

    logging.info('****** Validation with PFI *******')
    train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    model.evaluate_with_pfi(valid_gen,valid_result)
    del train_gen, valid_gen
    gc.collect()
    # --- update for pfi end ---

    logging.info('******** Test evaluation ********')
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
        test_result = model.evaluate(test_gen)

    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                 .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                         ' '.join(sys.argv), experiment_id, params['dataset_id'],
                         "N.A.", print_to_list(valid_result), print_to_list(test_result)))



