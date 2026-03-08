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

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='WideDeep_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--cp', type=str, help='checkpoint path or model name')
    parser.add_argument('--normk', nargs='+',type=float, help='search range for the norm k')
    parser.add_argument('--interp', default= 5, type=int, help='interpolation number')
    parser.add_argument('--baseline', default='smooth', type=str, help='baseline inputs')
    parser.add_argument('--stage', type=str, default='all', choices=['pretrain', 'eval', 'all'],
                        help='Running stage: pretrain (train only), eval (evaluate only), all (train + eval)')
    args = vars(parser.parse_args())
    
    # Validate arguments based on stage
    # if args['stage'] == 'eval' and args.get('cp') is None:
    #     parser.error("--stage eval requires --cp (checkpoint path or model name) to be specified")

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)

    params['status'] = 'embig'
    params['gpu'] = args['gpu']
    params['interp'] = args['interp']
    params['baseline'] = args['baseline']
    params['expid'] = args['expid']

    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    logging.info(f"Running stage: {args['stage']}")
    seed_everything(seed=params['seed'])

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

    cur_time = datetime.now()
    formatted_time = cur_time.strftime("%m%d%H%M")
    
    stage = args['stage']
    logging.info(f"========== Running in '{stage}' stage ==========")

    for normk in args.get('normk'):
        params['normk'] = normk
        params['cur_time'] = formatted_time
        model_class = getattr(model_zoo, params['model'])
        print('params[model]', params['model'])
        model = model_class(feature_map, **params)
        model.count_parameters()  # print number of parameters used in model

        # ========== Stage: Pretrain or All ==========
        if stage in ['pretrain', 'all']:
            logging.info(f'****** Training Phase: normk={normk}, baseline={args["baseline"]} *******')
            train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
            
            if args.get('cp', None) is not None and stage == 'pretrain':
                logging.info(f'Resuming training from checkpoint: {args["cp"]}')
                model.load_state_dict(torch.load(args['cp']))
            
            model.fit_for_embig(train_gen, validation_data=valid_gen, **params)
            logging.info(f'Training completed. Model saved.')
            del train_gen, valid_gen
            gc.collect()
        
        # ========== Stage: Eval or All ==========
        if stage in ['eval', 'all']:
            # For eval-only stage, load checkpoint
            if stage == 'eval':
                if args['cp'] is None:
                    args['cp'] = model.checkpoint.replace('.model', f'expid={args["expid"]}_k={normk}_baseline={args["baseline"]}.model')
                logging.info(f'****** Evaluation Phase: Loading checkpoint {args["cp"]} *******')
                model.load_state_dict(torch.load(args['cp']))
            
            logging.info(f'****** Evaluating with FairFS-Eval (Feature Importance): interp={args["interp"]} *******')
            valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()[1]
            native_log_loss, native_feature_importance_result = model.evaluate_with_fmcr_native(valid_gen, **params)
            del valid_gen
            gc.collect()
        
        logging.info(f'===== Completed processing for normk={normk}, baseline={args["baseline"]}, interp={args["interp"]} =====')
