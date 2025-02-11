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

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNet
import numpy as np
import logging
from tqdm import tqdm
import sys
import pandas as pd
# from utils import *
from torch import log
import pandas as pd
from feat_select.AdaFS_module import AdaFS,AdaFS_hard
import torch.nn.functional as F
from model_zoo.utils import get_gates_prob_autofield,get_gates_prob_my,get_sum_feature_dimisions
from itertools import cycle
import torch.optim as optim
import feat_select.MvFS_module as Mv

EPS = 1e-6
class DCN(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="DCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 select_num=0,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0.3,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.nanchor = kwargs.get('nanchor')
        self.mcdropout_n = int(net_dropout * 100)

        input_dim = feature_map.sum_emb_out_dim()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, num_cross_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        if select_num > 0:
            self.controller = Mv.MvFS_Controller(input_dim=get_sum_feature_dimisions(self.embedding_layer),
                                                 embed_dims=len(self.feature_map.features), num_selections=select_num)

        # # --- update for droprank start---
        # self.gates_theta = nn.Parameter(torch.ones(len(self.feature_map.features)) * 0.5)
        # self.gates_theta.requires_grad_(requires_grad=True)
        # # --- update for droprank end---

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)

        # Put it after compile!!!! update it use self optimizer
        # --- update for AdaFS start---
        if kwargs.get('mode', None):
            if kwargs.get('mode') == 1:  # soft
                self.adafs = AdaFS(feature_map.num_fields, embedding_dim)
            else: # hard
                self.adafs = AdaFS_hard(feature_map.num_fields,embedding_dim,
                                        select_num=kwargs.get('select_num'))
        # --- update for AdaFS end---

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_total_parameters(self):
        # only embedding layer and controller params will be counted
        params = 0

        # embedding layer
        for emb in  self.embedding_layer.embedding_layer.embedding_layers.values():
            params += sum(p.numel() for p in emb.parameters())

        # controller
        if hasattr(self,'controller'):
            for param in self.controller.parameters():
                params += param.numel()

        return params

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_embig(self, inputs):
        X = self.get_inputs(inputs)
        self.feature_emb = self.embedding_layer(X, flatten_emb=True)
        self.feature_emb.requires_grad_(requires_grad=True)
        cross_out = self.crossnet(self.feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(self.feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_fmcr(self, inputs, type = 'zero',seed=2019):
        X = self.get_inputs(inputs)
        torch.manual_seed(seed)

        # --- update for fmcr ---
        self.feature_emb = self.embedding_layer(X, flatten_emb=True)
        self.feature_emb.requires_grad_(requires_grad=True)


        if type == 'zero':
            self.feature_emb_mean = torch.zeros_like(self.feature_emb)
        elif type == 'mean':
            self.feature_emb_mean = torch.mean(self.feature_emb, axis=0)
        elif type == 'sample_mean':
            row_means = torch.mean(self.feature_emb, dim=1)
            self.feature_emb_mean = row_means.unsqueeze(1).expand_as(self.feature_emb)
        else:
            # raise errors
            pass

        self.feature_emb_delta_step = (self.feature_emb - self.feature_emb_mean) / self.nanchor
        self.feature_emb_list = [self.feature_emb]
        for i in range(self.nanchor):
            self.feature_emb_list.append(self.feature_emb - (i + 1) * self.feature_emb_delta_step)
        self.feature_emb = torch.concat(self.feature_emb_list, dim=0)
        self.feature_emb.retain_grad()
        # --- update for fmcr ---

        cross_out = self.crossnet(self.feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(self.feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_dr(self,inputs,gates_prob):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]
        # --- update for droprank end---

        feature_emb = feature_emb.flatten(start_dim=1)

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_adafs(self,inputs):
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in AdaFS
        feature_emb = self.adafs(feature_emb.transpose(1,2))

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_adafs_eval(self,inputs,seed=2019):
        self.eval()
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in AdaFS
        batch_weight = self.adafs.forward_for_eval(feature_emb.transpose(1,2))
        return batch_weight

    def forward_with_mvfs(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        # The embedding process for X will be in MvFS
        self.weight = self.controller(feature_emb)
        selected_field = feature_emb * torch.unsqueeze(self.weight, 2)
        feature_emb = selected_field.flatten(start_dim=1)

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    # def forward_with_dr(self, inputs, gates_prob, seed=2019):
    #     return self.forward(inputs,seed)
    def eval_mvfs(self,data_generator = None):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        self.eval()  # set to evaluation mode
        if data_generator is None:
            data_generator = self.valid_gen
        metrics = self._monitor.get_metrics()
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward_with_mvfs(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        super().checkpoint_and_earlystop(val_logs)
        self.train()

    def evaluate_with_fmcr(self, data_generator, baseline = 'zero',metrics=None, seed=2019):
        y_pred = []
        y_true = []
        group_id = []

        fmcr_score_final_result = None

        data_generator = tqdm(data_generator, disable=False, file=sys.stdout)

        for batch_data in data_generator:
            # require_grad for feature_emb for we will use the gradient on it.
            return_dict = self.forward_with_fmcr(batch_data, baseline,seed)

            # 进行一次梯度回传
            y_true_fmcr = self.get_labels(batch_data)
            y_true_fmcr = y_true_fmcr.repeat(self.nanchor + 1, 1)
            loss = self.compute_loss(return_dict, y_true_fmcr)
            loss.backward()

            # Get the gradient of feature_emb as the importance
            fmcr_gradient = self.feature_emb.grad

            # 计算fmcr单batch指标
            emb_size_sum = self.feature_emb.shape[1]
            field_n = batch_data.shape[1] - 1
            emb_size_single = int(emb_size_sum / field_n)

            fmcr_field_gradient = torch.split(fmcr_gradient, emb_size_single, dim=1)
            fmcr_field_delta = [i.repeat(self.nanchor + 1, 1) for i in
                                torch.split(self.feature_emb_delta_step, emb_size_single, dim=1)]

            fmcr_loss_delta = []
            for i in range(field_n):
                fmcr_loss_delta.append(torch.einsum('ij,ij->i', fmcr_field_gradient[i],
                                                    fmcr_field_delta[i]).data.cpu().mean().detach().numpy())

            # 计算fmcr累计batch指标
            if fmcr_score_final_result is None:
                fmcr_score_final_result = np.abs(np.array(fmcr_loss_delta))
            else:
                fmcr_score_final_result += np.abs(np.array(fmcr_loss_delta))
            self.optimizer.zero_grad()

            y_true_tmp = self.get_labels(batch_data).data.cpu().numpy().reshape(-1)
            y_true.extend(y_true_tmp)
            y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1)[:len(y_true_tmp)])

        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None

        if metrics is not None:
            val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
        else:
            val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))

        # 处理成可读的特征重要性指标
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': fmcr_score_final_result.tolist()})
        return feature_importance_result, val_logs['logloss']

    def evaluate_with_fmcr_native(self, data_generator, metrics=None, **kwargs):
        self.eval()
        feature_importance_result, native_log_loss = self.evaluate_with_fmcr(data_generator, metrics=None)
        feature_importance_result_sorted = feature_importance_result.sort_values(by='feature_weight', ascending=False)
        feature_importance_result_sorted['cumsum_feature_weight'] = feature_importance_result_sorted[
            'feature_weight'].cumsum()
        logging.info('================= Fast MCR Result =================')
        logging.info(feature_importance_result_sorted)

        normk = kwargs.get('normk')
        base_method = kwargs.get('baseline')
        nanchor = kwargs.get('nanchor')

        suffix = f'_k={normk}_m={base_method}_itp={nanchor}'

        file_name = f'feature_importance_result{suffix}.csv'

        feature_importance_result_sorted.to_csv(file_name, index=False)
        return native_log_loss, feature_importance_result

    def evaluate_with_pfi(self, data_generator, valid_result = None, metrics=None, seed=2019):
        pfi_score_res = pd.DataFrame(columns=['AUC', 'logloss'])

        ## FIXME: 1 means the label column
        total_field = len(self.feature_map.features) +  1
        for feat_idx in range(total_field):
            if feat_idx == self.feature_map.get_column_index(self.feature_map.labels[0]):
                continue
            perm_gen = self._permute_feature(data_generator, feat_idx)
            cur_result = self.evaluate(perm_gen, metrics=metrics)
            diff = pd.DataFrame([{
                'AUC': cur_result['AUC'] - valid_result['AUC'],
                'logloss': cur_result['logloss'] - valid_result['logloss']
            }])
            pfi_score_res = pd.concat([pfi_score_res, diff], ignore_index=True)

        # Add feature name
        pfi_score_res.insert(0,'feature_name',list(self.feature_map.features.keys()))

        # Get the absolute value of AUC & logloss
        pfi_score_res['AUC'] = pfi_score_res['AUC'].abs()
        pfi_score_res['logloss'] = pfi_score_res['logloss'].abs()

        # Sort by AUC and see AUC as the feature_weight
        pfi_score_res = pfi_score_res.sort_values(by='AUC',ascending=False)
        pfi_score_res.insert(1, 'feature_weight', pfi_score_res['AUC'])
        pfi_score_res.to_csv('feature_importance_result.csv',index=False)
        return

    def evaluate_with_adafs(self, data_generator,seed = 2019):
        self.eval()
        data_generator = tqdm(data_generator, disable=False, file=sys.stdout)

        weight = torch.zeros(1,len(self.feature_map.features))

        for batch_data in data_generator:
            # Get the batch weight from validation dataset, and will get a (n*feature_num) matrix
            batch_weight = self.forward_with_adafs_eval(batch_data, seed)
            batch_weight_use = batch_weight.cpu().detach()
            batch_weight_use = batch_weight_use.sum(dim=0,keepdim=True)
            # print('batch_weight_use', batch_weight_use)
            weight += batch_weight_use

        # 2-Norm the weight
        weight /= torch.norm(weight)

        # 处理成可读的特征重要性指标
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': np.squeeze(weight.numpy()).tolist()})

        feature_importance_result = feature_importance_result.sort_values(by='feature_weight', ascending=False)
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)

        return


    def fit_for_dr(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        # Make a list of theta for each feature
        gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        gates_theta.requires_grad_(requires_grad=True)

        gates_sigma = torch.ones(len(self.feature_map.features)) * 0.5
        gates_sigma.requires_grad_(requires_grad=True)

        # --- update for droprank start---
        self.optimizer.add_param_group({'params': gates_theta, 'lr': self.learning_rate * 0.1})
        self.optimizer.add_param_group({'params': gates_sigma, 'lr': self.learning_rate * 0.1})
        # --- update for droprank end---

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # --- update for droprank start---
                gates_prob = get_gates_prob_my(gates_theta, gates_sigma)
                return_dict = self.forward_with_dr(batch_data, gates_prob)
                # --- update for droprank end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                # --- update for droprank start---
                loss += (torch.sum(gates_prob)) * 1e-5

                ####
                # dot = make_dot(loss, params=dict(self.named_parameters()))
                # dot.view()
                ####

                # --- update for droprank end---

                loss.backward()

                # # --- update for droprank start---
                # print("\n",gates_theta.grad,"\n")
                # # --- update for droprank end---

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

                # --- update for droprank start---
            logging.info("\n Gates Theta: {}".format(gates_theta))
            # --- update for droprank end---

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        print(gates_theta)
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': gates_theta.tolist(),
                                                  'feature_sigma': gates_sigma.tolist()})
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)
        return

    def fit_for_autofield(self, data_generator, epochs=1, validation_data=None,
                          max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0

        self._freq = 5

        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        cyc_val = cycle(validation_data)
        # Make a list of theta for each feature
        gates_a1 = torch.ones(len(self.feature_map.features)) * 0.5
        gates_a1.requires_grad_(requires_grad=True)

        # --- update for droprank start---
        # self.optimizer.add_param_group({'params': gates_a1, 'lr': self.learning_rate * 0.1})
        # --- update for droprank end---

        gates_optimizer = optim.Adam([gates_a1], lr=self.learning_rate * 0.1)
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # --- update for droprank start---

                gates_a1.data = torch.clamp(gates_a1.data, min=EPS, max=1)
                gates_prob = get_gates_prob_autofield(gates_a1, epoch=self._total_steps)

                return_dict = self.forward_with_dr(batch_data, gates_prob)
                # --- update for droprank end---

                self.optimizer.zero_grad()
                gates_optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss.backward()

                train_loss += loss.item()

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                if self._total_steps % self._freq == 0:
                    # A new computation graph is built
                    gates_prob = get_gates_prob_autofield(gates_a1, epoch=self._total_steps)

                    batch_val_data = next(cyc_val)
                    gates_optimizer.zero_grad()
                    return_dict = self.forward_with_dr(batch_val_data, gates_prob)
                    y_true = self.get_labels(batch_val_data)
                    loss_val = self.compute_loss(return_dict, y_true)
                    loss_val.backward()
                    gates_optimizer.step()

                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

                # --- update for droprank start---
            logging.info("\n Autofield Gates Theta: {}".format(gates_a1))
            # --- update for droprank end---

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        print(gates_a1)
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': gates_a1.tolist()})
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)
        return

    def fit_for_embig(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        k = kwargs.get('normk', 1e2)
        print('Using EmbIG Norm to train with normk {}, anchor numer {}, baseline methods {}......'
              .format(k, kwargs['nanchor'], kwargs['baseline']))

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                self.optimizer.zero_grad()

                # --- update for embig start---
                y_true = self.get_labels(batch_data)
                return_dict = self.forward_with_embig(batch_data)
                loss = self.compute_loss(return_dict, y_true)
                # grads = torch.autograd.grad(loss, self.feature_emb,
                #                             grad_outputs=torch.ones_like(loss), create_graph=True)

                emb_size_sum = self.feature_emb.shape[1]
                field_n = batch_data.shape[1] - 1

                # Get L2-Norm of grads and add it to loss
                # grad = torch.mean(grads[0],dim=0)
                # re_grads = grad.view(-1, self.embedding_dim)
                # re_grads = torch.mean(re_grads, dim=1)
                # re_grads = torch.norm(re_grads, dim=0)
                # loss += k*re_grads
                loss.backward()
                # --- update for embig end---

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))

        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        return

    def fit_for_adafs(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0

            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator

            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # --- update for adafs start---
                return_dict = self.forward_with_adafs(batch_data)
                # --- update for adafs end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss.backward()

                #print(self.adafs.controller.mlp.mlps[0][0].weight.grad)

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()


                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_step()
                if self._stop_training:
                    break

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        return

    def fit_for_mvfs(self, data_generator, epochs=1, validation_data=None,
                     max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                return_dict = self.forward_with_mvfs(batch_data)
                # --- update for droprank end---

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)
                # --- update for droprank end---
                loss.backward()
                # # --- update for droprank start---
                # print("\n",gates_theta.grad,"\n")
                # # --- update for droprank end---

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
                    self.eval_mvfs()
                if self._stop_training:
                    break
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def _get_gates_prob(self,gates_theta,gates_sigma):
        assert len(gates_theta) == len(gates_sigma), "gates_theta and gates_sigma should have the same length"
        gates_prob = gates_theta.clone()
        for i in range(gates_theta.shape[0]):
            gates_prob[i] = self._get_prob(gates_theta[i],gates_sigma[i])
        return gates_prob

    def _get_prob(self,unit,sigma_unit):
        eps = torch.randn(1)*sigma_unit

        # u = torch.randn(1)*sigma_unit
        # u.requires_grad = False
        # u = u.to(device=self.device)

        return torch.sigmoid((1.0 / 0.1) * (unit + eps))


    def _get_featuremap_size(self,feature_map):
        # 先写死每个维度的emb_size = 32，后续可以改成从feature_map中读取
        return [32]*len(feature_map.features)

    def _permute_feature(self,data_generator, feature_idx):
        """
        Permutes the values of a specific feature in each batch produced by the data_generator.

        Args:
        - data_generator: Original data generator.
        - feature_idx: The index of the feature you want to permute.

        Yields:
        - Batch with permuted feature values.
        """
        for batch in data_generator:
            # Deep copy to avoid modifying the original batch
            permuted_batch = batch.clone()

            # Permute the feature using PyTorch functions
            perm = torch.randperm(permuted_batch.size(0))
            permuted_batch[:, feature_idx] = permuted_batch[perm, feature_idx]

            yield permuted_batch

