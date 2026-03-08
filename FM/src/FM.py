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


from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, FactorizationMachine
import torch
from torch import nn
import numpy as np
import logging
from tqdm import tqdm
import sys
import os
import pandas as pd
from torch import log

EPS = 1e-6

class FM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FM", 
                 gpu=-1, 
                 learning_rate=1e-3,
                 embedding_dim=10, 
                 regularizer=None, 
                 **kwargs):
        super(FM, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer,
                                 **kwargs)
        self.learning_rate = learning_rate
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.expid = kwargs.get('expid')
        self.baseline = kwargs.get('baseline', 'smooth')

        if kwargs.get('normk') != None:
            kwargs['emb_layer'] = self.embedding_layer
            self.share_embedding_layer = True

        self.fm = FactorizationMachine(feature_map,**kwargs)
        self.embedding_dim = embedding_dim
        self.interpolate_n = kwargs.get('interp')

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        
        y_pred = self.fm(X, feature_emb)
        
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def _compute_baseline(self, feature_emb, inter_type='smooth'):
        """
        Calculate baseline embedding for FairFS to reduce baseline bias.
        
        Args:
            feature_emb: Current feature embeddings, shape (batch, total_emb_dim)
            inter_type: Type of baseline calculation
                - 'zero': Zero baseline (simplest)
                - 'mean': Global mean across all features in the batch
                - 'smooth': Smooth baseline - average across all fields (recommended for FairFS)
                - 'sample_mean': Mean across embedding dimensions for each sample
        
        Returns:
            baseline_emb: Baseline embeddings with same shape as feature_emb
        """
        if inter_type == 'zero':
            baseline_emb = torch.zeros_like(feature_emb)
        elif inter_type == 'mean':
            # Global mean across all features in the batch
            baseline_emb = torch.mean(feature_emb, axis=0)
        elif inter_type == 'smooth':
            # Smooth baseline: average embedding across all fields for each sample
            # This provides a feature-agnostic baseline to reduce baseline bias
            batch_size = feature_emb.shape[0]
            total_width = feature_emb.shape[1]
            segment_len = self.embedding_dim     
            num_segments = total_width // segment_len
            
            # Reshape to separate each field: (batch, num_fields, emb_dim)
            reshaped_emb = feature_emb.view(batch_size, num_segments, segment_len)
            
            # Average across all fields for each sample: (batch, emb_dim)
            avg_emb = reshaped_emb.mean(dim=1)
            
            # Expand back to match original shape: (batch, num_fields, emb_dim) -> (batch, total_width)
            baseline_emb = avg_emb.unsqueeze(1).repeat(1, num_segments, 1).view(batch_size, total_width)
        elif inter_type == 'sample_mean':
            # Mean across embedding dimensions for each sample
            row_means = torch.mean(feature_emb, dim=1)
            baseline_emb = row_means.unsqueeze(1).expand_as(feature_emb)
        else:
            raise ValueError(f"Invalid inter_type: {inter_type}")
        
        return baseline_emb

    def forward_with_embig(self, inputs, inter_type='smooth'):
        '''Used for training only'''

        X = self.get_inputs(inputs)
        self.feature_emb = self.embedding_layer(X).flatten(start_dim=1)
        self.feature_emb.requires_grad_(requires_grad=True)
        
        # Calculate baseline (feature_emb_mean) for FairFS
        self.feature_emb_mean = self._compute_baseline(self.feature_emb, inter_type)
        
        # Reshape back for FM layer
        batch_size = self.feature_emb.shape[0]
        total_width = self.feature_emb.shape[1]
        num_fields = total_width // self.embedding_dim
        feature_emb_3d = self.feature_emb.view(batch_size, num_fields, self.embedding_dim)

        y_pred = self.fm.forward_intp(feature_emb_3d)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_fmcr(self, inputs, inter_type = 'zero', seed=2019):
        '''Used for evaluating and getting importance for each field, NOT FOR TRAINING'''
        X = self.get_inputs(inputs)
        torch.manual_seed(seed)

        feature_emb_noflat = self.embedding_layer(X)
        self.feature_emb = feature_emb_noflat.flatten(start_dim=1)
        bt, field_n, emb_dim = feature_emb_noflat.shape
        self.feature_emb.requires_grad_(requires_grad=True)

        # Calculate baseline for integrated gradients
        self.feature_emb_mean = self._compute_baseline(self.feature_emb, inter_type)

        self.feature_emb_list = [self.feature_emb]
        if hasattr(self, "interpolate_n") and self.interpolate_n > 0:
            self.feature_emb_delta_step = (self.feature_emb - self.feature_emb_mean) / self.interpolate_n
        for i in range(self.interpolate_n):
            self.feature_emb_list.append(self.feature_emb - (i + 1) * self.feature_emb_delta_step)
        self.feature_emb = torch.concat(self.feature_emb_list, dim=0)
        self.feature_emb.retain_grad()

        y_pred = self.fm.forward_intp(self.feature_emb.view(-1,field_n,emb_dim))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def evaluate_with_fmcr(self, data_generator, inter_type = 'zero', metrics=None, seed=2019):
        y_pred = []
        y_true = []
        group_id = []

        fmcr_score_final_result = None

        data_generator = tqdm(data_generator, disable=False, file=sys.stdout)

        for batch_data in data_generator:
            return_dict = self.forward_with_fmcr(batch_data, inter_type=inter_type, seed=seed)

            y_true_fmcr = self.get_labels(batch_data)
            y_true_fmcr = y_true_fmcr.repeat(self.interpolate_n + 1, 1)
            loss = self.compute_loss(return_dict, y_true_fmcr)
            loss.backward()

            fmcr_gradient = self.feature_emb.grad

            emb_size_sum = self.feature_emb.shape[1]
            field_n = batch_data.shape[1] - 1
            emb_size_single = int(emb_size_sum / field_n)

            fmcr_field_gradient = torch.split(fmcr_gradient, emb_size_single, dim=1)
            fmcr_field_delta = [i.repeat(self.interpolate_n + 1, 1) for i in
                                torch.split(self.feature_emb_delta_step, emb_size_single, dim=1)]

            fmcr_loss_delta = []
            for i in range(field_n):
                fmcr_loss_delta.append(torch.einsum('ij,ij->i', fmcr_field_gradient[i],
                                                    fmcr_field_delta[i]).data.cpu().mean().detach().numpy())

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

        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': fmcr_score_final_result.tolist()})
        return feature_importance_result, val_logs['logloss']

    def evaluate_with_fmcr_native(self, data_generator, metrics=None, **kwargs):
        self.eval()
        feature_importance_result, native_log_loss = self.evaluate_with_fmcr(data_generator, 
                                                                             inter_type=kwargs.get('baseline', 'zero'),
                                                                             metrics=None)
        feature_importance_result_sorted = feature_importance_result.sort_values(by='feature_weight', ascending=False)
        feature_importance_result_sorted['cumsum_feature_weight'] = feature_importance_result_sorted[
            'feature_weight'].cumsum()
        logging.info('================= Fast MCR Result =================')
        logging.info(feature_importance_result_sorted)

        normk = kwargs.get('normk')
        baseline = kwargs.get('baseline')
        interp = kwargs.get('interp')

        dir_name = os.path.dirname(self.checkpoint)
        suffix = f'_k={normk}_baseline={baseline}_itp={interp}'
        file_name = os.path.join(dir_name, f'feature_importance_result{suffix}.csv')

        feature_importance_result_sorted.to_csv(file_name, index=False)
        return native_log_loss, feature_importance_result

    def fit_for_embig(self, data_generator, epochs=1, validation_data=None,
                   max_gradient_norm=10., **kwargs):
        """
        Training function implementing FairFS (unbiased Feature Importance Regularization).
        It addresses Layer Bias, Baseline Bias, and Approximation Bias in deep feature selection.
        """
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

        # 'normk' corresponds to the regularization coefficient lambda (phi) in the paper
        k = kwargs.get('normk', 1e-2) 
        print('Using FairFS (EmbIG) to train with lambda {}, baseline: {}...'
            .format(k, kwargs.get('baseline', 'smooth')))

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        
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

                # --- FairFS Core Logic Start ---
                
                # Step 1: Forward pass to compute the task loss (e.g., Cross-Entropy)
                # The forward_with_embig method calculates the current feature embeddings (e_i) 
                # and the smoothed baseline feature (~e_i).
                y_true = self.get_labels(batch_data)
                return_dict = self.forward_with_embig(batch_data, inter_type = kwargs.get('baseline', 'smooth'))
                loss_task = self.compute_loss(return_dict, y_true)

                # Step 2: Compute first-order gradients (dL / de_i)
                # We use autograd.grad with create_graph=True to maintain the gradient in the 
                # computational graph, enabling second-order gradient computation.
                grads = torch.autograd.grad(
                    outputs=loss_task,
                    inputs=self.feature_emb,
                    grad_outputs=torch.ones_like(loss_task),
                    create_graph=True,
                    retain_graph=True
                )[0]

                # Step 3: Estimate Feature Importance I(e_i)
                # Following Formula (4): I(e_i) ≈ <grad_e_i_L, (e_i - ~e_i)>
                # This captures the contribution across all non-linear layers to mitigate Layer Bias.
                # 'delta_e' represents the distance to the proximal smoothing baseline.
                delta_e = self.feature_emb - self.feature_emb_mean
                
                # Compute the inner product (dot product) to condense the field importance into a scalar.
                importance = torch.sum(grads * delta_e, dim=1) 

                # Step 4: Calculate Regularization Loss
                # Following Formula (8): L_reg = phi * ||I(E)||_2.
                # This directly sparsifies sensitive feature importance during training.
                reg_loss = k * torch.norm(importance, p=2)

                # Step 5: Combined Loss and Backward pass
                # Total Loss = Task Loss + Feature Importance Regularization.
                # Backward() here computes gradients of parameters, including the 2nd-order gradients from reg_loss.
                total_loss = loss_task + reg_loss
                total_loss.backward()
                

                # Standard gradient clipping to ensure training stability
                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += total_loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    self.eval_embig()
                    
                if self._stop_training:
                    break

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))

        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        
        # Extra save as checkpoint with hyperparameters in filename
        baseline = kwargs.get('baseline', 'smooth')
        self.save_weights(self.checkpoint.replace('.model', f'expid={self.expid}_k={k}_baseline={baseline}.model'))
        return

    def eval_embig(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        self.eval()  # set to evaluation mode
        data_generator = self.valid_gen
        metrics = self._monitor.get_metrics()
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward_with_embig(batch_data)
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

    def load_weights_with_k(self, k):
        data_path = self.checkpoint.replace('.model', f'_{k}.model')
        logging.info("Load pre-trained model: {}".format(data_path))
        self.load_weights(data_path)
        return
