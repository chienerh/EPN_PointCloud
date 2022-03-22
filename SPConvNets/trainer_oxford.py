from importlib import import_module
from SPConvNets import Dataloader_Oxford
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import os.path as osp
import SPConvNets.utils as M
# optional, wandb 
import wandb

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.opt.train_loss.equi_alpha > 0:
            self.summary.register(['Loss', 'InvLoss', 'Pos', 'Neg', 'Acc', \
                'EquiLoss', 'EquiPos', 'EquiNeg', 'EquiAcc' ])
        else:
            self.summary.register(['Loss', 'Pos', 'Neg', 'Acc'])

        self.epoch_counter = 0
        self.iter_counter = 0
        self.all_loss = 0
        self.TRAINING_LATENT_VECTORS = []

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_Oxford(self.opt)

            self.dataset_train = torch.utils.data.DataLoader(dataset, \
                                 batch_size=self.opt.batch_size, \
                                 shuffle=False, \
                                 sampler=None, \
                                 num_workers=self.opt.num_thread)

            self.dataset_iter = iter(self.dataset_train)

        if self.opt.mode == 'eval':
            self.dataset_train = None


    def _setup_eval_datasets(self):
        dataset_eval = Dataloader_Oxford(self.opt, mode='eval')
        self.dataset_eval = torch.utils.data.DataLoader(dataset_eval, \
                            batch_size=self.opt.batch_size, \
                            shuffle=False, \
                            num_workers=self.opt.num_thread)

    def _setup_model(self):
        param_outfile = osp.join(self.root_dir, "params.json")
        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)


    def _setup_metric(self):
        self.loss_function = M.quadruplet_loss

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset_train)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset_train)
            data = next(self.dataset_iter)
        self._optimize(data)


    def _prepare_input(self, data):
        in_tensor_anchor = data['anchor'].float().to(self.opt.device)
        in_tensor_positive = data['positive'].float().to(self.opt.device)
        in_tensor_negative = data['negative'].float().to(self.opt.device)
        in_tensor_other_neg = data['other_neg'].float().to(self.opt.device)

        return in_tensor_anchor, in_tensor_positive, in_tensor_negative, in_tensor_other_neg


    def _run_model(self, model, queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor, require_grad=True):
        feed_tensor = torch.cat(
            (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
        feed_tensor = feed_tensor.view((-1, self.opt.num_points, 3)) # ((-1, 1, self.opt.num_points, 3))
        feed_tensor.requires_grad_(require_grad)
        feed_tensor = feed_tensor.to(self.opt.device) # 22, 4096, 3
        if require_grad:
            output, _ = model(feed_tensor)
        else:
            with torch.no_grad():
                output, _ = model(feed_tensor)
        # print('feed_tensor', feed_tensor.shape)
        # print('output', output.shape)
        output = output.view(self.opt.batch_size, -1, self.opt.global_feature_dim)
        # print('output reshape', output.shape)
        # print('batch', self.opt.batch_size, 'output_num', self.opt.model.output_num)
        o1, o2, o3, o4 = torch.split(
            output, [1, self.opt.pos_per_query, self.opt.neg_per_query, 1], dim=1)

        return o1, o2, o3, o4


    def _optimize(self, data):
        queries, positives, negatives, other_neg = self._prepare_input(data)
        
        self.optimizer.zero_grad()
            
        output_queries, output_positives, output_negatives, output_other_neg = self._run_model(
                                self.model, queries, positives, negatives, other_neg)

        loss = self.loss_function(output_queries, output_positives, output_negatives, output_other_neg, \
                                self.opt.MARGIN_1, self.opt.MARGIN_2, use_min=self.opt.TRIPLET_USE_BEST_POSITIVES, \
                                lazy=self.opt.LOSS_LAZY, ignore_zero_loss=self.opt.LOSS_IGNORE_ZERO_BATCH) # config???
        loss.backward()

        self.optimizer.step()
        self.all_loss+=loss

        # Log training stats
        log_info = {
            'Loss': loss,
        }
        self.summary.update(log_info)
        self.iter_counter += 1

        # optional wandb
        wandb.log({"Training loss": loss})


    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')

    def test(self):
        pass

    def eval(self, select):
        '''
            Validation on validation dataset
        '''
        self.model.eval()
        self.optimizer.zero_grad()

        # set up where to store the output feature
        self._setup_eval_datasets()
        eval_batches_counted = 0

        for it, data in enumerate(self.dataset_eval):
            eval_queries, eval_positives, eval_negatives, eval_other_neg = self._prepare_input(data)

            output_queries, output_positives, output_negatives, output_other_neg = self._run_model(
                self.model, eval_queries, eval_positives, eval_negatives, eval_other_neg, require_grad=False)
            e_loss = self.loss_function(output_queries, output_positives, output_negatives, output_other_neg, self.opt.MARGIN_1, self.opt.MARGIN_2, use_min=self.opt.TRIPLET_USE_BEST_POSITIVES, lazy=self.opt.LOSS_LAZY, ignore_zero_loss=self.opt.LOSS_IGNORE_ZERO_BATCH)
            self.optimizer.step()
            eval_loss+=e_loss
            eval_batches_counted+=1
        average_eval_loss= float(eval_loss)/eval_batches_counted

        # Log training stats
        log_info = {
            'Validation Loss': average_eval_loss,
        }
        self.summary.update(log_info)

        # optional wandb
        wandb.log({"Validation loss": average_eval_loss})
