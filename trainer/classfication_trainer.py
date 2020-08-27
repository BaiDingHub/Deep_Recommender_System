from trainer.base_trainer import *

import torch
import torch.nn as nn
from tqdm import tqdm

class ClassficationTrainer(BaseTrainer):
    """[LeNet+Mnist]
    """
    def __init__(self,model,data_loader,criterion,optimizer,metrics,config):
       super(ClassficationTrainer,self).__init__(model,data_loader,criterion,optimizer,metrics,config)


    def _train_epoch(self,epoch):

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        trainloader_t = tqdm(self.data_loader,ncols=100)
        
        trainloader_t.set_description("Train Epoch: {}|{}  Batch size: {}  LR : {:.4}".format
                                      (epoch,self.EPOCH,self.data_loader.batch_size,self.optimizer.param_groups[0]['lr']))
        
        for idx,(user_feature, movie_feature , y) in enumerate(trainloader_t):
            y = torch.tensor(trans_arr_to_one_hot(y.unsqueeze(-1), class_num=6)).float()
            if self.use_gpu:
                for key in user_feature.keys():
                    user_feature[key] = user_feature[key].cuda(self.device_ids[0])
                for key in movie_feature.keys():
                    movie_feature[key] = movie_feature[key].cuda(self.device_ids[0])
                y = y.cuda(self.device_ids[0])
            # x = x.float()
            # y = y.long()
            self.optimizer.zero_grad()

            logits = self.model(user_feature, movie_feature)
            loss = self.criterion(logits,y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(logits.cpu().detach().numpy(),y.cpu().detach().numpy())
        
        log = {
            'loss' : total_loss /self.len_epoch,
        }
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.len_epoch

        return log


    def _test_epoch(self):
        total_metrics = np.zeros(len(self.metrics))

        test_loader_t = tqdm(self.data_loader,ncols=100)
        test_loader_t.set_description("Batch size: {}".format(self.data_loader.batch_size))

        for idx,(user_feature, movie_feature , y) in enumerate(test_loader_t):
            if self.use_gpu:
                for key in user_feature.keys():
                    user_feature[key] = user_feature[key].cuda(self.device_ids[0])
                for key in movie_feature.keys():
                    movie_feature[key] = movie_feature[key].cuda(self.device_ids[0])
                y = y.cuda(self.device_ids[0])
            # x = x.float()
            y = y.long()

            logits = self.model(user_feature, movie_feature)
            total_metrics += self._eval_metrics(logits.cpu().detach().numpy(),y.cpu().detach().numpy())
        
        log = {}
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.len_epoch

        return log

