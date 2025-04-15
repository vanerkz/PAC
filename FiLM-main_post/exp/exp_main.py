import os
import sys
from exp.exp_basic import Exp_Basic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Informer, Autoformer, Transformer, Logformer,FiLM,LSTM
#from models.pyraformer import Pyraformer_LR
# from models.reformer_pytorch.reformer_pytorch import Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import scipy.stats as stats
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import time
from data_factory.data_loader import get_loader_segment
import warnings
import matplotlib.pyplot as plt
import numpy as np
from utils.tools import loss_fn
warnings.filterwarnings('ignore')
import pickle

class anomalyc(nn.Module):
    def __init__(self):
        super(anomalyc, self).__init__()
    
        
    def forward(self, dec_label, predict,sigma):
        dec_label=dec_label.permute(0,2,1)
        predict=predict.permute(0,2,1)
        sigma=sigma.permute(0,2,1)
        res=torch.abs(dec_label-predict)
        resbatch=res.reshape(res.shape[0],-1)
        results = []
        for i,batch in enumerate(resbatch):
            batch=batch.cpu().detach().numpy()
            alpha, loc, beta = stats.invgamma.fit(batch)
            #result = minimize(neg_log_likelihood, initial_params, args=(batch,), bounds=((1e-5, None), (1e-5, None)))
            results.append([alpha,loc,beta])
        results = np.array(results)
        x,y=results.shape
        alpha=results[:,0].reshape(x,1,1)
        loc=results[:,1].reshape(x,1,1)
        beta = results[:,2].reshape(x,1,1)
        
        sumsigma=sigma**2
        prior=stats.invgamma.logpdf(sumsigma.cpu().detach().numpy(), alpha,loc=loc, scale=beta)

        prior=torch.tensor(prior, device=dec_label.device)
        LL = Normal(0, sigma)

        loss=torch.sum(LL.log_prob(dec_label-predict),1)+torch.sum(prior,1)
        return torch.softmax(-loss,-1)



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.args.root_path, batch_size=self.args.batch_size, win_size=self.args.win_size,
                                               mode='train',
                                               dataset=self.args.dataset)

        self.test_loader, _ = get_loader_segment(self.args.root_path, batch_size=self.args.batch_size, win_size=self.args.win_size,
                                              mode='test',
                                              dataset=self.args.dataset)
        self.thre_loader = self.vali_loader
        self.anomaly = anomalyc()
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Logformer': Logformer,
            'FiLM':FiLM,
            #'S4':S4_model,
            #'S4film':S4_FiLM,
            #'LSTM':LSTM,
            #'Seasonal':Seasonal,
            #'Ficonv':Ficonv,
            #'Pyraformer':Pyraformer_LR,
            #'Reformer': Reformer,
            #'Transformer_sin':Transformer_sin,
            #'Autoformer_sin':Autoformer_sin,
            #'TreeDRNet':TreeDRNet,
            #'HippoFNOformerMulti':HippoFNOformerMulti
            #'HippoFNOformer': HippoFNOformer,
        }
        
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.model=='LSTM':
            all_length = self.args.seq_len + self.args.label_len
            hidden = model.init_hidden(self.args.batch_size,all_length)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim




    def vali(self):
        total_loss = []
        ks_test_96,ks_test_192,ks_test_336,ks_test_720,ks_test_96_back=[],[],[],[],[]
        ks_result=[]
        ks_test_96_raw,ks_test_192_raw,ks_test_336_raw,ks_test_720_raw,ks_test_96_back_raw=[],[],[],[],[]
        self.model.eval()
        #input_len=720
        with torch.no_grad():
            for i, (enc_data,dec_data, labels) in enumerate(tqdm(self.vali_loader)):
                batch_x = enc_data.float().to(self.device)
                batch_y = dec_data.float()
                if self.args.add_noise_vali:
                    batch_x = batch_x + 0.3*torch.from_numpy(np.random.normal(0, 1, size=batch_x.shape)).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,sigma = self.model(batch_x)

                        else:
                            outputs,sigma = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs,sigma  = self.model(batch_x)

                    else:
                        outputs,sigma = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.to(self.device)

                pred = outputs.detach().cpu()
                sigma = sigma.detach().cpu()
                true = batch_y.detach().cpu()

                loss = loss_fn(true,pred,sigma)

                total_loss.append(loss)

                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        if os.path.exists(best_model_path):
            print("load:",setting)
            self.model.load_state_dict(torch.load(best_model_path))
        else:
             print("No File, Train new")


        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (enc_data,dec_data, labels) in enumerate(tqdm(self.train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = enc_data.float().to(self.device)
                batch_y = dec_data.float().to(self.device)
                
                if self.args.add_noise_train:
                    batch_x = batch_x + 0.3*torch.from_numpy(np.random.normal(0, 1, size=batch_x.float().shape)).float().to(self.device)
                    
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            
                            outputs,sigma = self.model(batch_x)
                        else:
                            outputs,sigma = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y.to(self.device)
                        loss = loss_fn(batch_y,outputs,sigma)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs,sigma   = self.model(batch_x)
                    else:
                        outputs,sigma   = self.model(batch_x)
    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    #print(batch_y.shape)
                    batch_y = batch_y.to(self.device)
                    loss = loss_fn(batch_y,outputs,sigma)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    #loss =loss.clone()
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali()

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting,evaluate=False):
        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        if evaluate:
            if os.path.exists(best_model_path):
                print("load:",setting)
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                print("No File")

        self.model.eval()

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, (enc_data,dec_data, labels) in enumerate(self.train_loader):
            batch_x = enc_data.float().to(self.device)
            batch_y = dec_data.float().to(self.device)
            outputs,sigma = self.model(batch_x)
            cri=self.anomaly(batch_y,outputs,sigma)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (enc_data,dec_data, labels) in enumerate(self.vali_loader):
            batch_x = enc_data.float().to(self.device)
            batch_y = dec_data.float().to(self.device)
            outputs,sigma = self.model(batch_x)
            cri=self.anomaly(batch_y,outputs,sigma)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.args.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (enc_data,dec_data, labels) in enumerate(tqdm(self.test_loader)):
            
            batch_x = enc_data.float().to(self.device)
            batch_y = dec_data.float().to(self.device)
            outputs,sigma = self.model(batch_x)
            cri=self.anomaly(batch_y,outputs,sigma)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.2f}, Precision : {:0.2f}, Recall : {:0.2f}, F-score : {:0.2f} ".format(
                accuracy*100, precision*100,
                recall*100, f_score*100))
                
        return accuracy, precision, recall, f_score

