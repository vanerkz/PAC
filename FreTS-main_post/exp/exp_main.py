
from exp.exp_basic import Exp_Basic
from models import DLinear, NLinear, FreTS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_factory.data_loader import get_loader_segment
import os
import time
from torch.autograd import Variable
from utils.tools import loss_fn
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from torch.distributions import Normal
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class anomalyc(nn.Module):
    def __init__(self):
        super(anomalyc, self).__init__()
    
        
    def forward(self, dec_label, predict,sigma):
        dec_label=dec_label.permute(0,2,1)
        predict=predict.permute(0,2,1)
        sigma=sigma.permute(0,2,1)
        res=dec_label-predict
        resbatch=res.reshape(res.shape[0],-1)

        results = []
        for i,batch in enumerate(resbatch):
            batch=batch.cpu().detach().numpy()
            alpha, loc, beta = stats.invgamma.fit(batch)
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
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FreLinear': FreTS
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (enc_data,dec_data, labels) in enumerate(self.vali_loader):
                batch_x = enc_data.float().to(self.device)
                batch_y = dec_data.float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,sigma = self.model(batch_x)
                else:
                    outputs,sigma  = self.model(batch_x)
                outputs = outputs
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
           

        #path = os.path.join(self.args.checkpoints, setting)
        #if not os.path.exists(path):
        #    os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (enc_data,dec_data, labels) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = enc_data.float().to(self.device)
                batch_y = dec_data.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,sigma  = self.model(batch_x)
                        loss = loss_fn(batch_y,outputs,sigma)
                        train_loss.append(loss.item())
                else:
                    
                    outputs,sigma  = self.model(batch_x)
        
                    outputs= outputs
                    batch_y = batch_y.to(self.device)
                    loss = loss_fn(batch_y,outputs,sigma)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, best_model_path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, best_model_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #best_model_path = path + '/' + 'checkpoint.pth'

        
        
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting,evaluate=True):
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
            outputs,sigma  = self.model(batch_x)
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
            outputs,sigma  = self.model(batch_x)
            cri=criterion(batch_y,outputs)
            cri=self.anomaly(batch_y,outputs,sigma)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        test_energy=test_energy.reshape(-1)

        km = KMeans(n_clusters=3)
        df = pd.DataFrame(test_energy, columns=['x'])
        km.fit(df)
        df['label'] =km.fit(df).labels_
        print(km.cluster_centers_)
        print(np.argmax(km.cluster_centers_, axis=0)[0])
        print(df[df['label']==0].count()['label'])
        print(df[df['label']==1].count()['label'])
        #print(df[df['label']==2].count()['label'])
        ratio=(df[df['label']==np.argmax(km.cluster_centers_, axis=0)[0]].count()['label']/len(test_energy))*100
        print(ratio)
        thresh = np.percentile(combined_energy, 100 - ratio)
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

        pred = (test_energy >= thresh).astype(int)

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


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
