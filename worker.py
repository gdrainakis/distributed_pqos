from torch import utils
import matplotlib.pyplot as plt
import time
import math
import os
import random
import sys
try:
    from skmultiflow.drift_detection import DDM
except:
    print('Warning: SKmultiflow not onboarded')


import numpy as np
import gparams
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
from neural_ntw import LSTM_bibatch
from sys import getsizeof
import copy

def get_custom_mse(a,b):
    res=mean_squared_error(a, b, squared=False)
    if res==0:
        res=gparams._RMSE_ZERO_GROUND
    return res

def get_custom_smape_0_100(A, F):
    if len(A)==0:
        res=100
    else:
        res=(100/len(A)) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))
    return res

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class training_set():
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]

class Worker:
    def __init__(self,id):
        self.id=int(id)
        self.subdataset=None
        self.model=None
        self.optimizer=None
        self.scheduler=None
        self.criterion=None
        self.loss=None
        self.rmse=[]
        self.baseline=[]
        self.enhance=[] # todo : not used right now
        self.nrmse=[]
        self.smape=[]
        self.mae=[]
        self.saved_loss=gparams._LOSS_INFINITE_VALUE

        self.reset_model()
        np.set_printoptions(threshold=sys.maxsize)

    def reset_model(self):

        self.model = LSTM_bibatch(num_classes=gparams._HORIZON, input_size=gparams._UNI_MULTI,
                          hidden_size=gparams._HIDDEN_FEATURES, num_layers=gparams._STACKED_LSTM)

        self.reset_model_params()

    def reset_model_params(self):
        if gparams._OPTIMIZER == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=gparams._LEARNING_RATE,
                                          weight_decay=gparams._WEIGHT_DECAY)
        elif gparams._OPTIMIZER == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=gparams._LEARNING_RATE, momentum=0.9,
                                         weight_decay=gparams._WEIGHT_DECAY)
        elif gparams._OPTIMIZER == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=gparams._LEARNING_RATE,
                                             weight_decay=gparams._WEIGHT_DECAY)
        else:
            print('Cannot find optimizer')
            exit(0)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10,
                                                            threshold=0.0001, threshold_mode='rel', verbose=False)

        if gparams._LOSS_FUNC == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif gparams._LOSS_FUNC == 'RMSLE':
            self.criterion = RMSLELoss()
        elif gparams._LOSS_FUNC == 'L1LOSS':
            self.criterion = torch.nn.L1Loss()
        elif gparams._LOSS_FUNC == 'HuberLoss':
            self.criterion = torch.nn.HuberLoss()
        elif gparams._LOSS_FUNC == 'DILATE':
            print('custom dilate')
            exit(0) # todo
        else:
            print('Cannot find loss func!')
            exit(0)

    def infer_base(self,input_model=None,plot_id=-1,curr_round=0):
        # model choice
        if ((input_model=='existing') or (input_model is None)):
            selected_model=self.model
        else:
            selected_model=input_model
        selected_model.eval()
        if gparams._STATEFUL:
            selected_model.reset_states()

        infer_input_x = self.subdataset.curr_tf_infer_x
        infer_truth_y = self.subdataset.curr_tf_infer_y
        infer_pred_y = selected_model(infer_input_x)
        infer_norm_matrix = self.subdataset.curr_infer_normalization_matrix

        num_pred_y = infer_pred_y.data.numpy()
        num_truth_y=infer_truth_y.data.numpy()
        if gparams._UNI_MULTI != 0 and gparams._FLAG_NORMALIZE:
            actual_pred_y = infer_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_pred_y)
            actual_truth_y = infer_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_truth_y)
        else:
            actual_pred_y = num_pred_y
            actual_truth_y = num_truth_y

        rms = []
        nrms = []
        smape=[]
        mae=[]

        for i in range(0, gparams._HORIZON):
            my_rmse = get_custom_mse(actual_truth_y[:, i], actual_pred_y[:, i])
            my_smape = get_custom_smape_0_100(actual_truth_y[:, i], actual_pred_y[:, i])
            my_mae = mean_absolute_error(actual_truth_y[:, i], actual_pred_y[:, i])
            q3, q1 = np.percentile(actual_truth_y[:, i], [75, 25])
            iqr = q3 - q1
            if iqr != 0:
                my_nrmse = my_rmse / iqr
            else:
                #print('iqr=0,replacing with MIN_THROUGHPUT')
                my_nrmse = my_rmse/gparams._MIN_THROUGHPUT

            rms.append(my_rmse)
            smape.append(my_smape)
            mae.append(my_mae)
            if gparams._NAIVE_PREDICTION=='single': # nrmse value is nrmse, else is baseline_5
                nrms.append(my_nrmse)

        # Baseline 1
        infer_baseline_y = []
        num_input_x = infer_input_x.data.numpy()

        for row in num_input_x:
            prediction = []
            row_temp = row
            for i in range(0, gparams._HORIZON):
                element = np.mean(row_temp[i:i + gparams._SLIDING_WINDOW, gparams._COL_PREDICTION])
                prediction.append(element)
                row_temp = np.append(row_temp, [row_temp[0]], axis=0)
                row_temp[-1, gparams._COL_PREDICTION] = element
            infer_baseline_y.append(prediction)

        num_baseline_y = np.array(infer_baseline_y)
        num_baseline_y = num_baseline_y.reshape((-1, gparams._HORIZON))

        if gparams._UNI_MULTI != 0 and gparams._FLAG_NORMALIZE:
            actual_baseline_y = infer_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_baseline_y)
        else:
            actual_baseline_y = num_baseline_y

        rms_baseline = []
        for i in range(0, gparams._HORIZON):
            rms_baseline.append(get_custom_mse(actual_truth_y[:, i], actual_baseline_y[:, i]))

        enhance = []
        for hor in range(0, len(rms)):
            en_abs = (rms_baseline[hor] - rms[hor]) / rms_baseline[hor]
            enhance.append(en_abs * 100)

        if plot_id!=-1:
            for hor in [0,3,7]:
                mystr=''
                mystr=mystr+'x='+str(range(len(actual_truth_y[:, hor])))+'\n'
                mystr=mystr+'truth='+str(actual_truth_y[:, hor].tolist())+'\n'
                mystr=mystr+'ml='+str(actual_pred_y[:, hor].tolist())+'\n'
                mystr=mystr+'naive='+str(actual_baseline_y[:, hor].tolist())+'\n'
                mystr=mystr+'scheme_'+str(gparams._ML)+'_client_'+str(plot_id)+'_rnd'+str(curr_round)+'_hor'+str(hor)+'_time_'+str(time.time())+'\n'
                mystr=mystr+'--------------'+'\n'
                with open('plot_quanty', mode='a') as file:
                    file.write(mystr)

                #plt.plot(range(len(actual_truth_y[:, hor])), actual_truth_y[:, hor], label='Ground Truth', color='b')
                #plt.plot(range(len(actual_truth_y[:, hor])), actual_pred_y[:, hor], label='ML Prediction',color='y')
                #plt.plot(range(len(actual_truth_y[:, hor])), actual_baseline_y[:, hor], label='Naive Prediction', color='r')
                #plt.xlabel('Time', fontsize=15, weight='bold')
                #plt.ylabel('Throughput', fontsize=15, weight='bold')
                #plt.legend(loc="upper left", fontsize=15)
                #plt.show()

                #plt.savefig(mystr+'.png')
                #plt.clf()

        if gparams._NAIVE_PREDICTION=='double':
            # Baseline 5
            infer_baseline_y = []
            num_input_x = infer_input_x.data.numpy()

            for row in num_input_x:
                prediction = []
                row_temp = row
                for i in range(0, gparams._HORIZON):
                    element = np.mean(row_temp[i + gparams._SLIDING_WINDOW - 5:i + gparams._SLIDING_WINDOW, gparams._COL_PREDICTION])
                    prediction.append(element)
                    row_temp = np.append(row_temp, [row_temp[0]], axis=0)
                    row_temp[-1, gparams._COL_PREDICTION] = element
                infer_baseline_y.append(prediction)

            num_baseline_y = np.array(infer_baseline_y)
            num_baseline_y = num_baseline_y.reshape((-1, gparams._HORIZON))

            if gparams._UNI_MULTI != 0 and gparams._FLAG_NORMALIZE:
                actual_baseline_y = infer_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_baseline_y)
            else:
                actual_baseline_y = num_baseline_y

            nrms = []
            for i in range(0, gparams._HORIZON):
                nrms.append(get_custom_mse(actual_truth_y[:, i], actual_baseline_y[:, i]))

        return rms,rms_baseline,enhance,nrms,smape,mae

    def validate_base(self,input_model=None):
        # model choice
        if ((input_model == 'existing') or (input_model is None)):
            selected_model=self.model
        else:
            selected_model=input_model
        selected_model.eval()

        if gparams._STATEFUL:
            selected_model.reset_states()
        valid_input_x = self.subdataset.curr_tf_valid_x
        valid_truth_y = self.subdataset.curr_tf_valid_y
        valid_pred_y = selected_model(valid_input_x)
        valid_norm_matrix = self.subdataset.curr_valid_normalization_matrix

        num_pred_y = valid_pred_y.data.numpy()
        num_truth_y=valid_truth_y.data.numpy()
        if gparams._UNI_MULTI != 0 and gparams._FLAG_NORMALIZE:
            actual_pred_y = valid_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_pred_y)
            actual_truth_y = valid_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_truth_y)
        else:
            actual_pred_y = num_pred_y
            actual_truth_y = num_truth_y

        rms = []
        nrms = []
        for i in range(0, gparams._HORIZON):
            my_rmse = get_custom_mse(actual_truth_y[:, i], actual_pred_y[:, i])
            q3, q1 = np.percentile(actual_truth_y[:, i], [75, 25])
            iqr = q3 - q1
            if iqr != 0:
                my_nrmse = my_rmse / iqr
            else:
                #print('iqr=0,replacing with MIN_THROUGHPUT')
                my_nrmse = my_rmse/gparams._MIN_THROUGHPUT
            rms.append(my_rmse)
            nrms.append(my_nrmse)

        # Baseline 1
        infer_baseline_y = []
        num_input_x = valid_input_x.data.numpy()

        for row in num_input_x:
            prediction = []
            row_temp = row
            for i in range(0, gparams._HORIZON):
                element = np.mean(row_temp[i:i + gparams._SLIDING_WINDOW, gparams._COL_PREDICTION])
                prediction.append(element)
                row_temp = np.append(row_temp, [row_temp[0]], axis=0)
                row_temp[-1, gparams._COL_PREDICTION] = element
            infer_baseline_y.append(prediction)

        num_baseline_y = np.array(infer_baseline_y)
        num_baseline_y = num_baseline_y.reshape((-1, gparams._HORIZON))

        if gparams._UNI_MULTI != 0 and gparams._FLAG_NORMALIZE:
            actual_baseline_y = valid_norm_matrix[gparams._COL_PREDICTION].inverse_transform(num_baseline_y)
        else:
            actual_baseline_y = num_baseline_y

        rms_baseline = []
        for i in range(0, gparams._HORIZON):
            rms_baseline.append(get_custom_mse(actual_truth_y[:, i], actual_baseline_y[:, i]))

        enhance = []
        for hor in range(0, len(rms)):
            en_abs = (rms_baseline[hor] - rms[hor]) / rms_baseline[hor]
            enhance.append(en_abs * 100)

        return rms,rms_baseline,enhance,nrms

    def train_base(self,input_model=None,saved_name='lstm.pt'):
        # model choice
        if ((input_model == 'existing') or (input_model is None)):
            selected_model=self.model
        else:
            selected_model=input_model
        selected_model.train()

        # transform tf dataset into pytorch dataloader
        training_dataset = training_set(self.subdataset.curr_tf_train_x, self.subdataset.curr_tf_train_y)
        validation_dataset = training_set(self.subdataset.curr_tf_valid_x, self.subdataset.curr_tf_valid_y)
        if gparams._BATCH_SIZE <= 0:
            curr_batch_size = training_dataset.__len__()
        else:
            curr_batch_size=gparams._BATCH_SIZE
        train_loader = utils.data.DataLoader(training_dataset, batch_size=curr_batch_size, shuffle=False)
        #valid_loader = utils.data.DataLoader(validation_dataset, batch_size=curr_batch_size, shuffle=False)

        if curr_batch_size>len(self.subdataset.curr_tf_train_x):
            print('Error in training, too big of a batch size')
            print('batch='+str(curr_batch_size))
            print('dataset=' + str(len(self.subdataset.curr_tf_train_x)))
            exit(0)

        validation_list=[]
        validation_has_converged=False
        validation_accuracy_reached=-999

        for param_group in self.optimizer.param_groups:
            print('--Training phase, current lr='+str(param_group['lr']))

        for epoch in range(1, gparams._EPOCHS):
            selected_model.train()
            if validation_has_converged:
                break
            batch_num, batch_train_loss, batch_val_loss = 0, 0., 0.
            # stateful requires state reset typically every epoch (stateless=for each sample)
            if gparams._STATEFUL:
                selected_model.reset_states()
            for batch, (x, y) in enumerate(train_loader):
                # print(x.size())
                # print('Entering batch='+str(batch)+'/'+str(len(x)/gparams._BATCH_SIZE))
                # neglect last batch if statefull (need all batches to be equal)
                if gparams._STATEFUL and x.shape[0] != curr_batch_size:
                    # print('Case stateful, avoiding last batch...')
                    pass
                else:
                    outputs = selected_model(x)
                    self.optimizer.zero_grad()
                    # print('batch output=' + str(outputs.data.numpy()))
                    if gparams._LOSS_FUNC == 'DILATE':
                        self.loss, loss_shape, loss_temporal = 0,0,0 #dilate_loss(y, outputs, alpha=0.5, gamma=0.01, device='cpu') todo
                    else:
                        self.loss = self.criterion(outputs, y)  # critirion mean-squared error for regression
                    self.loss.backward()
                    selected_model.float()  # add this here
                    self.optimizer.step()
                    batch_num = batch_num + 1
                    batch_train_loss = batch_train_loss + self.loss.item()

            self.saved_loss = batch_train_loss / batch_num
            self.scheduler.step(self.saved_loss)  # here we need validation loss actually

            # validation
            selected_model.eval()
            rms, rms_baseline, enhance, nrms=self.validate_base(input_model='existing')
            # inference for debug
            #rms2, rms_baseline2, enhance2, nrms2=self.infer_base(input_model='existing')

            validation_list.append(np.mean(enhance)) # for horizon>1 i take the average of all predictions enhancements
            if len(validation_list) > gparams._EARLY_STOP_MIN_EPOCHS and (validation_list[-gparams._EARLY_STOP_LOOKTHROUGH_EPOCHS] > max(validation_list[-gparams._EARLY_STOP_LOOKTHROUGH_EPOCHS+1:])): # with nrms change to < and min
                validation_has_converged=True
            validation_accuracy_reached=enhance

        print('Client=' + str(self.id) + ' Early stop at epoch=' + str(epoch) + ', with valid acc=' + str(
            validation_accuracy_reached))
        selected_model.eval()
        torch.save(selected_model.state_dict(), os.path.join(gparams._FOLDER_ML_MODELS,saved_name))
        return selected_model, epoch

class Client(Worker):
    def __init__(self, id):
        Worker.__init__(self, id)
        self.bw_UL=0
        self.bw_DL = 0
        self.energy_UL = 0
        self.energy_DL = 0
        self.energy_PROC=0

    def download_model(self,model,curr_time):
        self.model = model
        self.reset_model_params()

        curr_throughput=self.subdataset.get_thru_by_timestamp(time=curr_time) # bps

        model_size_bytes= getsizeof(model)
        time_for_download=model_size_bytes*8/curr_throughput #sec

        self.bw_DL=self.bw_DL+model_size_bytes
        self.energy_DL=self.energy_DL+gparams._POWER_LTE_DL*time_for_download

    def get_curr_raw_train(self):
        return self.subdataset.curr_raw_train

    def get_curr_raw_valid(self):
        return self.subdataset.curr_raw_valid

    def download_value(self,data_size_bytes,curr_time):
        new_bytes=data_size_bytes

        curr_throughput=self.subdataset.get_thru_by_timestamp(time=curr_time) # bps
        time_needed=new_bytes*8/curr_throughput #sec

        self.bw_DL=self.bw_DL+new_bytes
        self.energy_DL=self.energy_DL+gparams._POWER_LTE_UL*time_needed


    def upload_curr_raw_train(self,curr_time):
        new_bytes_for_upload=getsizeof(self.subdataset.curr_raw_train)
        curr_throughput=self.subdataset.get_thru_by_timestamp(time=curr_time) # bps
        time_for_upload=new_bytes_for_upload*8/curr_throughput #sec

        self.bw_UL=self.bw_UL+new_bytes_for_upload
        self.energy_UL=self.energy_UL+gparams._POWER_LTE_UL*time_for_upload

        return self.subdataset.curr_raw_train,time_for_upload

    def upload_curr_raw_valid(self,curr_time):
        new_bytes_for_upload=getsizeof(self.subdataset.curr_raw_valid)
        curr_throughput=self.subdataset.get_thru_by_timestamp(time=curr_time) # bps
        time_for_upload=new_bytes_for_upload*8/curr_throughput #sec

        self.bw_UL=self.bw_UL+new_bytes_for_upload
        self.energy_UL=self.energy_UL+gparams._POWER_LTE_UL*time_for_upload

        return self.subdataset.curr_raw_valid,time_for_upload

    def get_curr_raw_infer(self):
        return self.subdataset.curr_raw_infer

    def has_data_available(self,min_time,max_time,min_records_per_round):
        return self.subdataset.has_data_available(min_time=min_time,max_time=max_time,
                                                  min_records_per_round=min_records_per_round)

    def split_rounds_dataset(self,min_time,max_time, train_perc, valid_perc, infer_perc):
        self.subdataset.split_rounds_dataset(min_time=min_time,
                                             max_time=max_time,
                                             train_perc=train_perc,
                                             valid_perc=valid_perc,
                                             infer_perc=infer_perc)
    def infer(self,model,plot_list=None,curr_round=0):
        # preprocess inference data
        self.subdataset.curr_tf_infer_x,self.subdataset.curr_tf_infer_y,self.subdataset.curr_infer_normalization_matrix\
            =self.subdataset.preprocess(self.subdataset.curr_raw_infer)

        if plot_list is not None and (self.id in plot_list):
            plot_id=self.id
        else:
            plot_id=-1

        self.rmse,self.baseline,self.enhance,self.nrmse,self.smape,self.mae=self.infer_base(model,plot_id,curr_round)


    def train_local_and_upload(self,curr_time):
        # preprocess new data
        self.subdataset.curr_tf_train_x, self.subdataset.curr_tf_train_y, self.subdataset.curr_train_normalization_matrix \
            = self.subdataset.preprocess(self.subdataset.curr_raw_train)
        self.subdataset.curr_tf_valid_x, self.subdataset.curr_tf_valid_y, self.subdataset.curr_valid_normalization_matrix \
            = self.subdataset.preprocess(self.subdataset.curr_raw_valid)
        # train
        print('Saving local model')
        this_rounds_model,final_epochs = self.train_base(None, saved_name='lstm_fl.pt')
        print('debug model size-=+'+str(getsizeof(this_rounds_model)))
        # calc resources training
        train_time=(len(self.subdataset.curr_raw_train)*final_epochs)/gparams._SPEED_TRAIN_UE_LSTM
        self.energy_PROC = self.energy_PROC +train_time*gparams._POWER_TRAIN_UE_LSTM
        # calc resources upload
        curr_throughput=self.subdataset.get_thru_by_timestamp(time=curr_time) # bps
        time_for_upload=getsizeof(this_rounds_model)*8/curr_throughput #sec

        self.bw_UL=self.bw_UL+getsizeof(this_rounds_model)
        self.energy_UL=self.energy_UL+gparams._POWER_LTE_UL*time_for_upload

        # return
        self.model = this_rounds_model
        return this_rounds_model

    def upload_kpi(self):
        # calc resources upload
        kpi=np.mean(self.enhance)
        self.bw_UL=self.bw_UL+gparams._KPI_SIZE
        return kpi

class Cloud(Worker):
    def __init__(self, id,subdataset):
        Worker.__init__(self, id)
        self.energy=0
        self.subdataset=subdataset
        self.subdataset.reset()
        self.curr_ddm_result='train'

        self.curr_client_kpi_list=None
        self.ddm = None
        self.flag_ddm_warn=None
        self.mean_kpi_history=None
        self.reset_drift_n_conv()

        self.adapt_previous_mean=0
        self.adapt_previous_variance = 0
        self.adapt_lr_coeff = 0
        self.adapt_beta1=gparams._ADAPT_BETA1
        self.adapt_beta2 = gparams._ADAPT_BETA2
        self.adapt_beta3 = gparams._ADAPT_BETA3

        print('Initialized cloud with model size: bytes='+str(getsizeof(self.model)))

    def reset_drift_n_conv(self):
        self.curr_client_kpi_list=[]
        try:
            self.ddm = DDM(min_num_instances=gparams._TEST_DRIFT_SAMPLES_MIN,
                       warning_level=gparams._TEST_DRIFT_WARN_LEVEL,
                       out_control_level=gparams._TEST_DRIFT_OUT_LEVEL)
        except:
            print('Warning: DDM not onboarded')
        self.flag_ddm_warn=False
        self.mean_kpi_history=[]

    def central_train(self,train_list,valid_list):
        # delete previous data
        self.subdataset.curr_raw_train=[]
        self.subdataset.curr_raw_valid=[]
        # merge new data
        for dset in train_list:
            self.subdataset.curr_raw_train.extend(dset)
        for dset in valid_list:
            self.subdataset.curr_raw_valid.extend(dset)
        # preprocess new data
        self.subdataset.curr_tf_train_x,self.subdataset.curr_tf_train_y,self.subdataset.curr_train_normalization_matrix\
            =self.subdataset.preprocess(self.subdataset.curr_raw_train)
        self.subdataset.curr_tf_valid_x, self.subdataset.curr_tf_valid_y, self.subdataset.curr_valid_normalization_matrix \
            = self.subdataset.preprocess(self.subdataset.curr_raw_valid)
        # train
        this_rounds_model,final_epochs = self.train_base(None, saved_name='lstm_cl.pt')
        train_time=(len(self.subdataset.curr_raw_train)*final_epochs)/gparams._SPEED_TRAIN_CLOUD_LSTM
        self.energy=self.energy+train_time*gparams._POWER_TRAIN_CLOUD_LSTM
        # return
        self.model = this_rounds_model
        return this_rounds_model

    def adaptive_calc(self,model,curr_round):
        agg_time_start = time.time()
        # Create the array containing newParameters
        newParameters_arr = np.array([])
        for i in model.parameters():
            for j in i:
                newParameters_arr = np.append(newParameters_arr, j.clone().cpu().data.numpy())

        # EMA on the mean
        my_mean = self.adapt_previous_mean * self.adapt_beta1 + (1-self.adapt_beta1)*newParameters_arr

        # Initialization Bias correction
        my_mean = my_mean / (1-pow(self.adapt_beta1, curr_round+1))

        # EMA on the Variance
        my_variance = self.adapt_previous_variance * self.adapt_beta2 + (1 - self.adapt_beta2)*np.mean((newParameters_arr-self.adapt_previous_mean)*(newParameters_arr-self.adapt_previous_mean))
        self.adapt_previous_mean = copy.deepcopy(my_mean)
        temp = copy.deepcopy(self.adapt_previous_variance)
        self.adapt_previous_variance = copy.deepcopy(my_variance)

        # Initialization Bias correction
        my_variance = my_variance / (1-pow(self.adapt_beta2, curr_round+1))

        if curr_round < 2:
            r = 1
        else:
            r = np.abs(my_variance / (temp/(1-pow(self.adapt_beta2, curr_round))))

        self.adapt_lr_coeff = self.adapt_lr_coeff * self.adapt_beta3 + (1 - self.adapt_beta3) * r

        coeff = self.adapt_lr_coeff / (1 - pow(self.adapt_beta3, curr_round + 1))
        coeff = min(gparams._LEARNING_RATE, (gparams._LEARNING_RATE * coeff) / (curr_round + 1))
        self.curr_ddm_result=coeff
        print('AdaptiveFL new lr='+str(coeff)+',at round='+str(curr_round))

        agg_time_end = time.time()
        self.energy = self.energy + (agg_time_end - agg_time_start) * gparams._POWER_AGGREGATION_CLOUD_LSTM
        return coeff

    def aggregate(self,model_list):
        agg_time_start = time.time()
        new_params = list()
        print('aggregation start: layers='+str(range(len(list(self.model.parameters())))))
        for param_i in range(len(list(self.model.parameters()))):
            spdz_params = list()
            for uploaded_model in model_list:
                spdz_params.append(list(uploaded_model.parameters())[param_i])
            new_param_sum = 0
            for i in spdz_params:
                new_param_sum = new_param_sum + i
            new_param = new_param_sum / len(spdz_params)
            new_params.append(new_param)
        # cleanup
        with torch.no_grad():
            # clear old central model
            for model in list(self.model.parameters()):
                for param in model:
                    param *= 0
            # update central model
            for param_index in range(len(list(self.model.parameters()))):
                list(self.model.parameters())[param_index].set_(new_params[param_index])
        this_rounds_model = self.model
        print('Saving aggregated model')
        torch.save(this_rounds_model.state_dict(), os.path.join(gparams._FOLDER_ML_MODELS, 'lstm_fl.pt'))
        agg_time_end=time.time()
        self.energy=self.energy+(agg_time_end-agg_time_start)*gparams._POWER_AGGREGATION_CLOUD_LSTM
        return this_rounds_model

    def is_dead_round(self,check_drift=False,check_converge=False):
        if len(self.curr_client_kpi_list)==0:
            print('Check dead round FALSE: insufficient client KPIs')
            return False
        elif (not check_converge) and (not check_drift):
            print('Check dead round FALSE: No criteria')
            return False
        else:
            new_kpi=np.mean(self.curr_client_kpi_list)
            self.mean_kpi_history.append(new_kpi)
            print('New kpi abs='+str(new_kpi))

        # if still training or re-training
        if self.curr_ddm_result in ['train','drift']:
            if check_converge and self.detected_convergence():
                print('Check dead round TRUE: converged')
                self.curr_ddm_result = 'conv'
                return True
            else:
                print('Check dead round FALSE: not converged')
                return False
        elif self.curr_ddm_result in ['conv']:
            if check_drift and self.detected_drift():
                print('Check dead round FALSE: drift detected')
                self.curr_ddm_result = 'drift'
                self.reset_drift_n_conv()
                return False
            else:
                print('Check dead round TRUE: converged, no drift')
                return True

    def detected_convergence(self):
        # Get mean kpi of all clients
        mean_kpi=np.mean(self.curr_client_kpi_list)
        # Add to mean_kpi list
        self.mean_kpi_history.append(mean_kpi)
        # Check if kpi history list is sufficient to perform test
        if len(self.mean_kpi_history)<=gparams._TEST_CONV_SAMPLES_MIN:
            print('Check convergence FALSE: insufficient kpi history')
            return False

        # deprecated method
        #max_element=np.max(self.mean_kpi_history)
        #found_enhancement=False
        #for i in range(len(self.mean_kpi_history)-1,len(self.mean_kpi_history)-gparams._TEST_CONV_SAMPLES_NUM-1,-1):
        #    if self.mean_kpi_history[i]>max_element:
        #        found_enhancement=True

        # Check if accuracy improved over the last _TEST_CONV_SAMPLES_NUM samples
        if self.mean_kpi_history[-gparams._TEST_CONV_SAMPLES_NUM]>max(self.mean_kpi_history[-gparams._TEST_CONV_SAMPLES_NUM+1:]):
            print('Check convergence=TRUE: not found better accuracy')
            return True
        else:
            print('Check convergence=FALSE: found better accuracy')
            return False

    def detected_drift(self):
        # Transform client kpi list into succ/fail list i.e. 0/1 (1=misclassification)
        binary_list=[]
        for kpi in self.curr_client_kpi_list:
            if kpi>=gparams._TEST_DRIFT_ENHANCE_THRESHOLD_PERC: # success!
                binary_list.append(0)
            else:
                binary_list.append(1) # misclassification

        # Add this rounds classifications one-by-one to the ddm detector (existing series of 1s and 0s)
        for el in binary_list:
            try:
                self.ddm.add_element(el)

                if self.ddm.detected_warning_zone():
                    print('DDM: Warning zone has been detected')
                    self.flag_ddm_warn=True

                if self.ddm.detected_change():
                    if self.flag_ddm_warn:
                        print('DDM: Detected change with prior warning - RED ALERT')
                        return True
                    else:
                        print('DDM: Detected change without prior warning - yellow ALERT')
                        self.flag_ddm_warn=True
                        return True
            except:
                print('Warning: DDM detection not function')

        # all clear
        self.flag_ddm_warn=False
        return False

    def get_cloud_energy(self):
        return self.energy

class Clients:
    def __init__(self,subdatasets,network):
        self.db=[]
        self.create_with_data(subdatasets=subdatasets,max_clients=math.inf,match_dataset_id=True)
        self.network=network
        self.curr_available_clients_list=[]
        self.curr_train_clients_list=[]
        self.curr_infer_clients_list=[]

    def infer(self,model,selected,plot_list=None,curr_round=0):
        for cl in self.db:
            if cl.id in self.curr_available_clients_list:
                if selected=='all' or cl.id in selected:
                    cl.infer(model=model,plot_list=plot_list,curr_round=curr_round)

    def check_data_availability(self,min_time,max_time,min_records_per_round):
        total=0
        for cl in self.db:
            if cl.has_data_available(min_time=min_time,max_time=max_time,min_records_per_round=min_records_per_round):
                total=total+1
        if gparams._VALID_CLIENT_POOL_PER_ROUND==total:
            return True
        return False

    def create_availability_list(self,min_time,max_time,min_records_per_round,valid_clients_pool):
        avail_list=[]
        for cl in self.db:
            if cl.has_data_available(min_time=min_time,max_time=max_time,min_records_per_round=min_records_per_round):
                avail_list.append(cl.id)
        self.curr_available_clients_list=avail_list
        print(str(self.curr_available_clients_list))
        print(str(min_time)+'-'+str(max_time))
        if len(self.curr_available_clients_list)>=valid_clients_pool:
            return True
        return False

    def split_rounds_dataset(self,client_list,min_time,max_time,train_perc,valid_perc,infer_perc):
        for cl in self.db:
            if cl.id in client_list:
                cl.split_rounds_dataset(min_time=min_time,
                                        max_time=max_time,
                                        train_perc=train_perc,
                                        valid_perc=valid_perc,
                                        infer_perc=infer_perc)

    def select_clients(self,num,total_list):
        if num<len(total_list):
            random_ids=random.sample(total_list,num)
        else:
            random_ids=total_list
        return random_ids

    def client_selection(self,train_clients,infer_clients,strategy,min_time,max_time):
        my_method=strategy[0]
        my_train_perc=strategy[1]
        my_valid_perc=strategy[2]
        my_infer_perc=strategy[3]

        if my_method=='homo_rand':
            # randomly select training clients from all available
            self.curr_train_clients_list=self.select_clients(num=train_clients,total_list=self.curr_available_clients_list)
            # infer clients is all available clients
            self.curr_infer_clients_list=self.select_clients(num=infer_clients,total_list=self.curr_available_clients_list)
            # all clients split datasets (homogeneous way) into train-valid-infer
            self.split_rounds_dataset(client_list=self.curr_train_clients_list, min_time=min_time, max_time=max_time,
                                      train_perc=my_train_perc, valid_perc=my_valid_perc, infer_perc=my_infer_perc)
            self.split_rounds_dataset(client_list=self.curr_infer_clients_list, min_time=min_time, max_time=max_time,
                                      train_perc=my_train_perc, valid_perc=my_valid_perc, infer_perc=my_infer_perc)
        elif my_method=='hetero_rand':
            # randomly select training clients from all available
            self.curr_train_clients_list=self.select_clients(num=train_clients,total_list=self.curr_available_clients_list)
            # the rest become "testers"
            infer_list=[x for x in self.curr_available_clients_list if x not in self.curr_train_clients_list]
            self.curr_infer_clients_list=self.select_clients(num=infer_clients,total_list=infer_list)
            # trainers split dataset in 80-20-0 manner
            self.split_rounds_dataset(client_list=self.curr_train_clients_list, min_time=min_time, max_time=max_time,
                                      train_perc=my_train_perc, valid_perc=my_valid_perc, infer_perc=0)
            # testers in 0-0-100
            self.split_rounds_dataset(client_list=self.curr_infer_clients_list, min_time=min_time, max_time=max_time,
                                      train_perc=0, valid_perc=0, infer_perc=my_infer_perc)

        print('Total selected clients for training=' + str(len(self.curr_train_clients_list))+' ,are:'+str(self.curr_train_clients_list))
        print('Total selected clients for inference=' + str(len(self.curr_infer_clients_list))+' ,are:'+str(self.curr_infer_clients_list))

    def get_client_bw_UL(self):
        total=0
        for cl in self.db:
            total=total+cl.bw_UL
        return total

    def get_client_bw_DL(self):
        total=0
        for cl in self.db:
            total=total+cl.bw_DL
        return total

    def get_client_energy_UL(self):
        total = 0
        for cl in self.db:
            total = total + cl.energy_UL
        return total

    def get_client_energy_DL(self):
        total = 0
        for cl in self.db:
            total = total + cl.energy_DL
        return total

    def get_client_energy_PROC(self):
        total = 0
        for cl in self.db:
            total = total + cl.energy_PROC
        return total

    def get_network_energy_UL(self):
        return self.network.energy_UL

    def get_network_energy_DL(self):
        return self.network.energy_DL

    def get_round_ml_stats(self):
        ml_rmse=[]
        baseline_rmse=[]
        n_rmse=[]
        saved_loss=[]
        ml_smape=[]
        ml_mae=[]

        for client in self.db:
            if client.id in self.curr_infer_clients_list:
                try:
                    ml_rmse.append(client.rmse)
                    baseline_rmse.append(client.baseline)
                    n_rmse.append(client.nrmse)
                    ml_smape.append(client.smape)
                    ml_mae.append(client.mae)
                except:
                    print('ERROR: Cannot find rmse list for client='+str(client.id))
                    exit()
            if client.id in self.curr_train_clients_list:
                try:
                    saved_loss.append(client.saved_loss)
                except:
                    print('ERROR: Cannot find saved ;pss list for client='+str(client.id))
                    exit()

        return ml_rmse,baseline_rmse,n_rmse,saved_loss,ml_smape,ml_mae

    def is_mean_acc_same_for_last_rounds(self, n_rounds, threshold_abs, num_of_clients):
        res = False

        if num_of_clients<len(self.db):
            client_list=random.sample(self.db,k=num_of_clients)
        else:
            client_list=self.db

        if len(client_list[0].rmse)<n_rounds:
            pass
        else:
            accuracies=self.get_mean_acc_per_round(client_list=client_list,horizon=0,n_rounds=n_rounds)
            diff = (max(accuracies) - min(accuracies)) / (max(accuracies))
            res = (diff < threshold_abs)
        return res

    def get_mean_acc_per_round(self, client_list,horizon,n_rounds):
        accuracies=[]
        for myround in range(0, n_rounds):
            mean_round_acc=[]
            for client in client_list:
                element=client.rmse[-myround][horizon]
                mean_round_acc.append(element)
                client.total_bw=client.total_bw+2*sys.getsizeof(element)
            accuracies.append(np.mean(mean_round_acc))
        return accuracies

    def create_with_data(self,subdatasets,max_clients=math.inf,match_dataset_id=False):
        total_clients=0
        for sub in subdatasets:
            if match_dataset_id: # each client assumes an id equal to his subdataset's id
                new_client=Client(id=sub.get_client_id())
            else:
                new_client=Client(id=total_clients)
            new_client.subdataset=sub
            self.db.append(new_client)
            total_clients=total_clients+1
            if total_clients==max_clients:
                print('Reached requested max client limit. Stop assigning subdatasets')
                break

        if total_clients<gparams._VALID_CLIENT_POOL_PER_ROUND:
            print('Error: Requested a total of clients='+str(gparams._VALID_CLIENT_POOL_PER_ROUND)+',created:'+str(total_clients))
            exit(0)

        for cl in self.db:
            print('Client '+str(cl.id)+' dataset len:'+str(len(cl.subdataset.raw_data)))

    def renew_data(self,subdataset_list,delete_old=True):
        i=0
        while i<len(self.db) and i<len(subdataset_list):
            if delete_old:
                self.db[i].subdataset=subdataset_list[i]
            else:
                self.db[i].subdataset.extend(subdataset_list[i])
            i=i+1
        print('Renewed data for clients:'+str(i)+',total clients are='+str(len(self.db)))

    def get_list_of_all_ids(self):
        mylist=[]
        for client in self.db:
            mylist.append(client.id)
        return mylist

    def download_model(self,model,curr_time):
        for client in self.db:
            if client.id in self.curr_available_clients_list:
                client.download_model(model,curr_time)
                self.network.add_energy_DL(mb=getsizeof(model)/1e6)

    def update_lr(self,lr,selected,curr_time):
        for client in self.db:
            if client.id in selected:
                client.download_value(data_size_bytes=gparams._KPI_SIZE,curr_time=curr_time)
                client.optimizer.param_groups[0]['lr'] = lr
        self.network.add_energy_DL(mb=(getsizeof(gparams._KPI_SIZE)/1e6)*len(selected))

    def upload_curr_raw_train(self,selected,curr_time):
        client_list_of_raw_data = []
        max_time=0
        for client in self.db:
            if client.id in selected:
                client_raw_data,client_time=client.upload_curr_raw_train(curr_time=curr_time)
                max_time=max(max_time,curr_time)
                client_list_of_raw_data.append(client_raw_data)

        self.network.add_energy_UL(mb=getsizeof(client_list_of_raw_data)/1e6)
        max_time = max_time + self.network.calc_time_UL(mb=getsizeof(client_list_of_raw_data) / 1e6)
        return client_list_of_raw_data,max_time

    def upload_curr_raw_valid(self,selected,curr_time):
        client_list_of_raw_data = []
        max_time=0
        for client in self.db:
            if client.id in selected:
                client_raw_data,client_time=client.upload_curr_raw_valid(curr_time=curr_time)
                max_time=max(max_time,curr_time)
                client_list_of_raw_data.append(client_raw_data)

        self.network.add_energy_UL(mb=getsizeof(client_list_of_raw_data)/1e6)
        max_time=max_time+self.network.calc_time_UL(mb=getsizeof(client_list_of_raw_data)/1e6)
        return client_list_of_raw_data,max_time

    def local_train_and_upload_models(self,selected,curr_time):
        all_models = []
        debug_client_checked=[]
        for client in self.db:
            if client.id in selected:
                print('Client='+str(client.id))
                new_model = client.train_local_and_upload(curr_time)
                all_models.append(new_model)
                self.network.add_energy_UL(mb=getsizeof(new_model) / 1e6)
                debug_client_checked.append(client.id)
        if set(debug_client_checked)==set(selected):
            pass
        else:
            print('ERROR: client selection with selected='+str(selected))
        return all_models

    def upload_stats(self,selected,curr_time):
        kpi_list=[]

        for client in self.db:
            if client.id in selected and (client.id in self.curr_available_clients_list):
                new_kpi=client.upload_kpi()
                kpi_list.append(new_kpi)
        self.network.add_energy_UL(mb=getsizeof(kpi_list)/1e6)
        return kpi_list