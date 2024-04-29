import gparams
from simulation import Simulation
import time

def run_experiment(sim_id):

        tic=time.time()
        ############################ param setting #################################################
        # experiment related
        gparams._REPETITION = sim_id
        # dataset related
        gparams._VALID_CLIENT_POOL_PER_ROUND = 10  # number of actual clients to be created from the total number of available subdatasets
        gparams._CLIENTS_PER_ROUND_TRAIN = 8  # clients to participate in each round from the total pool
        gparams._CLIENTS_PER_ROUND_INFER = 2
        gparams._MIN_RECORDS_PER_USER = 20  # min num of records required to constitute a valid user subdataset during MainDataset split
        gparams._MIN_RECORDS_PER_ROUND = 700  # int(1000*(1/1)) 450 min num of records for a user per round to account for train, validation and inference
        gparams._SPLIT_METHOD = ['hetero_rand',0.8,0.2,1]  # ['hetero_rand',0.8,0.2,1],['homo_rand', 0.7, 0.1, 0.2]
        gparams._PLOT_CLIENTS = _PLOT_CLIENTS # [1,2,3] or [] to keep track or client actual values per round
        gparams._NAIVE_PREDICTION = 'double'  # single means that window=sliding window and nrmse=normalized rmse, double also produces with window=5 as nrmse
        gparams._DATASET_STEP = _DATASET_STEP  # sec
        gparams._T_START = int(780*_DATASET_STEP)
        gparams._T_PREV=0
        gparams._T_END = _T_END
        gparams._SPLIT_ROUND_TYPE = 'time'  # ['data','time'], data when no leaks, time when client lean
        gparams._ROUND_DURATION = int(790*_DATASET_STEP)
        gparams._COL_PREDICTION=_COL_PREDICTION
        # ML related
        gparams._ML= _ML   # CL,SL,FL_forever
        gparams._UNI_MULTI = _UNI_MULTI # 11 for UE-based, 13 for BS-alos
        gparams._SCALER = 'standard'  # ['min_max_minus', 'min_max_zero', 'standard', 'robust', 'max_abs']
        gparams._FLAG_NORMALIZE = True  # true/false if feature normalization
        gparams._LEARNING_RATE = _LEARNING_RATE
        gparams._DROPOUT = 0.8
        gparams._SLIDING_WINDOW = _SLIDING_WINDOW  #
        gparams._HIDDEN_FEATURES = _HIDDEN_FEATURES  #
        gparams._EPOCHS = 50  # 100, 1000
        gparams._STACKED_LSTM = 1  # 50...200 number of stacked lstm layers (200,100)=0,(500,50)
        gparams._BATCH_SIZE = _BATCH_SIZE  # 164 # for no batch (all dataset atpr once), use batch=0
        gparams._BIDIRECTIONAL = False  # true/false if lstm is bidirectional
        gparams._STATEFUL = False
        gparams._OPTIMIZER = 'adam'  # ['adam', 'sgd', 'rmsprop']
        gparams._WEIGHT_DECAY = 1e-5
        gparams._ACTIVATION = 'relu'  # ['tanh', 'relu', 'none']
        gparams._ML_MODEL = 'LSTM_bibatch'  # ['LSTM_CNN', 'LSTM_seq2seq', 'ANN', 'LSTM_bibatch', 'GRU_bibatch','LSTM_bibatch_stacked','CNN_batch']
        gparams._LOSS_FUNC = 'MSE'  # ['DILATE', 'MSE', 'RMSLE', 'L1LOSS','HuberLoss']
        gparams._HORIZON = _HORIZON
        gparams._EARLY_STOP_MIN_EPOCHS = 15     # minimum number of epochs before early stop can be triggered
        gparams._EARLY_STOP_LOOKTHROUGH_EPOCHS = 5 # number of epochs to search in the past for accuracy improvement
        gparams._LOSS_INFINITE_VALUE = 99 # value to assume instead of assigning math.inf to initial loss
        gparams._ROUNDSTOP=_ROUNDSTOP
        # DDM related
        if gparams._ML=='FL_ddm_all':
                gparams._CLIENTS_MONITORED = gparams._VALID_CLIENT_POOL_PER_ROUND
        gparams._TEST_CONV_SAMPLES_MIN = 5  # minimum size of list of accuracies to accept convergence result = number of rounds without conv detection
        gparams._TEST_CONV_SAMPLES_NUM = 5  # last N samples to check if accuracy improved, else convergence reached (beta4)
        gparams._TEST_DRIFT_SAMPLES_MIN = 5*gparams._CLIENTS_MONITORED #number of rounds without drift detection x clients monitored per round
        gparams._TEST_DRIFT_WARN_LEVEL = _TEST_DRIFT_WARN_LEVEL #(beta2)
        gparams._TEST_DRIFT_OUT_LEVEL = _TEST_DRIFT_OUT_LEVEL #(beta3)
        gparams._TEST_DRIFT_ENHANCE_THRESHOLD_PERC = _TEST_DRIFT_ENHANCE_THRESHOLD_PERC  # minimum improvement over baseline to account for successful prediction (%) (beta1)
        # Resource consumption related
        gparams._MIN_THROUGHPUT = 120000  # bps (from dataset check)
        gparams._POWER_LTE_UL = float(2.5)  # watt https://xiaoshawnzhu.github.io/5g-sigcomm21.pdf
        gparams._POWER_LTE_DL = float(3.5)  # watt https://xiaoshawnzhu.github.io/5g-sigcomm21.pdf
        gparams._POWER_TRAIN_UE_LSTM = float(50)  # watt https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139681 and https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7723730
        gparams._POWER_TRAIN_CLOUD_LSTM = float(225)  # watt https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139681 (V100, lstm, 900samples/sec, 0.25jouesl/samples
        gparams._POWER_AGGREGATION_CLOUD_LSTM = float(100)  #  https://dl.acm.org/doi/pdf/10.1145/1058129.1058148 (assume half since cache bw 50%)
        gparams._SPEED_TRAIN_UE_LSTM = 25  # samples per sec  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139681
        gparams._SPEED_TRAIN_CLOUD_LSTM = 900  # samples per sec, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139681 (V100)
        gparams._SPEED_AGGREGATION_CLOUD_LSTM = 0.1  # secs per model , https://dl.acm.org/doi/pdf/10.1145/1058129.1058148
        ##########################################################################################
        # Adaptive_FL_params
        gparams._ADAPT_BETA1 = 0.7
        gparams._ADAPT_BETA2 = 0.7
        gparams._ADAPT_BETA3 = 0.7

        # only for debugging
        if gparams._ML=='Adapt_FL':
                gparams._TEST_DRIFT_ENHANCE_THRESHOLD_PERC=gparams._ADAPT_BETA1
                gparams._TEST_DRIFT_WARN_LEVEL = gparams._ADAPT_BETA2
                gparams._TEST_DRIFT_OUT_LEVEL = gparams._ADAPT_BETA3


        simulation=Simulation(id=sim_id)
        simulation.run()

        toc=time.time()
        mystr='Finised exp=' + str(sim_id) +',ml='+str(gparams._ML)+',at time='+str(toc-tic)+ '\n'+'-----------'
        with open(gparams._FILE_TEMP_LOGGER, mode='a') as file:
            file.write(mystr+'\n')

for repett in range(52,110):
        print('Running repet='+str(repett))
        _T_END= 8200
        _TEST_DRIFT_WARN_LEVEL = float(2) #(beta2)
        _TEST_DRIFT_OUT_LEVEL = float(3) #(beta3)
        _TEST_DRIFT_ENHANCE_THRESHOLD_PERC = 0  # minimum improvement over baseline to account for successful prediction (%) (beta1)
        _PLOT_CLIENTS=[1,4,7]
        #for _ML in ['FL_oneshot_with_conv_stop','FL_forever','FL_freq_3rounds','FL_ddm_all','Adapt_FL']:
        for _ML in ['FL_forever','CL']:
                _ROUNDSTOP=1
                _DATASET_STEP=1
                _UNI_MULTI=22 #13 for MNO-based
                _COL_PREDICTION=0 # if uni=11, col=0 is thru and col=9 is snr
                _SLIDING_WINDOW= 75   #int(75*(1/_DATASET_STEP))
                _HIDDEN_FEATURES=50 #int(50*(1/_DATASET_STEP))
                _HORIZON=int(8/_DATASET_STEP)
                if _ML == 'CL':
                        _BATCH_SIZE=128
                        _LEARNING_RATE=1e-4
                else:
                        _BATCH_SIZE = 64
                        _LEARNING_RATE = 1e-5

                run_experiment(sim_id=repett)

