import os
if os.name == 'nt':
    _DIR_ROOT= 'C:\\Pycharm\\Projects\\pqos'
else:
    _DIR_ROOT='/home/iccs/git/pqos'

_CURR_DATASET_NAME='hidrive_lapland2024_processed_broken10_delay.csv' # asym_4bsod4_full_cl26_1hz, asym_4bs2_cl26_1hz

_FOLDER_ML_MODELS=os.path.join(_DIR_ROOT,'ml_models')
_FOLDER_DATASETS=os.path.join(_DIR_ROOT,'datasets')
_FOLDER_LOGS=os.path.join(_DIR_ROOT,'logs')

_FILE_TEMP_LOGGER=os.path.join(_FOLDER_LOGS,'temp_logger.txt')
_FILE_CSV_LOGGER=os.path.join(_FOLDER_LOGS,'csv_logger.txt')
_DELIMITER=';'
_FEATURES_CSV_LOGGER=['repetition',
                      'curr_dataset_name',
                      'valid_client_pool_per_round',
                      'clients_per_round_train',
                      'clients_per_round_infer',
                      'ml',
                      'learning_rate',
                      'uni_multi',
                      'sliding_window',
                      'hidden_features',
                      'batch_size',
                      'horizon',
                      'test_drift_enhance_threshold_perc',
                      'test_drift_warn_level',
                      'test_drift_out_level',
                      'test_conv_samples_num',
                      'curr_ddm_result',
                      'curr_round',
                      'client_bw_ul',
                      'client_bw_dl',
                      'client_energy_ul',
                      'client_energy_dl',
                      'client_energy_proc',
                      'cloud_energy',
                      'network_energy_ul',
                      'network_energy_dl',
                      'ml_rmse',
                      'n_rmse',
                      'baseline_rmse',
                      'saved_loss',
                      'ml_smape',
                      'ml_mae']

_DATASET_LOC=os.path.join(_FOLDER_DATASETS,_CURR_DATASET_NAME)

# experiment related
_REPETITION=0
# dataset related
_VALID_CLIENT_POOL_PER_ROUND = 25 # number of actual clients to be created from the total number of available subdatasets
_CLIENTS_PER_ROUND_TRAIN = 5 # clients to participate in each round from the total pool
_CLIENTS_PER_ROUND_INFER=20
_MIN_RECORDS_PER_USER = 20 # min num of records required to constitute a valid user subdataset during MainDataset split
_MIN_RECORDS_PER_ROUND= 500 # 450 min num of records for a user per round to account for train, validation and inference
_SPLIT_METHOD=['homo_rand',0.7,0.1,0.2] # ['hetero_rand',0.8,0.2,1]
_PLOT_CLIENTS=[1,9,19]
_NAIVE_PREDICTION='single' # single means that window=sliding window, double also produces with window=5
_DATASET_STEP=1 #sec
_T_START=0
_T_PREV = 0
_T_END=1200*3
_SPLIT_ROUND_TYPE='time' # ['data','time']
_ROUND_DURATION=1200
_COL_PREDICTION=0
_RMSE_ZERO_GROUND= 0.02 # epsilon that accounts for zero RMSE, ~1% of mean feature value //tweak to avoid division by 0
_KPI_SIZE=1 #byte
# ML related
_ML='CL' # CL, FL, SL
_UNI_MULTI = 8 #(SRFG=13, LUMOS=14, terminalUL=74)
_SCALER='min_max_minus'
_FLAG_NORMALIZE=True
_LEARNING_RATE=0.01
_DROPOUT=0.8
_SLIDING_WINDOW=5
_HIDDEN_FEATURES = 50 #number of features in hidden state
_EPOCHS=200
_STACKED_LSTM=1 # number of stacked lstm layers
_BATCH_SIZE=256
_BIDIRECTIONAL=False
_STATEFUL=False
_OPTIMIZER='adam'
_WEIGHT_DECAY=0
_ACTIVATION='relu'
_ML_MODEL='MLss'
_LOSS_FUNC='MSE'
_HORIZON=2 # horizon
_EARLY_STOP_MIN_EPOCHS=30
_EARLY_STOP_LOOKTHROUGH_EPOCHS=20
_LOSS_INFINITE_VALUE=99
_ROUNDSTOP=3
# DDM related
_CLIENTS_MONITORED=10
_TEST_CONV_SAMPLES_MIN=2 # minimum size of list of accuracies to accept convergence result
_TEST_CONV_SAMPLES_NUM=3 # last N samples to check if accuracy improved, else convergence reached
_TEST_DRIFT_SAMPLES_MIN=20
_TEST_DRIFT_WARN_LEVEL=float(2)
_TEST_DRIFT_OUT_LEVEL=float(3)
_TEST_DRIFT_ENHANCE_THRESHOLD_PERC=5  # minimum improvement over baseline to account for successful prediction (%)
# Resource consumption related
_MIN_THROUGHPUT=120000 #bps (from dataset check)
_POWER_LTE_UL=float(2.5) # watt https://xiaoshawnzhu.github.io/5g-sigcomm21.pdf
_POWER_LTE_DL=float(3.5) # watt https://xiaoshawnzhu.github.io/5g-sigcomm21.pdf
_POWER_TRAIN_UE_LSTM=float(1.8) #watt
_POWER_TRAIN_CLOUD_LSTM= float(125)  # watt for 90 perc cpu
_POWER_AGGREGATION_CLOUD_LSTM = float(15)  # watt for 10 perc cpu
_SPEED_TRAIN_UE_LSTM=300 # samples per sec
_SPEED_TRAIN_CLOUD_LSTM=40000 # samples per sec
_SPEED_AGGREGATION_CLOUD_LSTM=0.4 # secs per model
# Adaptive_FL_params
_ADAPT_BETA1 = 0.5
_ADAPT_BETA2 = 0.5
_ADAPT_BETA3 = 0.5



