import csv
import math
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import os
import numpy as np
import gparams

def sliding_windows_uni_no_horizon(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)
def sliding_windows_multi_no_horizon(data, seq_length, y_column):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length, y_column]
        x.append(_x)
        y.append(_y)

    y = np.array(y)
    y = y.reshape(-1, 1)
    return np.array(x), y
def sliding_windows(data, window, y_column, horizon):
    x = []
    y = []

    for i in range(len(data) - window - horizon + 1):
        _x = data[i:(i + window)]
        _y = data[i + window:i + window + horizon, y_column]
        x.append(_x)
        y.append(_y)

    y = np.array(y)
    y = y.reshape(-1, horizon)
    return np.array(x), y
def build_normalization_matrix(np_set):

    train_normalization_matrix = []
    if gparams._FLAG_NORMALIZE:
        ll = []
        for i in range(0, gparams._UNI_MULTI):
            if gparams._SCALER=='min_max_minus':
                alpha = MinMaxScaler(feature_range=(-1, 1))
            elif gparams._SCALER=='min_max_zero':
                alpha = MinMaxScaler()
            elif gparams._SCALER=='standard':
                alpha = StandardScaler()
            elif gparams._SCALER=='robust':
                alpha = RobustScaler()
            elif gparams._SCALER=='max_abs':
                alpha = MaxAbsScaler()
            else:
                print('Scaler not defined!')
                exit(0)
            norm_col = alpha.fit_transform(np_set[:, i:i + 1])
            train_normalization_matrix.append(alpha)
            ll.append(norm_col)
        norm_np_set = np.concatenate(tuple(ll), axis=1)
    else:
        norm_np_set=np_set

    np_x, np_y = sliding_windows(norm_np_set, gparams._SLIDING_WINDOW,
                                 y_column=gparams._COL_PREDICTION, horizon=gparams._HORIZON)

    tensor_x = Variable(torch.Tensor(np_x))
    tensor_y = Variable(torch.Tensor(np_y))
    return tensor_x, tensor_y, train_normalization_matrix

class Dataset:
    def __init__(self):
        self.subdatasets=[]

    def get_subdatasets(self):
        print('Retrieving a total of subdatasets:' + str(len(self.subdatasets)))
        return self.subdatasets

class Subdataset:
    def __init__(self,id):
        self.id=id
        self.raw_data=[]
        self.client_id=None

        # Current
        self.curr_raw_train=[]
        self.curr_tf_train_x = None
        self.curr_tf_train_y = None
        self.curr_train_normalization_matrix = []

        self.curr_raw_valid = []
        self.curr_tf_valid_x=None
        self.curr_tf_valid_y=None
        self.curr_valid_normalization_matrix=[]

        self.curr_raw_infer = []
        self.curr_tf_infer_x=None
        self.curr_tf_infer_y=None
        self.curr_infer_normalization_matrix=[]

    def get_client_id(self):
        return self.client_id

    def reset(self):
        #todo: and maybe add curr_train, curr_valid, curr_infer (deleting from split_rounds_dataset)
        self.curr_tf_train_x = None
        self.curr_tf_train_y = None
        self.curr_train_normalization_matrix = []
        self.curr_tf_test_x=None
        self.curr_tf_test_y=None
        self.curr_test_normalization_matrix=[]

    def get_thru_by_timestamp(self,time):
        for rec in self.raw_data:
            if math.isclose(time, rec.time, rel_tol=0.1):
                if rec.throughput!=0:
                    return rec.throughput
                else:
                    return gparams._MIN_THROUGHPUT
        print('Error cannot find timestamp='+str(time))
        exit(0)

class Record_OMNET:
    def __init__(self):
        self.id=-1
        self.client_id=None
        self.is_used=False

        self.time=None
        self.averageCqiDl = None
        self.averageCqiUl = None
        self.endToEndDelay = None
        self.harqErrorRate_1st_Dl = None
        self.harqErrorRate_1st_Ul = None
        self.harqErrorRate_2nd_Dl = None
        self.harqErrorRate_2nd_Ul = None
        self.harqErrorRate_3rd_Dl = None
        self.harqErrorRate_3rd_Ul = None
        self.harqErrorRate_4th_Dl = None
        self.harqErrorRateDl = None
        self.harqErrorRateUl = None
        self.macDelayUl = None
        self.mob_x = None
        self.mob_y = None
        self.mob_z = None
        self.mob_vel_x = None
        self.mob_vel_y=None
        self.mob_vel_z=None
        self.packetReceived=None
        self.packetSent=None
        self.passedUpPk=None
        self.rcvdSinr=None
        self.receivedPacketFromLowerLayer=None
        self.receivedPacketFromUpperLayer=None
        self.sentPacketToLowerLayer=None
        self.sentPacketToUpperLayer=None
        self.servingCell=None
        self.throughput=None


    def load(self,time,averageCqiDl,endToEndDelay,measuredSinrDl,
                mob_x,mob_y,mob_z,mob_vel_x,mob_vel_y,mob_vel_z,
             rcvdSinrDl,rlcThroughputDl,servingCell,throughput,bs_clients,bs_throughput,client_id):
        try:
            self.time=float(time)
            self.averageCqiDl=float(averageCqiDl)
            self.averageCqiUl=float(-1)
            self.endToEndDelay=float(endToEndDelay)
            self.harqErrorRate_1st_Dl=float(-1)
            self.harqErrorRate_1st_Ul=float(-1)
            self.harqErrorRateDl=float(-1)
            self.harqErrorRateUl=float(-1)
            self.harqTxAttemptsDl=float(-1) #have data- not for pred
            self.harqTxAttemptsUl=float(-1)
            self.macDelayUl=float(-1)
            self.measuredSinrDl=float(measuredSinrDl)
            self.measuredSinrUl=float(-1)
            self.mob_vel_x=float(mob_vel_x)
            self.mob_vel_y=float(mob_vel_y)
            self.mob_vel_z=float(mob_vel_z)
            self.mob_x=float(mob_x)
            self.mob_y=float(mob_y)
            self.mob_z=float(mob_z)
            self.passedUpPk=float(-1) #have data- not for pred
            self.rcvdSinrDl=float(rcvdSinrDl)
            self.rcvdSinrUl=float(-1)
            self.rlcDelayDl=float(-1)
            self.rlcDelayUl=float(-1)
            self.rlcPacketLossDl=float(-1) #have data- not for pred
            self.rlcPacketLossTotal=float(-1) #have data- not for pred
            self.rlcPacketLossUl=float(-1)
            self.rlcPduDelayDl=float(-1)
            self.rlcPduDelayUl=float(-1)
            self.rlcPduPacketLossDl=float(-1) #have data- not for pred
            self.rlcPduPacketLossUl=float(-1)
            self.rlcPduThroughputDl=float(-1)
            self.rlcPduThroughputUl=float(-1)
            self.rlcThroughputDl=float(rlcThroughputDl)
            self.rlcThroughputUl=float(-1)
            self.servingCell=float(servingCell)
            self.throughput=float(throughput)
            self.bs_clients = float(bs_clients)
            self.bs_throughput = float(bs_throughput)
            self.client_id=float(client_id)
            return True
        except:
            return False
class Dataset_OMNET(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.db=[]
        self.load_db()
        print('Created DB records='+str(len(self.db)))
        self.create_subdatasets()

    def get_schema(self):
        schema=Sub_OMNET(id=0)
        return schema

    def load_db(self):
        cnt=0
        rec_id = 0
        database_location=os.path.join(gparams._DATASET_LOC)
        with open(database_location) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                new_record = Record_OMNET()
                #is_loaded=new_record.load(row['time'], row['averageCqiDl'], row['averageCqiUl'],row['endToEndDelay'],
                #                         row['harqErrorRate_1st_Dl'], row['harqErrorRate_1st_Ul'],row['harqErrorRateDl'],
                #                          row['harqErrorRateUl'],row['harqTxAttemptsDl'], row['harqTxAttemptsUl'],
                #                        row['macDelayUl'],row['measuredSinrDl'],row['measuredSinrUl'],
                #                          row['mob_x'],row['mob_y'],row['mob_z'],
                #                          row['mob_vel_x'], row['mob_vel_y'],row['mob_vel_z'],
                #                         row['passedUpPk'],row['rcvdSinrDl'],row['rcvdSinrUl'],
                #                          row['rlcDelayDl'],row['rlcDelayUl'],row['rlcPacketLossDl'],row['rlcPacketLossTotal'],
                #                          row['rlcPacketLossUl'],row['rlcPduDelayDl'],row['rlcPduDelayUl'],row['rlcPduPacketLossDl'],
                #                          row['rlcPduPacketLossUl'],row['rlcPduThroughputDl'],row['rlcPduThroughputUl'],row['rlcThroughputDl'],
                #                          row['rlcThroughputUl'],row['servingCell'],row['throughput'],row['client_id'])

                is_loaded=new_record.load(time=row['time'],
                                          averageCqiDl=row['averageCqiDl'],
                                          endToEndDelay=row['endToEndDelay'],
                                          measuredSinrDl=row['measuredSinrDl'],
                                          mob_x=row['mob_x'],
                                          mob_y=row['mob_y'],
                                          mob_z=row['mob_z'],
                                          mob_vel_x=row['mob_vel_x'],
                                          mob_vel_y=row['mob_vel_y'],
                                          mob_vel_z=row['mob_vel_z'],
                                          rcvdSinrDl=row['rcvdSinrDl'],
                                          rlcThroughputDl=row['rlcThroughputDl'],
                                          servingCell=row['servingCell'],
                                          throughput=row['throughput'],
                                          bs_clients=row['bs_clients'],
                                          bs_throughput=row['bs_throughput'],
                                          client_id=row['client_id'])
                if is_loaded:
                    new_record.id=rec_id
                    rec_id=rec_id+1
                    self.db.append(new_record)
                else:
                    pass
                    #print('Failed to load line='+str(cnt))
                cnt=cnt+1

    def create_subdatasets(self):
        print('Creating subdatasets from traindata, a total of datapoints=' + str(len(self.db)))

        old_file = 'mpla_mpla_mpla'
        curr_sub = None
        id=0

        for rec in self.db:
            if rec.client_id != old_file:
                # Found new subdataset
                if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
                    # Save old
                    self.subdatasets.append(curr_sub)
                curr_sub = Sub_OMNET(id=id)
                curr_sub.client_id=rec.client_id
                id = id + 1
            if curr_sub is not None:
                curr_sub.raw_data.append(rec)
            old_file = rec.client_id
        # Save last client
        if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
            self.subdatasets.append(curr_sub)
        print('Created a total of subdatasets=' + str(len(self.subdatasets)))
class Sub_OMNET(Subdataset):
    def __init__(self,id):
        Subdataset.__init__(self,id)

    def has_data_available(self,min_time,max_time,min_records_per_round):
        total_records=0
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                total_records=total_records+1
        return (total_records>=min_records_per_round)

    def split_rounds_dataset(self,min_time,max_time, train_perc, valid_perc, infer_perc):
        curr_raw_total=[]
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                row = []
                row.append(rec.throughput)
                if gparams._UNI_MULTI == 19:
                    row.append(rec.averageCqiDl)
                    row.append(rec.endToEndDelay)
                    row.append(rec.harqErrorRate_1st_Dl)
                    row.append(rec.harqErrorRateDl)
                    row.append(rec.measuredSinrDl)
                    row.append(rec.measuredSinrUl)
                    row.append(rec.mob_vel_x)
                    row.append(rec.mob_vel_y)
                    row.append(rec.mob_vel_z)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.mob_z)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.rlcDelayDl)
                    row.append(rec.rlcPduDelayDl)
                    row.append(rec.rlcPduThroughputDl)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                elif gparams._UNI_MULTI == 8:
                    row.append(rec.averageCqiDl)
                    row.append(rec.measuredSinrDl)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                elif gparams._UNI_MULTI == 3:
                    row.append(rec.measuredSinrDl)
                    row.append(rec.mob_x)
                elif gparams._UNI_MULTI == 9:
                    row.append(rec.endToEndDelay)
                    row.append(rec.measuredSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.rlcDelayDl)
                    row.append(rec.rlcPduDelayDl)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                elif gparams._UNI_MULTI == 7:
                    row.append(rec.averageCqiDl)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                elif gparams._UNI_MULTI == 10:
                    row.append(rec.averageCqiDl)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                    row.append(rec.mob_vel_x)
                    row.append(rec.mob_vel_y)
                    row.append(rec.measuredSinrDl)
                elif gparams._UNI_MULTI == 11:
                    row.append(rec.averageCqiDl)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                    row.append(rec.mob_vel_x)
                    row.append(rec.mob_vel_y)
                    row.append(rec.measuredSinrDl)
                    row.append(rec.endToEndDelay)
                elif gparams._UNI_MULTI == 13:
                    row.append(rec.averageCqiDl)
                    row.append(rec.rcvdSinrDl)
                    row.append(rec.mob_x)
                    row.append(rec.mob_y)
                    row.append(rec.rlcThroughputDl)
                    row.append(rec.servingCell)
                    row.append(rec.mob_vel_x)
                    row.append(rec.mob_vel_y)
                    row.append(rec.measuredSinrDl)
                    row.append(rec.endToEndDelay)
                    row.append(rec.bs_clients)
                    row.append(rec.bs_throughput)
                elif gparams._UNI_MULTI == 1:
                    pass

                curr_raw_total.append(row)

        num_train=int(len(curr_raw_total)*train_perc)
        num_valid = int(len(curr_raw_total) *valid_perc)

        self.curr_raw_train=curr_raw_total[:num_train]
        self.curr_raw_valid =curr_raw_total[num_train:num_train+num_valid]
        self.curr_raw_infer = curr_raw_total[num_train+num_valid:]

        # sanity check
        if train_perc==0:
            dbg_train_ok=(len(self.curr_raw_train)/len(curr_raw_total)<0.1)
        else:
            dbg_train_ok=math.isclose(len(self.curr_raw_train),train_perc * len(curr_raw_total),rel_tol=0.1)
        if valid_perc==0:
            dbg_valid_ok = (len(self.curr_raw_valid) / len(curr_raw_total) < 0.1)
        else:
            dbg_valid_ok=math.isclose(len(self.curr_raw_valid),valid_perc * len(curr_raw_total),rel_tol=0.1)
        if infer_perc==0:
            dbg_infer_ok = (len(self.curr_raw_infer) / len(curr_raw_total) < 0.1)
        else:
            dbg_infer_ok=math.isclose(len(self.curr_raw_infer),infer_perc * len(curr_raw_total),rel_tol=0.1)
        dbg_split_ok=dbg_train_ok and dbg_valid_ok and dbg_infer_ok

        if not dbg_split_ok:
            print('ERROR in splitting: train-valid-infer checks='+str(dbg_train_ok)+'-'+str(dbg_valid_ok)+'-'+str(dbg_infer_ok))
            print('total,train,valid,infer=')
            print('total dataset len'+str(len(curr_raw_total)))
            print('train len='+str(len(self.curr_raw_train))+',perc_total='+str(len(self.curr_raw_train)/len(curr_raw_total)))
            print(str(len(self.curr_raw_valid))+',perc_total='+str(len(self.curr_raw_valid)/len(curr_raw_total)))
            print(str(len(self.curr_raw_infer))+',perc_total='+str(len(self.curr_raw_infer)/len(curr_raw_total)))
            exit(0)

    def preprocess(self,raw_data):
        np_set = np.array(raw_data)
        np_set = np_set.reshape((-1, gparams._UNI_MULTI))
        x,y,matrix=build_normalization_matrix(np_set)
        return x,y,matrix

class Record_BERLIN:
    def __init__(self):
        self.id=-1
        self.client_id=None
        self.is_used=False

        self.time=None
        self.Latitude = None
        self.Longitude = None
        self.Altitude = None
        self.speed_kmh = None
        self.PCell_SNR_1 = None
        self.PCell_Cell_Identity = None
        self.PCell_RSSI_1 = None
        self.PCell_RSRQ_1 = None
        self.jitter = None
        self.ping_ms = None
        self.throughput=None

    def load(self,time,Latitude,Longitude,Altitude,speed_kmh,PCell_SNR_1,PCell_Cell_Identity,
             PCell_RSSI_1,PCell_RSRQ_1,jitter,ping_ms,throughput,client_id):
        try:
            self.time = float(time)
            self.Latitude = float(Latitude)
            self.Longitude = float(Longitude)
            self.Altitude = float(Altitude)
            self.speed_kmh = float(speed_kmh)
            self.PCell_SNR_1 = float(PCell_SNR_1)
            self.PCell_Cell_Identity = float(PCell_Cell_Identity)
            self.PCell_RSSI_1 = float(PCell_RSSI_1)
            self.PCell_RSRQ_1 = float(PCell_RSRQ_1)
            self.jitter = float(jitter)
            self.ping_ms = float(ping_ms)
            self.throughput = float(throughput)
            self.client_id=float(client_id)
            return True
        except:
            return False
class Dataset_BERLIN(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.db=[]
        self.load_db()
        print('Created DB records='+str(len(self.db)))
        self.create_subdatasets()

    def get_schema(self):
        schema=Sub_BERLIN(id=0)
        return schema

    def load_db(self):
        cnt=0
        rec_id = 0
        database_location=os.path.join(gparams._DATASET_LOC)
        with open(database_location) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                new_record = Record_BERLIN()

                is_loaded=new_record.load(time=row['time'],
                                          Latitude=row['Latitude'],
                                          Longitude=row['Longitude'],
                                          Altitude=row['Altitude'],
                                          speed_kmh=row['speed_kmh'],
                                          PCell_SNR_1=row['PCell_SNR_1'],
                                          PCell_Cell_Identity=row['PCell_Cell_Identity'],
                                          PCell_RSSI_1=row['PCell_RSSI_1'],
                                          PCell_RSRQ_1=row['PCell_RSRQ_1'],
                                          jitter=row['jitter'],
                                          ping_ms=row['ping_ms'],
                                          throughput=row['throughput'],
                                          client_id=row['client_id']
                                          )
                if is_loaded:
                    new_record.id=rec_id
                    rec_id=rec_id+1
                    self.db.append(new_record)
                else:
                    pass
                    #print('Failed to load line='+str(cnt))
                cnt=cnt+1

    def create_subdatasets(self):
        print('Creating subdatasets from traindata, a total of datapoints=' + str(len(self.db)))

        old_file = 'mpla_mpla_mpla'
        curr_sub = None
        id=0

        for rec in self.db:
            if rec.client_id != old_file:
                # Found new subdataset
                if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
                    # Save old
                    self.subdatasets.append(curr_sub)
                curr_sub = Sub_BERLIN(id=id)
                curr_sub.client_id=rec.client_id
                id = id + 1
            if curr_sub is not None:
                curr_sub.raw_data.append(rec)
            old_file = rec.client_id
        # Save last client
        if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
            self.subdatasets.append(curr_sub)
        print('Created a total of subdatasets=' + str(len(self.subdatasets)))
class Sub_BERLIN(Subdataset):
    def __init__(self,id):
        Subdataset.__init__(self,id)

    def has_data_available(self,min_time,max_time,min_records_per_round):
        total_records=0
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                total_records=total_records+1
        return (total_records>=min_records_per_round)

    def split_rounds_dataset(self,min_time,max_time, train_perc, valid_perc, infer_perc):
        curr_raw_total=[]
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                row = []
                row.append(rec.throughput)

                if gparams._UNI_MULTI == 11:
                    row.append(rec.Latitude)
                    row.append(rec.Longitude)
                    row.append(rec.Altitude)
                    row.append(rec.speed_kmh)
                    row.append(rec.PCell_SNR_1)
                    row.append(rec.PCell_Cell_Identity)
                    row.append(rec.PCell_RSSI_1)
                    row.append(rec.PCell_RSRQ_1)
                    row.append(rec.jitter)
                    row.append(rec.ping_ms)
                elif gparams._UNI_MULTI == 1:
                    pass

                curr_raw_total.append(row)

        num_train=int(len(curr_raw_total)*train_perc)
        num_valid = int(len(curr_raw_total) *valid_perc)

        self.curr_raw_train=curr_raw_total[:num_train]
        self.curr_raw_valid =curr_raw_total[num_train:num_train+num_valid]
        self.curr_raw_infer = curr_raw_total[num_train+num_valid:]

        # sanity check
        if train_perc==0:
            dbg_train_ok=(len(self.curr_raw_train)/len(curr_raw_total)<0.1)
        else:
            dbg_train_ok=math.isclose(len(self.curr_raw_train),train_perc * len(curr_raw_total),rel_tol=0.1)
        if valid_perc==0:
            dbg_valid_ok = (len(self.curr_raw_valid) / len(curr_raw_total) < 0.1)
        else:
            dbg_valid_ok=math.isclose(len(self.curr_raw_valid),valid_perc * len(curr_raw_total),rel_tol=0.1)
        if infer_perc==0:
            dbg_infer_ok = (len(self.curr_raw_infer) / len(curr_raw_total) < 0.1)
        else:
            dbg_infer_ok=math.isclose(len(self.curr_raw_infer),infer_perc * len(curr_raw_total),rel_tol=0.1)
        dbg_split_ok=dbg_train_ok and dbg_valid_ok and dbg_infer_ok

        if not dbg_split_ok:
            print('ERROR in splitting: train-valid-infer checks='+str(dbg_train_ok)+'-'+str(dbg_valid_ok)+'-'+str(dbg_infer_ok))
            print('total,train,valid,infer=')
            print('total dataset len'+str(len(curr_raw_total)))
            print('train len='+str(len(self.curr_raw_train))+',perc_total='+str(len(self.curr_raw_train)/len(curr_raw_total)))
            print(str(len(self.curr_raw_valid))+',perc_total='+str(len(self.curr_raw_valid)/len(curr_raw_total)))
            print(str(len(self.curr_raw_infer))+',perc_total='+str(len(self.curr_raw_infer)/len(curr_raw_total)))
            exit(0)

    def preprocess(self,raw_data):
        np_set = np.array(raw_data)
        np_set = np_set.reshape((-1, gparams._UNI_MULTI))
        x,y,matrix=build_normalization_matrix(np_set)
        return x,y,matrix

class Record_HIDRIVE:
    def __init__(self):
        self.id=-1
        self.client_id=None
        self.is_used=False

        self.time=None
        self.latitude=None
        self.longitude=None
        self.elevation=None
        self.velocity_abs=None
        self.acceleration_abs=None
        self.rsrq=None
        self.rssi=None
        self.rsrp=None
        self.sinr=None
        self.band=None
        self.ran=None
        self.serving_cell_id=None
        self.velocity_long=None
        self.acceleration_lat=None
        self.acceleration_long=None
        self.delay=None
        self.heading=None
        self.GNSS_mode=None
        self.Service_status=None
        self.Operator=None
        self.ifstat_out=None
        self.throughput=None

    def load(self,time,latitude, longitude, elevation, velocity_abs, acceleration_abs, rsrq, rssi, rsrp, sinr,
        band, ran, serving_cell_id, velocity_long, acceleration_lat, acceleration_long, delay,
        heading, GNSS_mode, Service_status, Operator, ifstat_out,throughput,client_id):

        try:
            self.time = float(time)
            self.latitude = float(latitude)
            self.longitude = float(longitude)
            self.elevation = float(elevation)
            self.velocity_abs = float(velocity_abs)
            self.acceleration_abs = float(acceleration_abs)
            self.rsrq = float(rsrq)
            self.rssi = float(rssi)
            self.rsrp = float(rsrp)
            self.sinr = float(sinr)
            self.band = float(band)
            self.ran = float(ran)
            self.serving_cell_id = float(serving_cell_id)
            self.velocity_long = float(velocity_long)
            self.acceleration_lat = float(acceleration_lat)
            self.acceleration_long = float(acceleration_long)
            self.delay = float(delay)
            self.heading = float(heading)
            self.GNSS_mode = float(GNSS_mode)
            self.Service_status = float(Service_status)
            self.Operator = float(Operator)
            self.ifstat_out = float(ifstat_out)
            self.throughput = float(throughput)
            self.client_id=float(client_id)
            return True
        except:
            return False
class Dataset_HIDRIVE(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.db=[]
        self.load_db()
        print('Created DB records='+str(len(self.db)))
        self.create_subdatasets()

    def get_schema(self):
        schema=Sub_HIDRIVE(id=0)
        return schema

    def load_db(self):
        cnt=0
        rec_id = 0
        database_location=os.path.join(gparams._DATASET_LOC)
        with open(database_location) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                new_record = Record_HIDRIVE()

                is_loaded=new_record.load(time=row['time'],
                                          latitude=row['latitude'],
                                          longitude=row['longitude'],
                                          elevation=row['elevation'],
                                          velocity_abs=row['velocity_abs'],
                                          acceleration_abs=row['acceleration_abs'],
                                          rsrq=row['rsrq'],
                                          rssi=row['rssi'],
                                          rsrp=row['rsrp'],
                                          sinr=row['sinr'],
                                          band=row['band'],
                                          ran=row['ran'],
                                          serving_cell_id=row['serving_cell_id'],
                                          velocity_long=row['velocity_long'],
                                          acceleration_lat=row['acceleration_lat'],
                                          acceleration_long=row['acceleration_long'],
                                          delay=row['delay'],
                                          heading=row['heading'],
                                          GNSS_mode=row['gnss_mode'],
                                          Service_status=row['service_status'],
                                          Operator=row['operator'],
                                          ifstat_out=row['throughput_uplink'],
                                          throughput=row['throughput'],
                                          client_id=row['client_id']
                                          )

                if is_loaded:
                    new_record.id=rec_id
                    rec_id=rec_id+1
                    self.db.append(new_record)
                else:
                    pass
                    #print('Failed to load line='+str(cnt))
                cnt=cnt+1

    def create_subdatasets(self):
        print('Creating subdatasets from traindata, a total of datapoints=' + str(len(self.db)))

        old_file = 'mpla_mpla_mpla'
        curr_sub = None
        id=0

        for rec in self.db:
            if rec.client_id != old_file:
                # Found new subdataset
                if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
                    # Save old
                    self.subdatasets.append(curr_sub)
                curr_sub = Sub_HIDRIVE(id=id)
                curr_sub.client_id=rec.client_id
                id = id + 1
            if curr_sub is not None:
                curr_sub.raw_data.append(rec)
            old_file = rec.client_id
        # Save last client
        if curr_sub is not None and len(curr_sub.raw_data) > gparams._MIN_RECORDS_PER_USER:
            self.subdatasets.append(curr_sub)
        print('Created a total of subdatasets=' + str(len(self.subdatasets)))
class Sub_HIDRIVE(Subdataset):
    def __init__(self,id):
        Subdataset.__init__(self,id)

    def has_data_available(self,min_time,max_time,min_records_per_round):
        total_records=0
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                total_records=total_records+1
        return (total_records>=min_records_per_round)

    def split_rounds_dataset(self,min_time,max_time, train_perc, valid_perc, infer_perc):
        curr_raw_total=[]
        for rec in self.raw_data:
            if (min_time<=rec.time and rec.time<max_time):
                row = []
                row.append(rec.throughput)

                if gparams._UNI_MULTI == 22:
                    row.append(rec.latitude)
                    row.append(rec.longitude)
                    row.append(rec.elevation)
                    row.append(rec.velocity_abs)
                    row.append(rec.acceleration_abs)
                    row.append(rec.rsrq)
                    row.append(rec.rssi)
                    row.append(rec.rsrp)
                    row.append(rec.sinr)
                    row.append(rec.band)
                    row.append(rec.ran)
                    row.append(rec.serving_cell_id)
                    row.append(rec.velocity_long)
                    row.append(rec.acceleration_lat)
                    row.append(rec.acceleration_long)
                    row.append(rec.delay)
                    row.append(rec.heading)
                    row.append(rec.GNSS_mode)
                    row.append(rec.Service_status)
                    row.append(rec.Operator)
                    row.append(rec.ifstat_out)
                elif gparams._UNI_MULTI == 18:
                    row.append(rec.latitude)
                    row.append(rec.longitude)
                    row.append(rec.elevation)
                    row.append(rec.velocity_abs)
                    row.append(rec.acceleration_abs)
                    row.append(rec.band)
                    row.append(rec.ran)
                    row.append(rec.serving_cell_id)
                    row.append(rec.velocity_long)
                    row.append(rec.acceleration_lat)
                    row.append(rec.acceleration_long)
                    row.append(rec.delay)
                    row.append(rec.heading)
                    row.append(rec.GNSS_mode)
                    row.append(rec.Service_status)
                    row.append(rec.Operator)
                    row.append(rec.ifstat_out)


                curr_raw_total.append(row)

        num_train=int(len(curr_raw_total)*train_perc)
        num_valid = int(len(curr_raw_total) *valid_perc)

        self.curr_raw_train=curr_raw_total[:num_train]
        self.curr_raw_valid =curr_raw_total[num_train:num_train+num_valid]
        self.curr_raw_infer = curr_raw_total[num_train+num_valid:]

        # sanity check
        if train_perc==0:
            dbg_train_ok=(len(self.curr_raw_train)/len(curr_raw_total)<0.1)
        else:
            dbg_train_ok=math.isclose(len(self.curr_raw_train),train_perc * len(curr_raw_total),rel_tol=0.1)
        if valid_perc==0:
            dbg_valid_ok = (len(self.curr_raw_valid) / len(curr_raw_total) < 0.1)
        else:
            dbg_valid_ok=math.isclose(len(self.curr_raw_valid),valid_perc * len(curr_raw_total),rel_tol=0.1)
        if infer_perc==0:
            dbg_infer_ok = (len(self.curr_raw_infer) / len(curr_raw_total) < 0.1)
        else:
            dbg_infer_ok=math.isclose(len(self.curr_raw_infer),infer_perc * len(curr_raw_total),rel_tol=0.1)
        dbg_split_ok=dbg_train_ok and dbg_valid_ok and dbg_infer_ok

        if not dbg_split_ok:
            print('ERROR in splitting: train-valid-infer checks='+str(dbg_train_ok)+'-'+str(dbg_valid_ok)+'-'+str(dbg_infer_ok))
            print('total,train,valid,infer=')
            print('total dataset len'+str(len(curr_raw_total)))
            print('train len='+str(len(self.curr_raw_train))+',perc_total='+str(len(self.curr_raw_train)/len(curr_raw_total)))
            print(str(len(self.curr_raw_valid))+',perc_total='+str(len(self.curr_raw_valid)/len(curr_raw_total)))
            print(str(len(self.curr_raw_infer))+',perc_total='+str(len(self.curr_raw_infer)/len(curr_raw_total)))
            exit(0)

    def preprocess(self,raw_data):
        np_set = np.array(raw_data)
        np_set = np_set.reshape((-1, gparams._UNI_MULTI))
        x,y,matrix=build_normalization_matrix(np_set)
        return x,y,matrix