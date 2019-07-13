from sklearn.preprocessing import MinMaxScaler,RobustScaler
import pandas as pd
import numpy as np
import math
import  os

base_dir= os.getcwd()


def load_test_data(valid_file_name):
    test_data_frame = pd.read_csv(valid_file_name)

    testing_x = test_data_frame.get_values().T[:520].T
    testing_y = test_data_frame.get_values().T[[520, 521, 522, 523], :].T
    return testing_x, testing_y

#input data separated before
def load_data_perspective(file_name):
    data_frame = pd.read_csv(file_name)
    data_x = data_frame.get_values().T[:520].T
    data_y = data_frame.get_values().T[[520, 521, 522, 523], :].T
    return data_x, data_y

#preprocess training data
pd.set_option('display.max_columns',None)
def load_grouped_data(trainingData):
    data=pd.read_csv(trainingData)

    # data=data.replace(100,-110)
    data=data.groupby(['LONGITUDE','LATITUDE','FLOOR','BUILDINGID'],as_index=False).median()
    # data_frame=data.replace(-1000,100)
    data_x = data.get_values().T[4:524].T
    data_y = data.get_values().T[[0,1,2,3], :].T

    return data_x, data_y
# load_grouped_data(train_csv_path)
#get all data
def load_data_all(train,valid,test):
    return load_data_perspective(train),load_data_perspective(valid),load_data_perspective(test)


def load(train_file_name, valid_file_name):
    # Read the file
    if train_file_name == None or valid_file_name == None:
        print ('file name is None...')
        exit()
    train_data_frame = pd.read_csv(train_file_name)
    test_data_frame = pd.read_csv(valid_file_name)
    rest_data_frame = train_data_frame
    valid_data_trame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame)/10)
    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)
    valid_data_trame = valid_data_trame.append(sample_row)
    train_data = rest_data_frame

    training_x = train_data.get_values().T[:520].T
    training_y = train_data.get_values().T[[520, 521, 522, 523], :].T
    validation_x = valid_data_trame.get_values().T[:520].T
    validation_y = valid_data_trame.get_values().T[[520, 521, 522, 523], :].T
    testing_x = test_data_frame.get_values().T[:520].T
    testing_y = test_data_frame.get_values().T[[520, 521, 522, 523], :].T

    return training_x,training_y,validation_x,validation_y,testing_x,testing_y

def normalizeX_zero_to_one(arr):
    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if (res[i][j] == 100)|(res[i][j]==None):
                res[i][j] = 0
            elif res[i][j]<-100:
                res[i][j]=0
            else :
                res[i][j] = 0.01 * (100 + res[i][j])
    return res




class norm_X():
    def __init__(self):
        self.robust=RobustScaler()
        self.model=None
        # self.trained=False
    def fit(self,arr):
        res = np.copy(arr).astype(np.float)

        for i in range(np.shape(res)[0]):
            for j in range(np.shape(res)[1]):
                if (res[i][j] == 100) | (res[i][j] == None):
                    res[i][j] = 0
        self.model=self.robust.fit(res)
        # self.trained=True
        return self.model

    def transform(self,arr):
        if self.model==None:
            self.model=self.fit(arr)
            print(self.model)

        res = np.copy(arr).astype(np.float)

        for i in range(np.shape(res)[0]):
            for j in range(np.shape(res)[1]):
                if (res[i][j] == 100) | (res[i][j] == None):
                    res[i][j] = 0
        return self.model.transform(res)


def normalizeX_powed(arr,b):
    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if (res[i][j] >50)|(res[i][j]==None)|(res[i][j]<-95):
                res[i][j] = 0
            elif (res[i][j]>=0):
                res[i][j]=1

            else :
                res[i][j] = ((95 + res[i][j])/95.0) ** b
            # res[i][j] = (0.01 * (110 + res[i][j])) ** 2.71828
    return res


def normalizeX_powed_noise(arr,rate=0):
    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        facter=(1+rate/2-np.random.random()*rate)
        for j in range(np.shape(res)[1]):
            if (res[i][j] == 100) | (res[i][j] == None):
                res[i][j] = 0
            elif res[i][j] < -100:
                res[i][j] = 0
                # res[i][j] = -0.01 * res[i][j]
            else :
                res[i][j]=res[i][j]*facter
                res[i][j]=max(res[i][j],-99)
                res[i][j] = (0.01 * (100 + res[i][j])) ** 2.71828
    return res


normx=norm_X()

def normalizeX(arr,b=2.71828):
    # return normalizeX_powed_noise(arr,rate=noise_rate).reshape([-1,520])
    # return normalizeX_zero_to_one(arr).reshape([-1, 520])
    return normalizeX_powed(arr,b).reshape([-1, 520])

    # return normx.transform(arr).reshape([-1,520])

class NormY(object):
    long_max=None
    long_min=None
    lati_max=None
    lati_min=None
    long_scale=None
    lati_scale=None

    def __init__(self):
        pass

    def fit(self,long,lati):
        self.long_max=max(long)
        self.long_min=min(long)
        self.lati_max=max(lati)
        self.lati_min=min(lati)
        self.long_scale=self.long_max-self.long_min
        self.lati_scale=self.lati_max-self.lati_min

    def normalizeY(self,longitude_arr, latitude_arr):

        longitude_arr = np.reshape(longitude_arr, [-1, 1])
        latitude_arr = np.reshape(latitude_arr, [-1, 1])
        long=(longitude_arr-self.long_min)/self.long_scale
        lati=(latitude_arr-self.lati_min)/self.lati_scale
        return long,lati

    def reverse_normalizeY(self,longitude_arr, latitude_arr):

        longitude_arr = np.reshape(longitude_arr, [-1, 1])
        latitude_arr = np.reshape(latitude_arr, [-1, 1])
        long=(longitude_arr*self.long_scale)+self.long_min
        lati=(latitude_arr*self.lati_scale)+self.lati_min
        return long,lati

def oneHotEncode(arr):
    return pd.get_dummies(np.reshape(arr, [-1])).values

def oneHotDecode(arr):
    return np.argmax(np.round(arr), axis=1)
def oneHotDecode_list(arrs):
    # return np.argmax(np.round(arr),axis=2)
    res=[]
    for arr in arrs:
        res.append(np.argmax(np.round(arr),axis=1))
    return res