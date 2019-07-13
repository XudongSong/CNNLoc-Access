import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape
from keras.constraints import max_norm
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax,SGD,RMSprop
from keras.losses import MSE,MAE,MAPE,MSLE,KLD,squared_hinge,hinge,categorical_hinge,categorical_crossentropy,sparse_categorical_crossentropy,kullback_leibler_divergence,poisson
from keras.layers.advanced_activations import PReLU
import data_helper
import numpy as np
import keras
base_dir=os.getcwd()
AE_model_dir=os.path.join(base_dir,'AE_model')
Building_model_dir=os.path.join(base_dir,'Building_model')
Floor_model_dir=os.path.join(base_dir,'Floor_model')

#rewrite class Earlystopping()

class myEarlyStopping(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None
                 ):
        super(myEarlyStopping,self).__init__(

            monitor=monitor,
            baseline=baseline,
            patience=patience,
            verbose=verbose,
            min_delta=min_delta,
            mode=mode

        )

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        elif self.best-current>0.02 or current<self.baseline:
            self.wait=0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        # if current>self.baseline:
        #     self.wait+=1
        #     if self.wait>=self.patience:
        #         self.stopped_epoch = epoch
        #         self.model.stop_training = True
        # else:
        #     self.wait=0

Train_AE=False
NO_AE=False
Train_Building=True
Run_Building=True #!!!Must be True
Train_Floor=True
Run_Floor=True  #!!!Must be True
Floor_retrain=True
Train_New_Floor=False
Train_Location=True
Run_Location=True  #!!!Must be True
class EncoderDNN(object):
    normY = data_helper.NormY()
    def __init__(self):
        self.epoch_AE=6
        self.epoch_floor=6
        self.epoch_position=6
        self.epoch_building=6
        self.loss='mse'
        self.opt='adam'
        self.lr=0.0001
        self.dropout=0.7
        self.patience=3
        self.b=2.71828
        self.input = Input((520,))
        self.AE_floor_bottleneck = '256_128'
        self.AE_building_bottleneck = '256_128'
#################AE_model
    def fnBuildAEModel(self):
        self.encode_layer = Dense(128, activation='elu', name='en1')(self.input)
        self.encode_layer = Dense(64, activation='elu', name='en2')(self.encode_layer)

        decode_layer = Dense(128, activation='elu', name='de2')(self.encode_layer)
        decode_layer = Dense(520, activation='elu', name='de-1')(decode_layer)
        self.encoder_model_256_128= Model(inputs=self.input, outputs=decode_layer)
        self.bottleneck_model_256_128= Model(inputs=self.input, outputs=self.encode_layer)

####################floor_model
    def fnBuildFloorModel(self):
        if Train_Floor or Run_Floor:
            self.floor_layers = '99-22,66-22,33-22'
            if (not Train_AE)&(not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))):
                self.floor_base_model=self.bottleneck_model_256_128
            elif NO_AE:
                self.floor_base_model=self.bottleneck_model_256_128
            else:
                self.floor_base_model =load_model(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))
            if Floor_retrain:
                for layer in self.floor_base_model.layers:
                    layer.trainable = True
                #正则化
                # layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            floor_net_input = Reshape((self.floor_base_model.output_shape[1], 1))(self.floor_base_model.output)
            floor_net_input=Dropout(self.dropout)(floor_net_input)
            floor_net = Conv1D(99, 22,activation='elu')(floor_net_input)
            floor_net = Conv1D(66, 22, activation='elu')(floor_net)
            floor_net = Conv1D(33, 22,activation='elu')(floor_net)
            floor_net = Flatten()(floor_net)
            output1 = Dense(5, activation='softmax')(floor_net)

            self.floor_model = Model(inputs=self.floor_base_model.input, outputs=output1)

####################position_model
    def fnBuildLocationModel(self):
        if Train_Location or Run_Location:
            self.floor_layers = '99-22,66-22,33-22'
            if (not Train_AE) & (
            not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))):
                self.position_base_model = self.bottleneck_model_256_128
            elif NO_AE:
                self.position_base_model = self.bottleneck_model_256_128
            else:
                self.position_base_model = load_model(
                    os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5')))

            for layer in self.position_base_model.layers:
                layer.trainable = True
                # 正则化
                # layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

            position_net_input = Reshape((self.position_base_model.output_shape[1], 1))(self.position_base_model.output)
            position_net_input = Dropout(self.dropout)(position_net_input)
            position_net = Conv1D(99, 22, activation='elu')(position_net_input)
            position_net = Conv1D(66, 22, activation='elu')(position_net)
            position_net = Conv1D(33,22, activation='elu')(position_net)
            position_net = Flatten()(position_net)
            self.position_predict_output = Dense(2, activation='elu')(position_net)
            self.position_model = Model(inputs=self.position_base_model.input, outputs=self.position_predict_output)


###################Building_model
    def fnBuildBuildingModel(self):

        if Train_Building or Run_Building:
            if os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5'))):
                self.building_base_model = load_model(os.path.join(AE_model_dir,('AE_bottleneck_' + self.AE_building_bottleneck + '.h5')))
            else:
                self.building_base_model=self.bottleneck_model_256_128
            for layer in self.building_base_model.layers:
                layer.trainable = True
            building_net = Dense(33)(self.building_base_model.output)
            self.buildingID_predict_output=Dense(3,activation='softmax')(building_net)
            self.building_model= Model(inputs=self.building_base_model.input, outputs=self.buildingID_predict_output)

    def _preprocess(self, x, y, valid_x, valid_y):
        self.normalize_x = data_helper.normalizeX(x,self.b)
        self.normalize_valid_x = data_helper.normalizeX(valid_x,self.b)

        self.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = self.normY.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = self.normY.normalizeY(valid_y[:, 0],valid_y[:, 1])
        self.floorID_valid_y = valid_y[:, 2]
        self.buildingID_valid_y = valid_y[:, 3]

    def fit(self, x, y, valid_x, valid_y,tensorbd=None):

        early_stopping = EarlyStopping(monitor='val_acc', patience=self.patience)
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        # adamm=Adam(lr=self.lr)
#############fit_AE
        if Train_AE:
            self.fnBuildAEModel()
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss=self.loss,
                optimizer=self.opt

            )
            h_AE=self.encoder_model.fit(self.normalize_x, self.normalize_x,validation_data=(self.normalize_valid_x,self.normalize_valid_x), epochs=self.epoch_AE,batch_size=66)#,callbacks=[early_stopping])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'.h5')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck+'.h5')

#################fit_Location
        if Train_Location:
            self.fnBuildLocationModel()
            self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
            self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
            self.position_model.compile(
                loss=self.loss,
                optimizer=self.opt
            )
            h_pos=self.position_model.fit(self.normalize_x, self.position,validation_data=(self.normalize_valid_x,self.position_valid),epochs=self.epoch_position, batch_size=66)#,callbacks=[early_stopping])
            self.position_model.save(os.path.join(base_dir,('Location_model/Location_model.h5')))

        self.floor_layers = '99-22,66-22,33-22'
##################fit_floor
        if Train_Floor:
            self.fnBuildFloorModel()
            # adamm=Adamax()
            # adamm=Adagrad(epsilon=1e-06)

            self.floor_layers = '99-22,66-22,33-22'
            self.floor_model.compile(
                # loss=keras.losses.mse,
                # loss=keras.losses.binary_crossentropy,
                # loss=keras.losses.categorical_crossentropy,

                loss=self.loss,
                optimizer=self.opt,
                metrics=['accuracy']
            )
            h_floor=self.floor_model.fit(self.normalize_x, data_helper.oneHotEncode(self.floorID_y), epochs=self.epoch_floor, batch_size=66,
                                 validation_data=(
                                 self.normalize_valid_x, data_helper.oneHotEncode(self.floorID_valid_y)))#,callbacks=[early_stopping])  # ,tensorbd])
            self.floor_model.save(
                os.path.join(Floor_model_dir,('floor_model(AE_' + self.AE_floor_bottleneck + ')-Conv(' + self.floor_layers + ').h5')))
##################fit_new_floor
        if Train_New_Floor:
            # adamm = Adam()
            self.new_floor_model.compile(
                loss=self.loss,
                optimizer=self.opt,
                metrics=['accuracy']
            )
            self.new_floor_model.fit(self.normalize_x, data_helper.oneHotEncode(self.floorID_y), epochs=166,
                                 batch_size=66,
                                 validation_data=(
                                     self.normalize_valid_x, data_helper.oneHotEncode(self.floorID_valid_y)),
                                 callbacks=[early_stopping])  # ,tensorbd])
            self.new_floor_model.save(
                os.path.join(Floor_model_dir, ('new_floor.h5')))

####################fit_Building

        if Train_Building:
            self.fnBuildBuildingModel()
            self.building_model.compile(
                loss=self.loss,
                optimizer=self.opt,
                metrics=['accuracy']
            )
            self.building_model.fit(self.normalize_x, data_helper.oneHotEncode(self.buildingID_y),validation_data=(self.normalize_valid_x,data_helper.oneHotEncode(self.buildingID_valid_y)),epochs=self.epoch_building, batch_size=66)#,callbacks=[early_stopping])
            self.building_model.save(os.path.join(Building_model_dir,('building_model(AE_' + self.AE_building_bottleneck + ')-3.h5')))
        re=[]
        if Train_AE:
            re.append(h_AE)
        else:
            re.append(0)
        if Train_Floor:
           re.append(h_floor)
        else:
            re.append(0)
        if Train_Location:
            re.append(h_pos)
        else:
            re.append(0)
        return tuple(re)
    def predict(self, x):
        predict_buildingID=[]
        predict_floorID=[]
        predict_longitude=[]
        predict_latitude=[]
        x = data_helper.normalizeX(x,self.b)

#################predict_Location
        if Run_Location:
            self.position_model = load_model('Location_model/Location_model.h5')
            predict_Location= self.position_model.predict(x)
            predict_longitude, predict_latitude = self.normY.reverse_normalizeY(predict_Location[:, 0],
                                                                             predict_Location[:, 1])

##################predict_floor
        if Train_New_Floor:
            predict_floorID = self.new_floor_model.predict(x)
            predict_floorID = data_helper.oneHotDecode(predict_floorID)
        elif Run_Floor:
            self.floor_model = load_model('Floor_model/floor_model(AE_' + self.AE_floor_bottleneck + ')-Conv(' + self.floor_layers + ').h5')
            predict_floorID = self.floor_model.predict(x)
            predict_floorID = data_helper.oneHotDecode(predict_floorID)

################predict_building
        if Run_Building:
            self.building_model= load_model(os.path.join(Building_model_dir,('building_model(AE_' + self.AE_building_bottleneck + ')-3.h5')))
            predict_buildingID = self.building_model.predict(x)
            predict_buildingID = data_helper.oneHotDecode(predict_buildingID)

        return predict_buildingID,predict_floorID ,predict_longitude,predict_latitude

    def error(self, x, y):
        _y = self.predict(x)
        building_right = np.sum(np.equal(np.round(_y[0]), y[:, 3]))
        floor_right=np.sum(np.equal(np.round(_y[1]),y[:,2]))
        predict_long=np.reshape(_y[2],(1,len(_y[2])))
        predict_lati=np.reshape(_y[3],(1,len(_y[3])))
        longitude_error = np.mean(np.sqrt(np.square(predict_long - y[:, 0])))
        latitude_error = np.mean(np.sqrt(np.square(predict_lati - y[:, 1])))
        mean_error=np.mean(np.sqrt(np.square(predict_long - y[:, 0])+np.square(predict_lati - y[:, 1])))
        return  building_right,floor_right,longitude_error,latitude_error,mean_error


