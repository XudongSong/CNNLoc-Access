from encoder_model import EncoderDNN
import numpy as np
import data_helper
import time
import keras
import csv
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax,SGD,RMSprop
from keras.losses import MSE,MAE,MAPE,MSLE,KLD,squared_hinge,hinge,categorical_hinge,categorical_crossentropy,sparse_categorical_crossentropy,kullback_leibler_divergence,poisson

os.environ["CUDA_VISIBLE_DEVICES"]='0'
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
set_session(tf.Session(config=config))

# OPT=[Adam,Adagrad,Nadam,Adamax,RMSprop]
OPT=[RMSprop,Adamax]

LOSS=[MSE]
LR_34=[0.005,0.001,0.0005,0.0001]
# LR_23=[0.05,0.01,0.005,0.001]
LR_23=[0.001,0.001,0.001]
LR_15=[0.005,0.005,0.005]
rng=np.random.RandomState(888)

base_dir= os.getcwd()
# train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'TestData.csv')
valid_csv_path=os.path.join(base_dir,'ValuationData.csv')
train_csv_path=os.path.join(base_dir,'TrainingData.csv')

log_dir='New_access_log.txt'
if __name__ == '__main__':
    # Load data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,test_csv_path)

    b = 2.8
    p = 3
    epoch_sae = 40
    epoch_building=40
    epoch_floor = 40
    epoch_position = 60
    dp = 0.7
    info="""
    b = 2.8
    p = 3
    epoch_sae = 40
    epoch_floor = 40
    epoch_building = 40
    epoch_position = 60
    dp = 0.7
    """
    with open(log_dir, 'a') as file:
        file.write('\n' + info)

    for loss in LOSS:
        for opt in OPT:
            if opt==RMSprop:
                LR=LR_23
            else:
                LR=LR_15

            for lr in LR:
                # Training
                encode_dnn_model = EncoderDNN()
                encode_dnn_model.patience=int(p)
                encode_dnn_model.b=b
                encode_dnn_model.epoch_AE=epoch_sae
                encode_dnn_model.epoch_floor=epoch_floor
                encode_dnn_model.epoch_position=epoch_position
                encode_dnn_model.epoch_building=epoch_building
                encode_dnn_model.dropout=dp
                encode_dnn_model.loss=loss
                encode_dnn_model.opt=opt(lr=lr)
                strat = time.time()
                # tbCallBack=keras.callbacks.TensorBoard(log_dir='./Graph',
                #                                        histogram_freq=1,
                #                                        write_graph=True,
                #                                        write_images=True)

                h=encode_dnn_model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)#,tensorbd=tbCallBack)
                end=time.time()
                trining_time=end-strat
                # if not isinstance(h[0],int):
                #
                #     # Plot training & validation loss values
                #     with open(log_dir,'a') as f:
                #         f.write('\n\nSAE_training_log bpde'+name+'\n')
                #         f.write('training loss:\t'+str(h[0].history['loss']))
                #         f.write('\nvalid loss:\t'+str(h[0].history['val_loss']))
                #     plt.plot(h[0].history['loss'])
                #     plt.plot(h[0].history['val_loss'])
                #     plt.title('SAE Model loss')
                #     plt.ylabel('Loss')
                #     plt.xlabel('Epoch')
                #     plt.legend(['Train', 'Test'], loc='upper left')
                #     plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'saeLoss.png')
                #     plt.clf()
                #
                # if not isinstance(h[1],int):
                #     # Plot training & validation accuracy values
                #     with open(log_dir,'a') as f:
                #         f.write('\n\ndropout rate='+name)
                #         f.write('\n\nFloor_training_acc_log bpde'+name+'\n')
                #         f.write('training acc:\n'+str(h[1].history['acc'])[1:-1])
                #         f.write('\nvalid acc:\n'+str(h[1].history['val_acc'])[1:-1])
                #     plt.plot(h[1].history['acc'])
                #     plt.plot(h[1].history['val_acc'])
                #     plt.title('Floor Model accuracy')
                #     plt.ylabel('Accuracy')
                #     plt.xlabel('Epoch')
                #     plt.legend(['Train', 'Test'], loc='upper left')
                #     plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'flooracc.png')
                #     plt.clf()
                #     # Plot training & validation loss values
                #     with open(log_dir,'a') as f:
                #         f.write('\n\nFloor_training_loss_log bpde'+name+'\n\n')
                #         f.write('training loss:\n'+str(h[1].history['loss'])[1:-1])
                #         f.write('\nvalid loss:\n'+str(h[1].history['val_loss'])[1:-1])
                #     plt.plot(h[1].history['loss'])
                #     plt.plot(h[1].history['val_loss'])
                #     plt.title('Floor Model loss')
                #     plt.ylabel('Loss')
                #     plt.xlabel('Epoch')
                #     plt.legend(['Train', 'Test'], loc='upper left')
                #     plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'floorLoss.png')
                #     plt.clf()
                #
                # if not isinstance(h[2], int):
                #     # Plot training & validation loss values
                #     with open(log_dir,'a') as f:
                #         f.write('\n\nLocation_training_log bpde'+name+'\n\n')
                #         f.write('training loss:\n'+str(h[2].history['loss'])[1:-1])
                #         f.write('\nvalid loss:\n'+ str(h[2].history['val_loss'])[1:-1])
                #     plt.plot(h[2].history['loss'])
                #     plt.plot(h[2].history['val_loss'])
                #     plt.title('Location Model loss')
                #     plt.ylabel('Loss')
                #     plt.xlabel('Epoch')
                #     plt.legend(['Train', 'Test'], loc='upper left')
                #     plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'LocationLoss.png')
                #     plt.clf()

                building_right, floor_right, longitude_error, latitude_error, mean_error=encode_dnn_model.error(test_x, test_y)


                del encode_dnn_model

                with open(log_dir,'a') as file:
                    file.write('\nloss,opt,lr,b_hr,f_hr,pos_longi_err,pos_lati_err,mean_err,time')
                    file.write('\n'+str(loss)+','+str(opt)+','+str(lr)+','+str((building_right/1111.0)*100)+'%,'+str((floor_right/1111.0)*100)+'%,'+str(longitude_error)+','+str(latitude_error)+','+str(mean_error)+','+str(end-strat))
