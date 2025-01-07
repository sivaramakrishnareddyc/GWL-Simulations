#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:30:05 2022

@author: chidesiv
"""


# 

modtype="GRU" #Select Model type #GRU, LSTM,BILSTM
ID=8 #8,19,57 Select station ID
seq_length=48 # Sequence length as required in the deep learning models
initm=30 #Number of initialisations as required for the final simulations
test_size=0.2 # Train test split size 


#Import all necessary libraries

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from tensorflow import device
from tensorflow.random import set_seed
from numpy.random import seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
import optuna
from optuna.samplers import TPESampler
from uncertainties import unumpy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error




gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*0.3)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
    
    
    
    
# Import and split data

data =pd.read_csv(r"/home/chidesiv/Desktop/Scripts_py/selectedstations/data/PE/Well_ID"+str(ID)+"data_PE76.csv")

data['date_p'] = pd.to_datetime(data['date_p'])

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(  data['Peff'], data["val_calc_ngf"], data['date_p'],test_size=test_size, random_state=1, shuffle=False)


#Scaling data using Min Max scaler to (0,1) which can be fitted to train data and transform on the test data

def Scaling_data(X_train,X_valid,y_train,y_valid):
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(np.array(X_train).reshape(-1,1))
    X_valid_s=Scaler.transform(np.array(X_valid).reshape(-1,1))
    target_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler.fit(np.array(y_train).reshape(-1,1))
    y_train_s=target_scaler.transform(np.array(y_train).reshape(-1,1))
    y_valid_s=target_scaler.transform(np.array(y_valid).reshape(-1,1))
    X_C_scaled=np.concatenate((X_train_s,X_valid_s),axis=0)
    y_C_scaled=np.concatenate((y_train_s,y_valid_s),axis=0)
    return X_train_s,X_valid_s,y_train_s,y_valid_s,X_C_scaled,y_C_scaled



def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


X_train_s, X_valid_s, y_train_s, y_valid_s,X_C_scaled,y_C_scaled=Scaling_data( X_train, X_test, y_train, y_test)
 

X_c, y_c = reshape_data(X_C_scaled,y_C_scaled,seq_length=seq_length)



X_train_l=X_c[0:int((len(X_c)+seq_length-1)*(1-test_size)-seq_length)]
y_train_l=y_c[0:int((len(X_c)+seq_length-1)*(1-test_size)-seq_length)]
X_valid_l=X_c[int((len(X_c)+seq_length-1)*(1-test_size)-seq_length+1):]
y_valid_l=y_c[int((len(X_c)+seq_length-1)*(1-test_size)-seq_length+1):]



X_train_ls, X_valid_ls, y_train_ls, y_valid_ls  = train_test_split(X_train_l, y_train_l , test_size=0.2,random_state=1,shuffle=False)






def func_dl(trial):
   with device('/gpu:0'):    
    #     tf.config.experimental.set_memory_growth('/gpu:0', True)    
        set_seed(2)
        seed(1)
        
        
        
        optimizer_candidates={
            "adam":Adam(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "SGD":SGD(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "RMSprop":RMSprop(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True))
        }
        optimizer_name=trial.suggest_categorical("optimizer",list(optimizer_candidates))
        optimizer1=optimizer_candidates[optimizer_name]

    
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights = True),TFKerasPruningCallback(trial, monitor='val_loss')]
        
        epochs=trial.suggest_int('epochs', 50, 500,step=50)
        batch_size=trial.suggest_int('batch_size', 16,256,step=16)
        #weight=trial.suggest_float("weight", 1, 5)
        n_layers = trial.suggest_int('n_layers', 1, 6)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 10, 100,step=10)
            return_sequences = True
            if i == n_layers-1:
                return_sequences = False
            # Activation function for the hidden layer
            if modtype == "GRU":
                model.add(GRU(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences))
            elif modtype == "LSTM":
                model.add(LSTM(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences))
            elif modtype == "BILSTM":
                model.add(Bidirectional(LSTM(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences)))
            model.add(Dropout(trial.suggest_float("dropout_l{}".format(i), 0.2, 0.2), name = "dropout_l{}".format(i)))
            #model.add(Dense(units = 1, name="dense_2", kernel_initializer=trial.suggest_categorical("kernel_initializer",['uniform', 'lecun_uniform']),  activation = 'Relu'))
        #model.add(BatchNormalization())study_blstm_
        model.add(Dense(1))
        model.compile(optimizer=optimizer1,loss="mse",metrics=['mse'])
        ##model.summary()


        model.fit(X_train_ls,y_train_ls,validation_data = (X_valid_ls,y_valid_ls ),shuffle = False,batch_size =batch_size,epochs=epochs,callbacks=callbacks
                  ,verbose = False)
  
        score=model.evaluate(X_valid_ls,y_valid_ls)

       

        return score[1]


# Study_DL= optuna.create_study(direction='minimize',sampler=TPESampler(seed=10),study_name='study_'+str(modtype)+str(ID))
# Study_DL.optimize(func_dl,n_trials=100)

modtype="BILSTM" #Select Model type #GRU, LSTM,BILSTM
ID=57 #8,19,57 Select station ID
import pickle
# pickle.dump(Study_DL,open('./'+'study_'+str(modtype)+str(ID)+'.pkl', 'wb'))
with open('/home/chidesiv/Desktop/Scripts_py/revised_version_scripts/PE/20pct_SA/'+'study_'+str(modtype)+str(ID)+'.pkl','rb') as f:
  Study_DL=pickle.load(f)


par = Study_DL.best_params
par_names = list(par.keys())
par

# def optimizer_1(learning_rate,optimizer):
#         tf.random.set_seed(init+11111)
#         if optimizer==Adam:
#             opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         elif optimizer==SGD:
#             opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#         else:
#             opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
#         return opt

# #@jit(nopython=True)
# def gwlmodel(init, params):
#         with tf.device('/gpu:0'):
#             seed(init+99999)
#             tf.random.set_seed(init+11111)
#             par= params.best_params
#             # seq_length = par.get(par_names[0])
#             learning_rate=par.get(par_names[0])
#             optimizer = par.get(par_names[1])
#             epochs = par.get(par_names[2])
#             batch_size = par.get(par_names[3])
#             n_layers = par.get(par_names[4])
#             # X_train, X_valid, y_train, y_valid=Scaling_data( X_tr'+ str(file_c['code_bss'][k])], X_va'+ str(file_c['code_bss'][k])], y_t'+ str(file_c['code_bss'][k])], y_v'+ str(file_c['code_bss'][k])])
#             # X_train_l, y_train_l = reshape_data(X_train, y_train,seq_length=seq_length)
#             model = Sequential()
#             # i = 1
#             for i in range(n_layers):
#                 return_sequences = True
#                 if i == n_layers-1:
#                     return_sequences = False
#                 if modtype == "GRU":
#                     model.add(GRU(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
#                 elif modtype == "LSTM":
#                     model.add(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
#                 elif modtype == "BILSTM":
#                     model.add(Bidirectional(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences)))
#                 model.add(Dropout(par["dropout_l{}".format(i)]))
#             model.add(Dense(1))
#             opt = optimizer_1(learning_rate,optimizer)
#             model.compile(optimizer = opt, loss="mse",metrics = ['mse'])
#             callbacks = [EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience=50,restore_best_weights = True), tf.keras.callbacks.ModelCheckpoint(filepath='best_model'+str(modtype)+str(init)+str(ID)+'.h5', monitor='val_loss', save_best_only = True, mode = 'min')]
#             model.fit(X_train_l, y_train_l,validation_split = 0.2, batch_size=batch_size, epochs=epochs,callbacks=callbacks)
#         return model
    

target_scaler = MinMaxScaler(feature_range=(0,1))
target_scaler.fit(np.array(y_train).reshape(-1,1))
sim_init= np.zeros((len(X_valid_l), initm))
sim_init[:]=np.nan
sim_tr= np.zeros((len(X_train_l), initm))
sim_tr[:]=np.nan


for init in range(initm):
    # globals()["model"+str(init)]= gwlmodel(init,Study_DL)
    globals()["model"+str(init)]=tf.keras.models.load_model('./'+'best_model'+str(modtype)+str(init)+str(ID)+'.h5')
    globals()["y_pred_valid"+str(init)] = globals()["model"+str(init)].predict(X_valid_l)
    globals()["sim_test"+str(init)]  = target_scaler.inverse_transform(globals()["y_pred_valid"+str(init)])
    globals()["y_pred_train"+str(init)] = globals()["model"+str(init)].predict(X_train_l)
    globals()["sim_train"+str(init)] = target_scaler.inverse_transform(globals()["y_pred_train"+str(init)])
    
    
    
for init in range(initm):
    sim_init[:,init]=globals()["sim_test"+str(init)][:,0]
    sim_tr[:,init]=globals()["sim_train"+str(init)][:,0]
    # sim_tc[:,init]=globals()["sim_c"+str(init)][:,0]
    
    sim_f=pd.DataFrame(sim_init)
    sim_t=pd.DataFrame(sim_tr)
    # sim_co=pd.DataFrame(sim_tc)
    sim_mean=sim_f.mean(axis=1) 
    sim_tr_mean=sim_t.mean(axis=1) 
    # sim_tc_mean=sim_co.mean(axis=1) 
    sim_init_uncertainty = unumpy.uarray(sim_f.mean(axis=1),1.96*sim_f.std(axis=1))
    sim_tr_uncertainty = unumpy.uarray(sim_t.mean(axis=1),1.96*sim_t.std(axis=1))
    # sim_tc_uncertainty = unumpy.uarray(sim_co.mean(axis=1),1.96*sim_co.std(axis=1))
    sim=np.asarray(sim_mean).reshape(-1,1)
    sim_train=np.asarray(sim_tr_mean).reshape(-1,1)
    # sim_comb=np.asarray(sim_tc_mean).reshape(-1,1)
    obs = np.asarray(target_scaler.inverse_transform(y_valid_l).reshape(-1,1))
    obs_tr = np.asarray(target_scaler.inverse_transform(y_train_l).reshape(-1,1))
    # obs_c = np.asarray(target_scaler.inverse_transform(globals()['y_c'+str(file_c['code_bss'][k])]).reshape(-1,1))
    y_err = unumpy.std_devs(sim_init_uncertainty)
    y_err_tr = unumpy.std_devs(sim_tr_uncertainty)
    # y_err_tc = unumpy.std_devs(sim_tc_uncertainty)
    

MAE_va=mean_absolute_error(obs, sim)
mse_va=mean_squared_error(obs, sim)
R2_va=r2_score(obs, sim)
RMSE_va=math.sqrt(mse_va)


MAE_tr=mean_absolute_error(obs_tr, sim_train)
mse_tr=mean_squared_error(obs_tr, sim_train)
R2_tr=r2_score(obs_tr, sim_train)
RMSE_tr=math.sqrt(mse_tr)

scores_tr = pd.DataFrame(np.array([[ R2_tr, RMSE_tr, MAE_tr]]),
                   columns=['R2_tr','RMSE_tr','MAE_tr'])

scores_va = pd.DataFrame(np.array([[ R2_va, RMSE_va, MAE_va]]),
                   columns=['R2_va','RMSE_va','MAE_va'])
print(scores_tr)
print(scores_va)

from datetime import datetime
from datetime import timedelta





import matplotlib.ticker as ticker
from matplotlib import pyplot
pyplot.figure(figsize=(20,6))
Train_index=indices_train
Test_index=indices_test

# Middle_Train_Index = Train_index.iloc[180] + timedelta(days=len(Train_index)/2)
# Middle_Val_Index = Val_index[0] + timedelta(days=len(Val_index)/4)
# Middle_Val_Index = Train_index[-1] -timedelta(days=len(Train_index)/4)
       
# pyplot.plot( Train_index,y_train, 'k', label ="observed train", linewidth=1.5,alpha=0.9)

# pyplot.fill_between(indices_train[f+seq_length-1:],sim_train.reshape(-1,) - y_err_tr,
                # sim_train.reshape(-1,) + y_err_tr, facecolor = (1.0, 0.8549, 0.72549),
                # label ='95% confidence training',linewidth = 1,
                # edgecolor = (1.0, 0.62745, 0.47843))    
pyplot.plot( Test_index,sim, 'r-^', label ="simulated mean", linewidth = 0.9)
        
pyplot.plot( Test_index,obs, 'k--', label ="observed", linewidth=1.5,alpha=0.9)
pyplot.ylim(60, 76)

pyplot.fill_between( Test_index,sim.reshape(-1,) - y_err,
                sim.reshape(-1,) + y_err, facecolor = (1,0.7,0,0.4),
                label ='95% confidence',linewidth = 1,
                edgecolor = (1,0.7,0,0.7))    
Middle_Test_Index = Test_index.iloc[-18]-timedelta(days=len(Test_index)/4)
Middle_Target_Index = obs.max()*0.95 #- y_train.min()/2
# #text
# pyplot.text(Middle_Train_Index, Middle_Target_Index, 'Train RMSE =' +str("%.2f" % scores_tr.RMSE_tr[0]) + '\nTrain MAE=' + str("%.2f" % scores_tr.MAE_tr[0])+ '\nTrain R2=' + str("%.2f" % scores_tr.R2_tr[0]),fontsize = 15)
# pyplot.text(Middle_Test_Index, Middle_Target_Index,'RMSE =' + str("%.2f" % scores_va.RMSE_va[0]) + '\n MAE=' + str("%.2f" % scores_va.MAE_va[0])+ '\n R2=' + str("%.2f" % scores_va.R2_va[0]),fontsize = 24,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
# pyplot.plot( indices_train[f+seq_length-1:],sim_train, 'r', label ="simulated  median train", linewidth = 0.9)
 
# pyplot.plot( globals()['MM'+str(file_c['code_bss'][k])].index[seq_length-1:],sim_comb, 'r', label ="simulated median combined", linewidth = 0.9)
        
# pyplot.plot( globals()['MM'+str(file_c['code_bss'][k])].index[seq_length-1:],obs_c, 'k', label ="observed combined", linewidth=1.5,alpha=0.9)


# pyplot.vlines(x=indices_test.iloc[0], ymin=[y_train.min()*0.9], ymax=[y_train.max()*1.1], colors='teal', ls='--', lw=2, label='Start of Testing data')
# pyplot.fill_between(globals()['MM'+str(file_c['code_bss'][k])].index[seq_length-1:],sim_comb.reshape(-1,) - y_err_tc,
#                 sim_comb.reshape(-1,) + y_err_tc, facecolor = (1,0.7,0,0.5),
#                 label ='95% confidence_test',linewidth = 1,
#                 edgecolor = (1,0.7,0,0.7))   
# # plt.vlines(x=Test_index.values[0], ymin=[0], ymax=[Data[Target[i]].max()*1.4], colors='teal', ls='--', lw=2, label='Start time of filling new data')
# pyplot.legend(fontsize=15,bbox_to_anchor=(1.18, 1),loc='upper right',fancybox = False, framealpha = 1, edgecolor = 'k')
# pyplot.legend(ncol=2,fontsize=24,loc='upper left')

pyplot.title(str(modtype)+' '+ "PE"+' - '+ 'Test set RMSE :' + str("%.2f" % scores_va.RMSE_va[0]), size=24,fontweight = 'bold')
# pyplot.ylabel('GWL [m asl]', size=48)
# pyplot.xlabel('Date',size=48)
# pyplot.legend(fontsize=15,bbox_to_anchor=(1.18, 1),loc='best',fancybox = False, framealpha = 1, edgecolor = 'k')
pyplot.tight_layout()
# pyplot.grid(visible=True, which='major', color='#666666', alpha = 0.3, linestyle='-')

# tick_spacing = 1
# pyplot.xticks(fontsize=36)
pyplot.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
pyplot.yticks(fontsize=36)

# s_tr = """R²  = {:.2f}\nRMSE = {:.2f}\nMAE = {:.2f}\n""".format(scores.R2[0],
# scores.RMSE[0],scores.MAE[0],
# )


# plt.show()

# s_tr = """R²  = {:.2f}\nRMSE = {:.2f}\nMAE = {:.2f}\n""".format(scores.R2[0],
# scores.RMSE[0],scores.MAE[0],
# # )
# pyplot.figtext(0.872, 0.18, s, bbox=dict(facecolor='white'),fontsize = 15)
pyplot.savefig('./'+'Well_ID'+str(ID)+str(modtype)+str(initm)+'PE_test.png', dpi=600, bbox_inches = 'tight', pad_inches = 0.1)



# mae_init= np.zeros((initm,1))
# mae_init[:]=np.nan


# mse_init= np.zeros((initm,1))
# mse_init[:]=np.nan

# rmse_init= np.zeros((initm,1))
# rmse_init[:]=np.nan

# r2_init= np.zeros((initm,1))
# r2_init[:]=np.nan


# for init in range(initm):
  
#     mae_init[init,:]=mean_absolute_error(obs, sim_init[:,init])
#     mse_init[init,:]=mean_squared_error(obs, sim_init[:,init])
#     rmse_init[init,:]=math.sqrt(mse_init[init,:])
#     r2_init[init,:]=r2_score(obs, sim_init[:,init])
    
    
# pd.DataFrame(mae_init).to_csv('./'+'Well_ID'+str(ID)+str(modtype)+'mae.csv')    
# pd.DataFrame(mse_init).to_csv('./'+'Well_ID'+str(ID)+str(modtype)+'mse.csv')   
# pd.DataFrame(rmse_init).to_csv('./'+'Well_ID'+str(ID)+str(modtype)+'rmse.csv')   
# pd.DataFrame(r2_init).to_csv('./'+'Well_ID'+str(ID)+str(modtype)+'r2.csv') 


# import matplotlib.pyplot as plt

# import shap      
# # import tensorflow.compat.v1.keras.backend as K
# # import tensorflow as tf
# # tf.compat.v1.disable_eager_execution()        
# # #SHAP VALUES com
# for init in range(initm):
#     # model = tf.keras.models.load_model('./'+'best_model'+str(k)+str(init)+'.h5')
#     # background = 
#     globals()["e"+str(init)] = shap.DeepExplainer((globals()["model"+str(init)].layers[0].input,globals()["model"+str(init)].layers[-1].output), X_train_l)
#     globals()["shap_values"+str(init)] = globals()["e"+str(init)].shap_values(X_valid_l)
   
    
#     shap_vals = np.asarray(globals()["shap_values"+str(init)][0])
#     shap_vals = shap_vals.reshape(-1, shap_vals.shape[-1])
    
    
#     if init == 0:
#              s = shap_vals
#              X = X_valid_l#.reshape(-1, X_valid_l)
#     else:
#              s = np.append(s,shap_vals,0)
#              # X = np.append(X,X_valid_l#.reshape(-1, X_valid_l.shape[-1]),0)
#              X = np.append(X,X_valid_l,0)
#     # np.savetxt('./'+
#     # shap.summary_plot( globals()["shap_values"+str(init)][0], plot_type = 'bar', feature_names = globals()['X_tr'+str(file_c['code_bss'][k])].columns)
# shap.summary_plot(s, X,feature_names = data.columns[1], show=False)

# plt.savefig('./'+'SHAP'++str(ID)+str(modtype)+str(initm)+'.png', dpi=300, bbox_inches = 'tight', pad_inches = 0.1)