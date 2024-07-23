
#Libraries
#------------------------------------------
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import models
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
import torch

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

#------------------------------------
#Constants
#-------------------------------------------
MAIN_FILE = 'customized_daily_rainfall_data.csv'
FILE_NOT_FOUND = "File not found!:/ "
SEED_NUMBER = 2832163
D_TYPE = tf.float16
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
gpus = tf.config.list_physical_devices('GPU')

print(tf.config.list_physical_devices('GPU') )

def organize_information():
    try:
        with open(MAIN_FILE) as csvfile:
            csv_reader = pd.read_csv(csvfile)
            search_station = {}
            count_rainFall = {}
            data_list = []
            for index, row in csv_reader.iterrows():
                day = str(row['Day'])
                year = str(row['Year'])
                month = str(row['Month'])
                station_index = row['Station']
                data_list.append(row['Rainfall'])
                date = day+'-'+month+'-'+year
                count_rainFall[date] = row['Rainfall']
                search_station[station_index] = count_rainFall
            
            # Create the input and the output for the RNN
            x = tf.constant(data_list[:-1], dtype=D_TYPE)
            y = tf.constant(data_list[1:], dtype=D_TYPE)
            x = tf.expand_dims(x, axis = 0)
            y = tf.expand_dims(y, axis = 0)
            return x, y, search_station
    except FileNotFoundError:
        print(FILE_NOT_FOUND)

x,y,inf_dict = organize_information()
print("x:", x)
print("y:", y)



#----------------------------------------------------------- 
#Setting and splitting the data
tf.random.set_seed(SEED_NUMBER)
x_max_value = tf.reduce_max(x, axis = 0, keepdims=False)
print(x.device, y.device, x.shape, y.shape)
xhat = tf.random.uniform(shape=x.shape, seed= SEED_NUMBER, minval=0, dtype= D_TYPE, maxval=x_max_value*1.5) 
x_train, x_eval, x_test = x
#I found the data have so many variance that I prefer to make a prediction with a little more sense and the error be higher than output values that doesnt even add up... 
#that's why i put  maxval=x_max_value * 1.5
#-------------------------------------------
#Plot the inputs