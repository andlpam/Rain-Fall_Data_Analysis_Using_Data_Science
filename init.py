
#Libraries
#------------------------------------------
import os 
import datetime
import re
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

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
RANDOM_STATE = 686958
D_TYPE = tf.float16

gpus = tf.config.list_physical_devices('GPU')

print(tf.config.list_physical_devices('GPU') )

def organize_information():
    try:
        with open(MAIN_FILE) as csvfile:
            csv_reader = pd.read_csv(csvfile)
            search_station = {}
            count_rainFall = {}
            data_list = []
            for _, row in csv_reader.iterrows():
                day = str(row['Day'])
                year = str(row['Year'])
                month = str(row['Month'])
                station_index = row['Station']
                data_list.append(row['Rainfall'])
                date = day+'-'+month+'-'+year
                count_rainFall[date] = row['Rainfall']
                search_station[station_index] = count_rainFall
            
            # Create the input and the output for the RNN
            return data_list,search_station
    except FileNotFoundError:
        print(FILE_NOT_FOUND)

data_list, station_dict = organize_information()


#----------------------------------------------------------- 
#Setting and splitting the data

SIF = 7 #Steps into future
SFP = 14 #Steps from past
x_train = data_list[:int(len(data_list) * 0.80)]
x_tmp = data_list[len(x_train):]
x_eval = x_tmp[:int(len(x_tmp)* 0.5)]
x_test = x_tmp[int(len(x_tmp)* 0.5):]

index = list(range(0, len(x_train)-SIF-SFP-1))
x_sample = []
y_sample = []

for i in index:
  x_sample.append(x_train[i:i + SFP])
  y_sample.append(x_train[i + SFP:i + SFP + SIF])

x_sample= tf.constant(x_sample, dtype=D_TYPE)
y_sample= tf.constant(y_sample, dtype=D_TYPE)

tf.random.set_seed(SEED_NUMBER)
rainfall_mean= np.sum(x_train, axis = 0) / len(x_train)
rainfall_stddiv= (1/len(x_train)*np.sum(np.square(x_train - rainfall_mean), axis = 0))

xhat = tf.random.normal(shape=x_sample.shape, seed= SEED_NUMBER, mean=rainfall_mean, dtype= D_TYPE, stddev= rainfall_stddiv) 

 

#I found the data have so many variance that I prefer to make a prediction with a little more sense and the error be higher than output values that doesnt even add up... 
#that's why i put  maxval=x_max_value * 1.5
#-------------------------------------------
#Plot the inputs

def yyyymmdd_parser(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d')

def get_data(tensor):
    
    df = pd.DataFrame(tensor.numpy(), columns=['Date/Time', 'Total Precip (mm)'])
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y-%m-%d')
    
    return df
#First graph data
weather1 = get_data(x_train)
rainfall1 = weather1['Total Precip (mm)'].to_list()
date1 = [x.to_pydatetime() for x in weather1['Date/Time']]

#Second graph data
weather2 = get_data(xhat)
rainfall2 = weather2['Total Precip (mm)'].to_list()
date2 = [x.to_pydatetime() for x in weather2['Date/Time']]

# Plot first graph 
fig1 = plt.figure(figsize=(10.0, 7.0))
ax1 = fig1.add_axes([0,0,1,1])
ax1.plot(date1, rainfall1)
ax1.set_xlabel('date')
ax1.set_ylabel('precipitation (mm)')
ax1.set_title('Bangladesh rainfall training data')
fig1.savefig('daily-precipitation-dataset1.png', bbox_inches='tight')

# PLot second graph
fig2 = plt.figure(figsize=(10.0, 7.0))
ax2 = fig2.add_axes([0,0,1,1])
ax2.plot(date2, rainfall2)
ax2.set_xlabel('date')
ax2.set_ylabel('precipitation (mm)')
ax2.set_title('Bangladesh rainfall real data')
fig2.savefig('daily-precipitation-dataset2.png', bbox_inches='tight')

plt.show()
