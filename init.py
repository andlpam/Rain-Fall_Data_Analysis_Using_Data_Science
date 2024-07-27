
#Libraries
#------------------------------------------
import os 
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
tf.random.set_seed(SEED_NUMBER)
def initializing_data_for_train(SIF, SFP, data):
    
    index = list(range(0, len(data)-SIF-SFP-1))
    x_sample = []
    y_sample = []

    for i in index:
        x_sample.append(data[i:i + SFP])
        y_sample.append(data[i + SFP:i + SFP + SIF])

    x_train = tf.constant(x_sample[:int(len(x_sample) * 0.80)], dtype=D_TYPE)
    x_tmp = tf.constant(x_sample[len(x_sample):], dtype = D_TYPE)
    x_eval = tf.constant(x_tmp[:int(len(x_tmp)* 0.5)], dtype = D_TYPE)
    x_test = tf.constant(x_tmp[int(len(x_tmp)* 0.5):], dtype=D_TYPE)

    y_train = tf.constant(y_sample[:int(len(y_sample) * 0.80)], dtype=D_TYPE)
    y_tmp = tf.constant(y_sample[len(y_sample):], dtype = D_TYPE)
    y_eval = tf.constant(y_tmp[:int(len(y_tmp)* 0.5)], dtype = D_TYPE)
    y_test = tf.constant(y_tmp[int(len(y_tmp)* 0.5):], dtype=D_TYPE)


    rainfall_mean_x= np.sum(x_sample, axis = 0) / len(x_sample)
    rainfall_stddiv_x= (1/len(x_sample)*np.sum(np.square(x_sample - rainfall_mean_x), axis = 0))
    rainfall_mean_y = np.sum(y_sample, axis = 0) / len(y_sample)
    rainfall_stddiv_y = (1/len(y_sample)*np.sum(np.square(y_sample - rainfall_mean_y), axis = 0))

    xhat = tf.random.normal(shape=x_train.shape, seed= SEED_NUMBER, mean=rainfall_mean_x, dtype= D_TYPE, stddev= rainfall_stddiv_x) 
    yhat = tf.random.normal(shape=y_train.shape, seed= SEED_NUMBER, mean=rainfall_mean_y, dtype= D_TYPE, stddev= rainfall_stddiv_y)
    return x_train, y_train, x_eval, y_eval, x_test, y_test, xhat, yhat
x_train, y_train, x_eval, y_eval, x_test, y_test, xhat, yhat = initializing_data_for_train(SIF,SFP, data_list)
 

#I found the data have so many variance that I prefer to make a prediction with a little more sense and the error be higher than output values that doesnt even add up... 
#that's why i put  maxval=x_max_value * 1.5
#-------------------------------------------
#Plot the inputs

# Function to convert tensor to pandas DataFrame
def plot_data(x_sample, xhat, y_sample, yhat):
    def get_data(tensor):
        tensor = tensor.numpy()
        flattened_tensor = tensor.reshape(-1, tensor.shape[-1])
        df = pd.DataFrame(flattened_tensor, columns=['Date/Time', 'Total Precip (mm)'])
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], unit='s')
        return df

    # Convert tensors to DataFrames
    weather1 = get_data(x_sample)
    rainfall1 = weather1['Total Precip (mm)'].to_list()
    date1 = [x.to_pydatetime() for x in weather1['Date/Time']]

    weather2 = get_data(xhat)
    rainfall2 = weather2['Total Precip (mm)'].to_list()
    date2 = [x.to_pydatetime() for x in weather2['Date/Time']]

    # Plot first graph 
    fig1 = plt.figure(figsize=(10.0, 7.0))
    ax1 = fig1.add_axes([0,0,1,1])
    ax1.plot(date1, rainfall1)
    ax1.set_xlabel('date')
    ax1.set_ylabel('precipitation (mm)')
    ax1.set_title('Bangladesh rainfall x_sample')
    fig1.savefig('bangladesh-rainfall-x_sample.png', bbox_inches='tight')

    # Plot second graph
    fig2 = plt.figure(figsize=(10.0, 7.0))
    ax2 = fig2.add_axes([0,0,1,1])
    ax2.plot(date2, rainfall2)
    ax2.set_xlabel('date')
    ax2.set_ylabel('precipitation (mm)')
    ax2.set_title('Bangladesh rainfall xhat')
    fig2.savefig('bangladesh-rainfall-xhat.png', bbox_inches='tight')

    plt.show()
