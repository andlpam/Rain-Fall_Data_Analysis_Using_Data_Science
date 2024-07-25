from keras import models
from keras import layers
from keras import Input
from keras import losses
from init import x, y, xhat, D_TYPE, SEED_NUMBER

n_batch = 4
n_layers = 4
#Define model
model = models.Sequential()
model.add(Input(shape=(x.shape), batch_size=n_batch, dtype= D_TYPE, tensor= xhat))
model.add(layers.Dense(n_layers, activation='relu', input_shape = (x.shape)))
model.add(layers.Dense(n_layers, activation='relu'))
model.add(layers.Dense(1,activation='linear'))

#compile the keras model
model.compile(loss= losses.Poisson(), optimizer='adamW')
