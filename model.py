from keras import models
from keras import layers
from keras import Input
from keras import losses
from keras import optimizers
from keras import regularizers
from init import x_train, y_train, x_eval, y_eval, x_test, y_test, xhat, yhat, D_TYPE, SEED_NUMBER
n_epochs = 300
n_batch = 4
n_layers = 4
init_lr = 1*10**-2
decay_rate = 1
decaying_lr = init_lr/(1+ (decay_rate* n_epochs))
#Define model
model = models.Sequential()
model.add(Input(shape=xhat.shape, batch_size=n_batch, dtype= D_TYPE, tensor= xhat))
model.add(layers.Dense(n_layers, activation='relu', bias_regularizer=regularizers.L2(1e-4), kernel_regularizer= regularizers.L1L2(l1=1e-5, l2=1e-4)), activity_regularizer=regularizers.L2(1e-5))
model.add(layers.Dense(n_layers, activation='relu'))
model.add(layers.Dense(1,activation='linear'))

#compile the keras model
model.compile(loss= losses.Poisson(), optimizer= optimizers.AdamW(learning_rate=decaying_lr), metrics=['accuracy'])
model.fit(xhat, yhat, epochs= n_epochs, validation_data=(x_train, y_train), batch_size=n_batch)
model.evaluate(xhat, yhat, batch_size=n_batch, sample_weight=(x_eval, y_eval))
model.predict(xhat)
model.summary()
