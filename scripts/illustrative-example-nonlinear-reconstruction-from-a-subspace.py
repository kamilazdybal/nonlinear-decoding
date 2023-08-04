import numpy as np
import time
from PCAfold import preprocess
import tensorflow as tf
from tensorflow.keras import layers, initializers
from keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping

random_seed = 100

angle_list = [i for i in range(0,180)]

n_points = 2000
np.random.seed(seed=10)
mean = [0,0]
covariance = [[2.5, 0.52], [0.52, 0.2]]
x_noise, y_noise = np.random.multivariate_normal(mean, covariance, n_points).T
dataset_2D = np.column_stack((x_noise, y_noise))
dataset_2D_normalized, centers_2D, scales_2D = preprocess.center_scale(dataset_2D, scaling='-1to1')

phi_1 = dataset_2D_normalized[:,0]**2 + 3*dataset_2D_normalized[:,0]
phi_2 = dataset_2D_normalized[:,0] * dataset_2D_normalized[:,1]
phi_3 = np.sin(dataset_2D_normalized[:,0]) + np.abs(dataset_2D_normalized[:,1])
phi_4 = np.exp(-(dataset_2D_normalized[:,0]**2+dataset_2D_normalized[:,1]**2) / (2.0 * 0.3**2))

phi_2D = np.column_stack((phi_1, phi_2, phi_3, phi_4))
phi_2D_normalized, centers_phi, scales_phi = preprocess.center_scale(phi_2D, scaling='-1to1')

for i in [3]:

    MSE_losses = []
    depvar = phi_2D_normalized[:,i:i+1]

    for angle in angle_list:

        tic = time.perf_counter()

        print('- '*30)

        print('Angle:\t' + str(angle) + ' degrees')

        theta = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_hat = np.array([1, 0])
        basis_vector = np.dot(rotation_matrix, x_hat)[:,None]

        data_projection = np.dot(dataset_2D_normalized, basis_vector)

        sample_random = preprocess.DataSampler(np.zeros((n_points,)).astype(int), random_seed=random_seed, verbose=False)
        (idx_train, idx_test) = sample_random.random(80)

        tf.random.set_seed(random_seed)

        input_X = Input(shape=(1,))
        decoded_1 = layers.Dense(2, activation='tanh', kernel_initializer=initializers.glorot_uniform(seed=random_seed), bias_initializer='zeros')(input_X)
        decoded_2 = layers.Dense(2, activation='tanh', kernel_initializer=initializers.glorot_uniform(seed=random_seed), bias_initializer='zeros')(decoded_1)
        decoded = layers.Dense(1, activation='tanh', kernel_initializer=initializers.glorot_uniform(seed=random_seed), bias_initializer='zeros')(decoded_2)

        autoencoder = Model(input_X, decoded)
        autoencoder.compile(tf.optimizers.legacy.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())

        weights_and_biases = autoencoder.get_weights()

        monitor = EarlyStopping(monitor='loss',
                    min_delta=1e-6,
                    patience=100,
                    verbose=0,
                    mode='auto',
                    restore_best_weights=True)

        history = autoencoder.fit(data_projection[idx_train,:], depvar[idx_train,:],
                    epochs=1000,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(data_projection[idx_test,:], depvar[idx_test,:]),
                    callbacks=[monitor],
                    verbose=0)

        n_epochs = len(history.history['loss'])
        print('AE was run with ' + str(n_epochs) + ' epochs.')

        autoencoder.set_weights(weights_and_biases)

        final_loss = history.history['val_loss'][-1]
        MSE_losses.append(final_loss)

        toc = time.perf_counter()

        print(f'Time it took for this run: {(toc - tic)/60:0.1f} minutes.\n')

    np.savetxt('synthetic-dataset-nonlinear-reconstruction-from-subspace-phi-' + str(i+1) + '-MSE-loss.csv', (MSE_losses), delimiter=',', fmt='%.16e')

    print('= '*30)

toc = time.perf_counter()

print(f'\n\n\tTotal time it took: {(toc - tic)/60:0.1f} minutes.')
