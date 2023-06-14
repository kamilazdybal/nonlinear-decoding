import numpy as np
import pandas as pd
import time
import pickle
import os
from PCAfold import preprocess
from PCAfold import analysis
from PCAfold import __version__ as PCAfold_version

print('numpy==' + np.__version__)
print('pandas==' + pd.__version__)
print('PCAfold==' + PCAfold_version)

# Select one dataset:
data_tag = 'H2'
# data_tag = 'CO-H2'
# data_tag = 'CH4'
# data_tag = 'C2H4'

########################################################################
## Load data - Hydrogen/air flamelet
########################################################################

if data_tag == 'H2':
    
    state_space = pd.read_csv('../data/' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()[:,0:-2]
    state_space_sources = pd.read_csv('../data/' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()[:,0:-2]
    state_space_names = pd.read_csv('../data/' + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()[0:-2]

    selected_state_variables = [0, 2, 4, 5, 6, 8]

########################################################################
## Load data - Syngas/air flamelet
########################################################################

if data_tag == 'CO-H2':
    
    state_space = pd.read_csv('../data/' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_sources = pd.read_csv('../data/' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_names = pd.read_csv('../data/' + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()[0:-1]

    (n_observations, n_variables) = np.shape(state_space)

    print('\nThe data set has ' + str(n_observations) + ' observations.')
    print('\nThe data set has ' + str(n_variables) + ' variables.')

    selected_state_variables = [0, 1, 2, 4, 5, 8, 9]

########################################################################
## Load data - Methane/air flamelet
########################################################################

if data_tag == 'CH4':
   
    state_space = pd.read_csv('../data/' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_sources = pd.read_csv('../data/' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_names = pd.read_csv('../data/' + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()[0:-1]

    species_to_remove = 'N2'
    (species_index, ) = np.where(state_space_names==species_to_remove)
    state_space = np.delete(state_space, np.s_[species_index], axis=1)
    state_space_sources = np.delete(state_space_sources, np.s_[species_index], axis=1)
    state_space_names = np.delete(state_space_names, np.s_[species_index])

    selected_state_variables = [0, 4, 5, 6, 14, 16]

########################################################################
## Load data - Ethylene/air flamelet
########################################################################

if data_tag == 'C2H4':
   
    state_space = pd.read_csv('../data/' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_sources = pd.read_csv('../data/' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()[:,0:-1]
    state_space_names = pd.read_csv('../data/' + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()[0:-1]

    selected_state_variables = [0, 4, 5, 6, 15, 22]

(n_observations, n_variables) = np.shape(state_space)

print('\nThe data set has ' + str(n_observations) + ' observations.')
print('\nThe data set has ' + str(n_variables) + ' variables.')

########################################################################
## Case settings
########################################################################

n_components = 2
random_seeds_list = [i for i in range(0,100)]
decoder_architecture = (6,9,10)
train_perc = 80
n_neighbors = 100
learning_rate = 0.001
batch_size = n_observations
n_epochs = 20000

########################################################################
## Run kernel regression
########################################################################

case_name = str(n_components) + 'D-LDM-lr-' + str(learning_rate) + '-bs-' + str(batch_size) + '-n-epochs-' + str(n_epochs) + '-architecture'
print(case_name)

architecture_without = str(n_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_architecture]) + '-' + str(len(selected_state_variables))
architecture_with = str(n_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_architecture]) + '-' + str(len(selected_state_variables) + n_components*2)

# Without Seta: - - - - - - - - - - - - - - - - - - - - - - - - -

NRMSE_Seta_1_without = []
NRMSE_Seta_2_without = []

R2_Seta_1_without = []
R2_Seta_2_without = []

for random_seed in random_seeds_list:
    
    print(random_seed)
    
    total_tic = time.perf_counter()

    basis_without_Seta = pd.read_csv('../results/QoIAwareProjection-basis-without-Seta-' + data_tag + '-' + case_name + '-' + architecture_without + '-random-seed-' + str(random_seed) + '.csv', sep = ',', header=None).to_numpy()
    
    (X_CS, centers, scales) = preprocess.center_scale(state_space, scaling='0to1')
    S_CS = state_space_sources / scales

    X_AE_without_Seta = np.dot(X_CS, basis_without_Seta)
    S_AE_without_Seta = np.dot(S_CS, basis_without_Seta)

    (X_AE_without_Seta_pp, centers_X_AE, scales_X_AE) = preprocess.center_scale(X_AE_without_Seta, '0to1')

    sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=random_seed, verbose=False)
    (idx_train, idx_test) = sample_random.random(train_perc)

    X_AE_without_Seta_pp_train = X_AE_without_Seta_pp[idx_train,:]
    OUTPUT_train = S_AE_without_Seta[idx_train,:]
    X_AE_without_Seta_pp_test = X_AE_without_Seta_pp[idx_test,:]
    OUTPUT_test = S_AE_without_Seta[idx_test,:]

    model = analysis.KReg(X_AE_without_Seta_pp_train, OUTPUT_train)

    OUTPUT_predicted_test = model.predict(X_AE_without_Seta_pp_test, bandwidth='nearest_neighbors_isotropic', n_neighbors=n_neighbors)

    NRMSE_test = analysis.normalized_root_mean_squared_error(OUTPUT_test[:,0], OUTPUT_predicted_test[:,0], norm='std')
    NRMSE_Seta_1_without.append(NRMSE_test)

    NRMSE_test = analysis.normalized_root_mean_squared_error(OUTPUT_test[:,1], OUTPUT_predicted_test[:,1], norm='std')
    NRMSE_Seta_2_without.append(NRMSE_test)

    R2_test = analysis.coefficient_of_determination(OUTPUT_test[:,0], OUTPUT_predicted_test[:,0])
    R2_Seta_1_without.append(R2_test)

    R2_test = analysis.coefficient_of_determination(OUTPUT_test[:,1], OUTPUT_predicted_test[:,1])
    R2_Seta_2_without.append(R2_test)
    
    total_toc = time.perf_counter()
    
    print(f'\nTotal time: {(total_toc - total_tic)/60:0.1f} minutes.')

np.savetxt('' + data_tag + '-NRMSE-Seta-1-without-Seta-' + str(n_components) + 'D-LDM.csv', np.array(NRMSE_Seta_1_without), delimiter=',', fmt='%.16e')
np.savetxt('' + data_tag + '-NRMSE-Seta-2-without-Seta-' + str(n_components) + 'D-LDM.csv', np.array(NRMSE_Seta_2_without), delimiter=',', fmt='%.16e')

np.savetxt('' + data_tag + '-R2-Seta-1-without-Seta-' + str(n_components) + 'D-LDM.csv', np.array(R2_Seta_1_without), delimiter=',', fmt='%.16e')
np.savetxt('' + data_tag + '-R2-Seta-2-without-Seta-' + str(n_components) + 'D-LDM.csv', np.array(R2_Seta_2_without), delimiter=',', fmt='%.16e')

# With Seta: - - - - - - - - - - - - - - - - - - - - - - - - -

NRMSE_Seta_1_with = []
NRMSE_Seta_2_with = []

R2_Seta_1_with = []
R2_Seta_2_with = []

for random_seed in random_seeds_list:
    
    print(random_seed)
    
    total_tic = time.perf_counter()

    basis_with_Seta = pd.read_csv('../results/QoIAwareProjection-basis-with-Seta-' + data_tag + '-' + case_name + '-' + architecture_with + '-random-seed-' + str(random_seed) + '.csv', sep = ',', header=None).to_numpy()
    
    (X_CS, centers, scales) = preprocess.center_scale(state_space, scaling='0to1')
    S_CS = state_space_sources / scales

    X_AE_with_Seta = np.dot(X_CS, basis_with_Seta)
    S_AE_with_Seta = np.dot(S_CS, basis_with_Seta)

    (X_AE_with_Seta_pp, centers_X_AE, scales_X_AE) = preprocess.center_scale(X_AE_with_Seta, '0to1')

    sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=random_seed, verbose=False)
    (idx_train, idx_test) = sample_random.random(train_perc)

    X_AE_with_Seta_pp_train = X_AE_with_Seta_pp[idx_train,:]
    OUTPUT_train = S_AE_with_Seta[idx_train,:]
    X_AE_with_Seta_pp_test = X_AE_with_Seta_pp[idx_test,:]
    OUTPUT_test = S_AE_with_Seta[idx_test,:]

    model = analysis.KReg(X_AE_with_Seta_pp_train, OUTPUT_train)

    OUTPUT_predicted_test = model.predict(X_AE_with_Seta_pp_test, bandwidth='nearest_neighbors_isotropic', n_neighbors=n_neighbors)

    NRMSE_test = analysis.normalized_root_mean_squared_error(OUTPUT_test[:,0], OUTPUT_predicted_test[:,0], norm='std')
    NRMSE_Seta_1_with.append(NRMSE_test)

    NRMSE_test = analysis.normalized_root_mean_squared_error(OUTPUT_test[:,1], OUTPUT_predicted_test[:,1], norm='std')
    NRMSE_Seta_2_with.append(NRMSE_test)

    R2_test = analysis.coefficient_of_determination(OUTPUT_test[:,0], OUTPUT_predicted_test[:,0])
    R2_Seta_1_with.append(R2_test)

    R2_test = analysis.coefficient_of_determination(OUTPUT_test[:,1], OUTPUT_predicted_test[:,1])
    R2_Seta_2_with.append(R2_test)
    
    total_toc = time.perf_counter()
    
    print(f'\nTotal time: {(total_toc - total_tic)/60:0.1f} minutes.')

np.savetxt('' + data_tag + '-NRMSE-Seta-1-with-Seta-' + str(n_components) + 'D-LDM.csv', np.array(NRMSE_Seta_1_with), delimiter=',', fmt='%.16e')
np.savetxt('' + data_tag + '-NRMSE-Seta-2-with-Seta-' + str(n_components) + 'D-LDM.csv', np.array(NRMSE_Seta_2_with), delimiter=',', fmt='%.16e')

np.savetxt('' + data_tag + '-R2-Seta-1-with-Seta-' + str(n_components) + 'D-LDM.csv', np.array(R2_Seta_1_with), delimiter=',', fmt='%.16e')
np.savetxt('' + data_tag + '-R2-Seta-2-with-Seta-' + str(n_components) + 'D-LDM.csv', np.array(R2_Seta_2_with), delimiter=',', fmt='%.16e')