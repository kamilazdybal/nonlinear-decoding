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
learning_rate = 0.001
n_epochs = 20000
decoder_architecture = (6,9,10)
random_seeds_list = [i for i in range(0,100)]
bandwidth_values = np.logspace(-7, 3, 150)

########################################################################
## Compute normalized variance derivative
########################################################################

architecture_without = str(n_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_architecture]) + '-' + str(len(selected_state_variables))
architecture_with = str(n_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_architecture]) + '-' + str(len(selected_state_variables) + n_components*2)

(X_CS, centers, scales) = preprocess.center_scale(state_space, scaling='0to1')
S_CS = state_space_sources / scales

tic = time.perf_counter()

print('- '*30)

for random_seed in random_seeds_list:
    
    print('Random seed: ' + str(random_seed) + '...\n')
    
    # Without Seta: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    tic_variance_data = time.perf_counter()
    
    case_name = str(n_components) + 'D-LDM-lr-' + str(learning_rate) + '-bs-' + str(n_observations) + '-n-epochs-' + str(n_epochs) + '-architecture-' + architecture_without

    basis_without_file = '../results/QoIAwareProjection-basis-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv'
    variance_data_without_file = '../results/QoIAwareProjection-VarianceData-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.pkl'
    
    if not os.path.exists(variance_data_without_file):
 
        if os.path.exists(basis_without_file):
        
            print('Computing without Seta...')

            basis_without = pd.read_csv(basis_without_file, sep = ',', header=None).to_numpy()
            
            X_AE = np.dot(X_CS, basis_without)
            S_AE = np.dot(S_CS, basis_without)

            power_transform_PC_sources = S_AE + 10**(-4)
            power_transform_PC_sources = np.sign(power_transform_PC_sources) * np.sqrt(np.abs(power_transform_PC_sources))

            depvar_names = ['Seta' + str(i+1) for i in range(0,n_components)] + ['~Seta' + str(i+1) for i in range(0,n_components)] + list(state_space_names[selected_state_variables])

            depvars = np.hstack((S_AE, power_transform_PC_sources, state_space[:,selected_state_variables]))

            variance_data = analysis.compute_normalized_variance(X_AE,
                                                                 depvars,
                                                                 depvar_names=depvar_names,
                                                                 scale_unit_box=True,
                                                                 bandwidth_values=bandwidth_values)

            pickle.dump(variance_data, open(variance_data_without_file, "wb" ))

            print('VarianceData computed and saved!')

            toc_variance_data = time.perf_counter()

            print(f'VarianceData computation time: {(toc_variance_data - tic_variance_data)/60:0.1f} minutes.')

        else:
            print('Basis without Seta for random seed ' + str(random_seed) + ' not computed yet!')
    else:
        print('VarianceData without Seta for random seed ' + str(random_seed) + ' is already computed, moving on...')
           
    print('')
            
    # With Seta: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    tic_variance_data = time.perf_counter()
    
    case_name = str(n_components) + 'D-LDM-lr-' + str(learning_rate) + '-bs-' + str(n_observations) + '-n-epochs-' + str(n_epochs) + '-architecture-' + architecture_with
    
    basis_with_file = '../results/QoIAwareProjection-basis-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv'
    variance_data_with_file = '../results/QoIAwareProjection-VarianceData-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.pkl'
    
    if not os.path.exists(variance_data_with_file):
    
        if os.path.exists(basis_with_file):
            
            print('Computing with Seta...')
    
            basis_with = pd.read_csv(basis_with_file, sep = ',', header=None).to_numpy()
            
            X_AE = np.dot(X_CS, basis_with)
            S_AE = np.dot(S_CS, basis_with)

            power_transform_PC_sources = S_AE + 10**(-4)
            power_transform_PC_sources = np.sign(power_transform_PC_sources) * np.sqrt(np.abs(power_transform_PC_sources))

            depvar_names = ['Seta' + str(i+1) for i in range(0,n_components)] + ['~Seta' + str(i+1) for i in range(0,n_components)] + list(state_space_names[selected_state_variables])

            depvars = np.hstack((S_AE, power_transform_PC_sources, state_space[:,selected_state_variables]))

            variance_data = analysis.compute_normalized_variance(X_AE,
                                                                 depvars,
                                                                 depvar_names=depvar_names,
                                                                 scale_unit_box=True,
                                                                 bandwidth_values=bandwidth_values)

            pickle.dump(variance_data, open(variance_data_with_file, "wb" ))

            print('VarianceData computed and saved!')

            toc_variance_data = time.perf_counter()

            print(f'VarianceData computation time: {(toc_variance_data - tic_variance_data)/60:0.1f} minutes.')
            
        else:
            print('Basis with Seta for random seed ' + str(random_seed) + ' not computed yet!')
    else:
        print('VarianceData with Seta for random seed ' + str(random_seed) + ' is already computed, moving on...')

    print('- '*30)
        
toc = time.perf_counter()

print(f'\n\n\tTotal time it took: {(toc - tic)/60:0.1f} minutes.')
