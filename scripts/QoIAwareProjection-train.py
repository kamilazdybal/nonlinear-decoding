import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras import __version__ as keras_version
from PCAfold import preprocess
from PCAfold import QoIAwareProjection
from PCAfold import __version__ as PCAfold_version

print('numpy==' + np.__version__)
print('pandas==' + pd.__version__)
print('tensorflow==' + tf.__version__)
print('keras==' + keras_version)
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

random_seeds_list = [i for i in range(0,100)]

n_components = 2
learning_rate = 0.001
n_epochs = 20000
decoder_interior_architecture = (6,9,10)
activation_decoder = 'tanh'
batch_size = n_observations
hold_initialization = None
hold_weights = None
transformed_projection_dependent_outputs = 'signed-square-root'
loss = 'MSE'
optimizer = 'Adam'
validation_perc = 10

save_intermediate_bases = []

########################################################################
## Run QoI-aware encoder-decoder projection with and without Seta
########################################################################

(X_CS, centers, scales) = preprocess.center_scale(state_space, scaling='0to1')
S_CS = state_space_sources / scales

tic = time.perf_counter()

for random_seed in random_seeds_list:
    
    print('Random seed: ' + str(random_seed) + '...\n')

    # Without Seta: - - - - - - - - - - - - - - - - - - - - - - - - -

    projection_without_Seta = QoIAwareProjection(X_CS,
                                               n_components,
                                               projection_independent_outputs=X_CS[:,selected_state_variables],
                                               activation_decoder=activation_decoder,
                                               decoder_interior_architecture=decoder_interior_architecture,
                                               encoder_weights_init=None,
                                               decoder_weights_init=None,
                                               hold_initialization=hold_initialization,
                                               hold_weights=hold_weights,
                                               loss=loss,
                                               optimizer=optimizer,
                                               batch_size=batch_size,
                                               n_epochs=n_epochs,
                                               learning_rate=learning_rate,
                                               validation_perc=validation_perc,
                                               random_seed=random_seed,
                                               verbose=True)
    
    case_name = str(n_components) + 'D-LDM-lr-' + str(learning_rate) + '-bs-' + str(batch_size) + '-n-epochs-' + str(n_epochs) + '-architecture-' + projection_without_Seta.architecture
   
    projection_without_Seta.train()

    WITHOUT_SETA_training_losses_across_epochs = projection_without_Seta.training_loss
    WITHOUT_SETA_validation_losses_across_epochs = projection_without_Seta.validation_loss
    WITHOUT_SETA_basis = projection_without_Seta.get_best_basis(method='min-training-loss')

    np.savetxt('../results/QoIAwareProjection-basis-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITHOUT_SETA_basis), delimiter=',', fmt='%.16e')
    np.savetxt('../results/QoIAwareProjection-MSE-training-losses-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITHOUT_SETA_training_losses_across_epochs), delimiter=',', fmt='%.16e')
    np.savetxt('../results/QoIAwareProjection-MSE-validation-losses-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITHOUT_SETA_validation_losses_across_epochs), delimiter=',', fmt='%.16e')
    
    if len(save_intermediate_bases) != 0:
        for e in save_intermediate_bases:
            np.savetxt('../results/QoIAwareProjection-intermediate-basis-at-epoch-' + str(e) + '-without-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (projection_without_Seta.bases_across_epochs[e]), delimiter=',', fmt='%.16e')

    # With Seta: - - - - - - - - - - - - - - - - - - - - - - - - -

    projection_with_Seta = QoIAwareProjection(X_CS,
                                               n_components,
                                               projection_independent_outputs=X_CS[:,selected_state_variables],
                                               projection_dependent_outputs=S_CS,
                                               activation_decoder=activation_decoder,
                                               decoder_interior_architecture=decoder_interior_architecture,
                                               encoder_weights_init=None,
                                               decoder_weights_init=None,
                                               hold_initialization=hold_initialization,
                                               hold_weights=hold_weights,
                                               transformed_projection_dependent_outputs=transformed_projection_dependent_outputs,
                                               loss=loss,
                                               optimizer=optimizer,
                                               batch_size=batch_size,
                                               n_epochs=n_epochs,
                                               learning_rate=learning_rate,
                                               validation_perc=validation_perc,
                                               random_seed=random_seed,
                                               verbose=True)
        
    case_name = str(n_components) + 'D-LDM-lr-' + str(learning_rate) + '-bs-' + str(batch_size) + '-n-epochs-' + str(n_epochs) + '-architecture-' + projection_with_Seta.architecture
    
    projection_with_Seta.train()

    WITH_SETA_training_losses_across_epochs = projection_with_Seta.training_loss
    WITH_SETA_validation_losses_across_epochs = projection_with_Seta.validation_loss
    WITH_SETA_basis = projection_with_Seta.get_best_basis(method='min-training-loss')

    np.savetxt('../results/QoIAwareProjection-basis-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITH_SETA_basis), delimiter=',', fmt='%.16e')
    np.savetxt('../results/QoIAwareProjection-MSE-training-losses-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITH_SETA_training_losses_across_epochs), delimiter=',', fmt='%.16e')
    np.savetxt('../results/QoIAwareProjection-MSE-validation-losses-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (WITH_SETA_validation_losses_across_epochs), delimiter=',', fmt='%.16e')

    if len(save_intermediate_bases) != 0:
        for e in save_intermediate_bases:
            np.savetxt('../results/QoIAwareProjection-intermediate-basis-at-epoch-' + str(e) + '-with-Seta-' + data_tag + '-' + case_name + '-random-seed-' + str(random_seed) + '.csv', (projection_with_Seta.bases_across_epochs[e]), delimiter=',', fmt='%.16e')
    
toc = time.perf_counter()

print(f'\n\n\tTotal time it took: {(toc - tic)/60:0.1f} minutes.')
