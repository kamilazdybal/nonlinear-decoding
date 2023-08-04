import numpy as np
import time
from PCAfold import preprocess
from PCAfold import analysis

angle_list = [i for i in range(0,180)]

n_points = 2000
random_seed = 100
np.random.seed(seed=10)
mean = [0,0]
covariance = [[2.5, 0.52], [0.52, 0.2]]
x_noise, y_noise = np.random.multivariate_normal(mean, covariance, n_points).T
dataset_2D = np.column_stack((x_noise, y_noise))
dataset_2D_normalized, centers_2D, scales_2D = preprocess.center_scale(dataset_2D, scaling='-1to1')
dataset_2D_normalized = dataset_2D_normalized - np.mean(dataset_2D_normalized, axis=0)

phi_1 = dataset_2D_normalized[:,0]**2 + 3*dataset_2D_normalized[:,0]
phi_2 = dataset_2D_normalized[:,0] * dataset_2D_normalized[:,1]
phi_3 = np.sin(dataset_2D_normalized[:,0]) + np.abs(dataset_2D_normalized[:,1])
phi_4 = np.exp(-(dataset_2D_normalized[:,0]**2+dataset_2D_normalized[:,1]**2) / (2.0 * 0.3**2))

phi_2D = np.column_stack((phi_1, phi_2, phi_3, phi_4))
phi_2D_normalized, centers_phi, scales_phi = preprocess.center_scale(phi_2D, scaling='-1to1')

for i in [0,1,2,3]:

    tic = time.perf_counter()

    costs_list = []

    QoI = phi_2D_normalized[:,i]

    for angle in angle_list:

        if angle%10==0: print('Angle:\t' + str(angle) + ' degrees')

        theta = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_hat = np.array([1, 0])
        basis_vector = np.dot(rotation_matrix, x_hat)[:,None]

        data_projection = np.dot(dataset_2D_normalized, basis_vector)

        variance_data = analysis.compute_normalized_variance(data_projection,
                                                             QoI,
                                                             depvar_names=['phi'],
                                                             scale_unit_box=True,
                                                             bandwidth_values=bandwidth_values)

        cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                     penalty_function='log-sigma-over-peak',
                                                                     norm=None,
                                                                     integrate_to_peak=False)

        costs_list.append(cost)

    np.savetxt('synthetic-dataset-costs-phi-' + str(i+1) + '.csv', (costs_list), delimiter=',', fmt='%.16e')

    print('= '*30)

    toc = time.perf_counter()

    print(f'\n\nTotal time it took: {(toc - tic)/60:0.1f} minutes.')
