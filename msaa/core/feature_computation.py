import csv
import os

import pandas as pd
import numpy as np
import pflacco.classical_ela_features as pflacco_ela
from .utilities import show_current_date_time


def compute_ela(X, y, min_y, max_y, lower_bound, upper_bound):
    # y_rescale = (max(y) - y) / (max(y) - min(y) + 1e-20)
    y_rescale = (max_y - y) / (max_y - min_y + 1e-30)
    # Calculate ELA_20-2-2024 features
    ela_meta = pflacco_ela.calculate_ela_meta(X, y_rescale)
    ela_distr = pflacco_ela.calculate_ela_distribution(X, y_rescale)
    ela_level = pflacco_ela.calculate_ela_level(X, y_rescale)
    pca = pflacco_ela.calculate_pca(X, y_rescale)
    limo = pflacco_ela.calculate_limo(X, y_rescale, lower_bound, upper_bound)
    nbc = pflacco_ela.calculate_nbc(X, y_rescale)
    disp = pflacco_ela.calculate_dispersion(X, y_rescale)
    ic = pflacco_ela.calculate_information_content(X, y_rescale, seed=100)
    ela_ = {**ela_meta, **ela_distr, **ela_level, **pca, **limo, **nbc, **disp, **ic}
    df_ela = pd.DataFrame([ela_])
    return df_ela


def step3_ela_feature(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type = temp

    ###################### PREPARE ELA FEATURE FILES ######################

    # import pandas as pd
    # import pflacco.classical_ela_features as pflacco_ela
    #
    # warnings.filterwarnings('ignore', category=RuntimeWarning)
    # warnings.filterwarnings('ignore', category=UserWarning)

    # CREATE A DIRECTORY
    required_dir = 'ELA'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _problem in prob_type:
        # CREATE A DIRECTORY
        required_results_dir = 'ELA/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    for _problem in prob_type:
        for _algo in ng_algs:
            for _dim in dims:
                for _func in fids:

                    result_exist_flag = os.path.exists(
                        './ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_ela.csv')

                    # Skip or not?
                    if force_replace_flag[0] or (not result_exist_flag):

                        target_csv_file = './CSV/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(
                            _func) + '_data.csv'

                        if not os.path.isfile(target_csv_file):
                            print('NOT FOUND: ', target_csv_file, end='\t')

                        else:
                            # print(target_csv_file, end='\t')

                            # CSV file to which the data is written
                            output_file = './ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(
                                _func) + '_ela.csv'

                            # Open the CSV file for writing
                            with open(output_file, 'a', newline='') as csvfile:
                                csvwriter = csv.writer(csvfile)

                                # # Optional: Write headers to the CSV file
                                # headers = ['algo', 'dim', 'fid', 'iid', 'run', 'eval', 'final_cost',
                                #            'current_cost']  # Adjust based on your data structure
                                # for d in range(_dim):
                                #     headers.append('x' + str(d + 1))
                                #
                                # # print(headers)
                                # csvwriter.writerow(headers)

                                # Retrieve observations
                                ela_x_input = []
                                ela_y_input = []
                                # final_y_best = []

                                iid_ = []
                                runid_ = []

                                # Iterate over each file in the file list
                                # Open the current raw data file for reading
                                with open(target_csv_file, 'r') as file:
                                    # Read the file line by line
                                    for line in file:
                                        # Assuming each line in the raw data file is comma-separated
                                        # Split the line into a list
                                        pre_screen = line.strip().split(',')
                                        # print(pre_screen)

                                        # Optional: process or transform data here before writing to CSV
                                        if pre_screen[0] == 'algo' and pre_screen[1] == 'dim':
                                            pass
                                        else:
                                            ela_x_input_component = []
                                            for d in range(_dim):
                                                ela_x_input_component.append(float(pre_screen[d + 8]))

                                            ela_x_input.append(ela_x_input_component)

                                            # It is impossible that current cost is better than final cost
                                            if float(pre_screen[7]) <= 0.0:
                                                ela_y_input.append(float(1e-10))
                                            else:
                                                ela_y_input.append(float(pre_screen[7]))

                                            # if len(final_y_best) < 1:
                                            #     final_y_best.append(float(pre_screen[6]))

                                            iid_.append(int(pre_screen[3]))
                                            runid_.append(int(pre_screen[4]))

                                            # if np.isnan(float(pre_screen[7])):
                                            #     print(float(pre_screen[7]))

                                ela_x_input_np = np.array(ela_x_input)
                                ela_y_input_np = np.array(ela_y_input)

                                min_y = ela_y_input_np.min()
                                max_y = ela_y_input_np.max()

                                final_y_best = ela_y_input_np.min()

                                # print(ela_y_input_np.shape)
                                # print(ela_y_input_np.min())
                                #
                                # exit()

                                # print(min_y)
                                # print(max_y)

                                # print(ela_x_input_np)
                                # print(ela_y_input_np)

                                # print('extracted', end='\t')

                                begin_ind = 0
                                end_ind = 99
                                batch_size = 100

                                dataset_instance_count = 0
                                iteration_done = 0
                                iteration_togo = _dim * bfacs[0]

                                while end_ind < ela_y_input_np.shape[0]:

                                    x_tmp = []
                                    y_tmp = []

                                    if iteration_done >= _dim * bfacs[0]:
                                        iteration_done = 0

                                    if iteration_togo <= 0:
                                        iteration_togo = _dim * bfacs[0]

                                    iteration_done = iteration_done + batch_size
                                    iteration_togo = iteration_togo - batch_size

                                    for i in range(begin_ind, end_ind + 1):
                                        x_tmp.append(ela_x_input_np[i, :].tolist())
                                        y_tmp.append(ela_y_input_np[i].tolist())
                                    # print(x_tmp)
                                    # print(y_tmp)

                                    # ela_features = compute_ela(x_tmp, y_tmp, min_y, max_y)

                                    try:

                                        lb = None
                                        ub = None
                                        if _problem == 'BBOB':
                                            lb = -5
                                            ub = 5
                                        elif _problem == 'PBO':
                                            lb = 0
                                            ub = 1

                                        ela_features = compute_ela(x_tmp, y_tmp, min_y, max_y, lb, ub)
                                        # print(ela_features.columns.tolist())

                                        if begin_ind == 0:
                                            # Optional: Write headers to the CSV file
                                            headers = []  # Adjust based on your data structure
                                            headers.append('algo')
                                            headers.append('dim')
                                            headers.append('func_id')
                                            headers.append('ins_id')
                                            headers.append('run_id')
                                            # headers.append('interval')
                                            for each_feature in ela_features.columns.tolist():
                                                headers.append(each_feature)
                                            headers.append('iteration_done')
                                            headers.append('iteration_togo')
                                            headers.append('current_best')
                                            headers.append('final_best')

                                            # print(headers)
                                            csvwriter.writerow(headers)

                                            # Extract values of features
                                            ela_data = np.array(ela_features.iloc[0, :]).tolist()
                                            data = []
                                            data.append(_algo)
                                            data.append(_dim)
                                            data.append(_func)
                                            data.append(iid_[begin_ind])
                                            data.append(runid_[begin_ind])
                                            # data.append(str(begin_ind+1) + '_' + str(end_ind+1))
                                            for value in ela_data:
                                                data.append(value)
                                            data.append(iteration_done)
                                            data.append(iteration_togo)
                                            data.append(float(ela_y_input_np[begin_ind: end_ind].min()))
                                            data.append(final_y_best)
                                            csvwriter.writerow(data)
                                        else:
                                            # Extract values of features
                                            ela_data = np.array(ela_features.iloc[0, :]).tolist()
                                            data = []
                                            data.append(_algo)
                                            data.append(_dim)
                                            data.append(_func)
                                            data.append(iid_[begin_ind])
                                            data.append(runid_[begin_ind])
                                            # data.append(str(begin_ind + 1) + '_' + str(end_ind + 1))
                                            for value in ela_data:
                                                data.append(value)
                                            data.append(iteration_done)
                                            data.append(iteration_togo)
                                            data.append(float(ela_y_input_np[begin_ind: end_ind].min()))
                                            data.append(final_y_best)
                                            csvwriter.writerow(data)

                                        # print('B: ', begin_ind, 'E: ', end_ind)

                                        dataset_instance_count = dataset_instance_count + 1

                                    except Exception as e:
                                        # print(e)
                                        pass

                                    begin_ind = begin_ind + batch_size
                                    end_ind = end_ind + batch_size

                                # print('calculated ', str(dataset_instance_count), ' instances', end='\t')

                        # print('completed!')

                    # else:
                    #     print('SKIP: ' + './ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_ela.csv')

    print(show_current_date_time() + ' ' + 'COMPLETED 3/4: ELA features were extracted')


def step3_non_ela_feature(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type = temp

    ###################### PREPARE FEATURE FILES ######################

    # import pandas as pd
    # import pflacco.classical_ela_features as pflacco_ela
    #
    # warnings.filterwarnings('ignore', category=RuntimeWarning)
    # warnings.filterwarnings('ignore', category=UserWarning)

    # CREATE A DIRECTORY
    required_dir = 'NON_ELA'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _problem in prob_type:
        # CREATE A DIRECTORY
        required_results_dir = 'NON_ELA/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    for _problem in prob_type:
        for _algo in ng_algs:
            for _dim in dims:
                for _func in fids:

                    result_exist_flag = os.path.exists(
                        './NON_ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_non_ela.csv')

                    # Skip or not?
                    if force_replace_flag[0] or (not result_exist_flag):

                        target_csv_file = './CSV/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(
                            _func) + '_data.csv'

                        if not os.path.isfile(target_csv_file):
                            print('NOT FOUND: ', target_csv_file, end='\t')

                        else:
                            # print(target_csv_file, end='\t')

                            # CSV file to which the data is written
                            output_file = './NON_ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(
                                _func) + '_non_ela.csv'

                            # Open the CSV file for writing
                            with open(output_file, 'a', newline='') as csvfile:
                                csvwriter = csv.writer(csvfile)

                                # Retrieve observations
                                raw_current_input = []

                                iid_ = []
                                runid_ = []

                                # Iterate over each file in the file list
                                # Open the current raw data file for reading
                                with open(target_csv_file, 'r') as file:
                                    # Read the file line by line
                                    for line in file:
                                        # Assuming each line in the raw data file is comma-separated
                                        # Split the line into a list
                                        pre_screen = line.strip().split(',')
                                        # print(pre_screen)

                                        # Optional: process or transform data here before writing to CSV
                                        if pre_screen[0] == 'algo' and pre_screen[1] == 'dim':
                                            pass
                                        else:
                                            raw_current_input.append(float(pre_screen[7]))
                                            iid_.append(int(pre_screen[3]))
                                            runid_.append(int(pre_screen[4]))

                                raw_current_input_np = np.array(raw_current_input)
                                raw_current_input_np_desc = np.sort(raw_current_input_np)

                                begin_ind = 0
                                end_ind = 99
                                batch_size = 100
                                step_size = 1

                                dataset_instance_count = 0
                                iteration_done = 0
                                iteration_togo = _dim * bfacs[0]

                                # input 100 points to predict a next point
                                while end_ind + 1 < raw_current_input_np_desc.shape[0]:

                                    x_tmp = []
                                    y_tmp = []

                                    if iteration_done >= _dim * bfacs[0]:
                                        iteration_done = 0

                                    if iteration_togo <= 0:
                                        iteration_togo = _dim * bfacs[0]

                                    iteration_done = iteration_done + batch_size
                                    iteration_togo = iteration_togo - batch_size

                                    for i in range(begin_ind, end_ind + 1):
                                        x_tmp.append(raw_current_input_np_desc[i].tolist())
                                    y_tmp.append(raw_current_input_np_desc[end_ind + 1].tolist())

                                    try:

                                        if begin_ind == 0:
                                            # Optional: Write headers to the CSV file
                                            headers = []  # Adjust based on your data structure
                                            headers.append('algo')
                                            headers.append('dim')
                                            headers.append('func_id')
                                            headers.append('ins_id')
                                            headers.append('run_id')
                                            # headers.append('interval')
                                            # headers.append('iteration_done')
                                            # headers.append('iteration_togo')
                                            for feature_ind in range(1, batch_size + 1):
                                                headers.append('x' + str(feature_ind))
                                            headers.append('target')
                                            # print(headers)
                                            csvwriter.writerow(headers)

                                            # Extract values of features
                                            data = []
                                            data.append(_algo)
                                            data.append(_dim)
                                            data.append(_func)
                                            data.append(iid_[begin_ind])
                                            data.append(runid_[begin_ind])
                                            # data.append(str(begin_ind+1) + '_' + str(end_ind+1))
                                            # data.append(iteration_done)
                                            # data.append(iteration_togo)
                                            for value in x_tmp:
                                                data.append(value)
                                            data.append(y_tmp[0])
                                            csvwriter.writerow(data)
                                        else:
                                            # Extract values of features
                                            data = []
                                            data.append(_algo)
                                            data.append(_dim)
                                            data.append(_func)
                                            data.append(iid_[begin_ind])
                                            data.append(runid_[begin_ind])
                                            # data.append(str(begin_ind+1) + '_' + str(end_ind+1))
                                            # data.append(iteration_done)
                                            # data.append(iteration_togo)
                                            for value in x_tmp:
                                                data.append(value)
                                            data.append(y_tmp[0])
                                            csvwriter.writerow(data)

                                        # dataset_instance_count = dataset_instance_count + 1

                                    except Exception as e:
                                        # print(e)
                                        pass

                                    begin_ind = begin_ind + step_size
                                    end_ind = end_ind + step_size

                                # print('calculated ', str(dataset_instance_count), ' instances', end='\t')

                        # print('completed!')

                    # else:
                    #     print('SKIP: ' + './ELA/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_ela.csv')

    print(show_current_date_time() + ' ' + 'COMPLETED 3/4: Features were prepared')
