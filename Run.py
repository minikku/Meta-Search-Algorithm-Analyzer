# -*- coding: utf-8 -*-
# IMPORT BASE LIBRARIES

import os

# Set environment variables for thread control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import ioh
from itertools import product
from multiprocessing import Pool, freeze_support
from decimal import Decimal, getcontext
import warnings
import csv
import json
import time
from datetime import datetime

# IMPORT NEVERGRAD ALGORITHMS

import nevergrad as ng
from nevergrad.optimization.optimizerlib import CMA, DE, PSO
import cma.evolution_strategy

# IMPORT CUSTOM ALGORITHMS

# from Algorithms.DCS_ng import DCS
from Algorithms.DCS_ng_v2 import DCS

# -------------------------------------

import pandas as pd
import pflacco.classical_ela_features as pflacco_ela

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------

# !pip install --upgrade scikit-learn

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pickle


# getcontext().prec = 16  # Set precision to 16 significant digits

###################### CREATE FEATURE LANDSCAPE ######################

# Function to run tasks in parallel
def run_parallel_function(runFunction, arguments, minPoolSize: int):
    arguments = list(arguments)
    p = Pool(min(minPoolSize, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()

    print(showCurrentDateTime() + ' ' + 'COMPLETED 1/4: Raw results were obtained')

    return results


# Algorithm evaluator
class AlgorithmEvaluator:
    def __init__(self, optimizer, bfac, _problem):
        self.alg = optimizer
        self.bfac = bfac
        self.problem_type = _problem

    def __call__(self, func, seed):
        np.random.seed(int(seed))
        warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)

        if self.problem_type == 'BBOB':
            lb = func.bounds.lb[0]
            ub = func.bounds.ub[0]

            parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(lb, ub)
            optimizer = eval(f"{self.alg}")(parametrization=parametrization, problem_dim=func.meta_data.n_variables,
                                            problem_bounds=list([lb, ub]),
                                            optimization_type=func.meta_data.optimization_type.name,
                                            budget=int(self.bfac * func.meta_data.n_variables))
            optimizer.minimize(func)

        elif self.problem_type == 'PBO':
            lb = func.bounds.lb[0]
            ub = func.bounds.ub[0]

            # parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(lb, ub)  # Continuous
            parametrization = ng.p.TransitionChoice(range(lb, ub + 1), repetitions=func.meta_data.n_variables)  # Discrete
            # parametrization = ng.p.Choice(list(range(lb, ub + 1)), repetitions=func.meta_data.n_variables)  # Discrete
            # parametrization.function = func
            optimizer = eval(f"{self.alg}")(parametrization=parametrization, problem_dim=func.meta_data.n_variables,
                                            problem_bounds=list([lb, ub]),
                                            optimization_type=func.meta_data.optimization_type.name,
                                            budget=int(self.bfac * func.meta_data.n_variables))
            optimizer.minimize(func)
            # optimizer.provide_recommendation()


# Function to run the optimizer
def run_optimizer(temp):
    algname, fid, iid, dim, bfac, force_replace_flag, rep_num, _problem = temp
    # print(algname, 'F', fid, ' I', iid, 'D', dim)

    algorithm = AlgorithmEvaluator(algname, bfac, _problem)
    max_budget = bfac * dim

    func = None
    if _problem == 'BBOB':
        func = ioh.get_problem(fid=fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.BBOB)
    elif _problem == 'PBO':
        func = ioh.get_problem(fid=fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.PBO)

    fname = func.meta_data.name
    result_exist_flag = os.path.exists(
        './Results/' + _problem + '/' + algname + '_' + str(max_budget) + '_F' + str(fid) + '_I' + str(
            iid) + '_' + str(
            dim) + 'D/IOHprofiler_f' + str(fid) + '_' + fname + '.json')

    # Skip or not?
    if force_replace_flag or not result_exist_flag:
        logger = ioh.logger.Analyzer(root=f"./Results/{_problem}/",
                                     folder_name=f"{algname}_{max_budget}_F{fid}_I{iid}_{dim}D",
                                     algorithm_name=f"{algname}_{max_budget}",
                                     store_positions=True, triggers=[ioh.logger.trigger.ALWAYS])

        func.attach_logger(logger)
        for rep in range(rep_num):
            algorithm(func, rep)
            func.reset()
        logger.close()

    # else:
    #     print('SKIP: ' + './Results/' + algname + '_' + str(max_budget) + '_F' + str(fid) + '_I' + str(iid) + '_' + str(
    #         dim) + 'D/IOHprofiler_f' + str(fid) + '_' + fname + '.json')


def step2(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type = temp

    ###################### PREPARE CSV FILES ######################

    # CREATE A DIRECTORY
    required_dir = 'CSV'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _problem in prob_type:
        # CREATE A DIRECTORY
        required_results_dir = 'CSV/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    for _problem in prob_type:
        for _algo in ng_algs:
            for _dim in dims:
                for _func in fids:

                    result_exist_flag = os.path.exists(
                        './CSV/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_data.csv')

                    # Skip or not?
                    if force_replace_flag[0] or (not result_exist_flag):

                        for _iid in iids:

                            max_budget = bfacs[0] * _dim
                            base_folder = './Results/' + _problem + '/' + _algo + '_' + str(max_budget) + '_F' + str(
                                _func) + '_I' + str(
                                _iid) + '_' + str(_dim) + 'D'

                            if not os.path.isdir(base_folder):
                                print('NOT FOUND: ', base_folder)

                            else:
                                # print(base_folder, end='\t')
                                base_entries = os.listdir(base_folder)
                                sub_entry = os.listdir(base_folder + '/' + base_entries[0])
                                # print(sub_entry[0])
                                # print(base_entries[1])

                                # CSV file to which the data is written
                                output_file = './CSV/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(
                                    _func) + '_data.csv'

                                # Open the CSV file for writing
                                with open(output_file, 'a', newline='') as csvfile:
                                    csvwriter = csv.writer(csvfile)

                                    # Optional: Write headers to the CSV file
                                    headers = ['algo', 'dim', 'fid', 'iid', 'run', 'eval', 'final_cost',
                                               'current_cost']  # Adjust based on your data structure
                                    for d in range(_dim):
                                        headers.append('x' + str(d + 1))

                                    # print(headers)
                                    csvwriter.writerow(headers)

                                    run_itr = 0

                                    with open(base_folder + '/' + base_entries[1], 'r') as json_file:
                                        # Read json summary file
                                        json_data = json.load(json_file)
                                        # print(json_data)

                                        # Iterate over each file in the file list
                                        # Open the current raw data file for reading
                                        with open(base_folder + '/' + base_entries[0] + '/' + sub_entry[0],
                                                  'r') as file:
                                            # Read the file line by line
                                            for line in file:
                                                # Assuming each line in the raw data file is comma-separated
                                                # Split the line into a list
                                                pre_screen = line.strip().split(' ')

                                                # Optional: process or transform data here before writing to CSV
                                                if pre_screen[0] == 'evaluations':
                                                    run_itr = run_itr + 1
                                                else:
                                                    # print(_func, _iid)
                                                    best_y = json_data['scenarios'][0]['runs'][run_itr - 1]['best']['y']
                                                    row_data = [_algo, str(_dim), str(_func), str(_iid), run_itr,
                                                                pre_screen[0],
                                                                best_y, pre_screen[1]]
                                                    # row_data = f"{_algo},{_dim},{_func},{_iid},{run_itr},{pre_screen[0]},{best_y},{pre_screen[1]}"
                                                    for d in range(_dim):
                                                        row_data.append(pre_screen[d + 2])
                                                        # row_data = f"{row_data},{pre_screen[d+2]}"

                                                    # Write the data to the CSV file
                                                    csvwriter.writerow(row_data)
                    # else:
                    #     print('SKIP: ' + './CSV/' + _problem + '/' + _algo + '_' + str(_dim) + 'D_F' + str(_func) + '_data.csv')

    print(showCurrentDateTime() + ' ' + 'COMPLETED 2/4: Raw results were formatted')


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

    print(showCurrentDateTime() + ' ' + 'COMPLETED 3/4: ELA features were extracted')


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

    print(showCurrentDateTime() + ' ' + 'COMPLETED 3/4: Features were prepared')


def step4_ela(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type = temp

    ###################### CREATE META-MODELS ######################

    # # !pip install --upgrade scikit-learn
    #
    # from sklearn.model_selection import GroupKFold
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error, r2_score
    # import matplotlib.pyplot as plt
    # from sklearn.pipeline import Pipeline
    # from sklearn.metrics import r2_score, root_mean_squared_error
    #
    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer, SimpleImputer
    # from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # import random
    # import pickle

    # CREATE A DIRECTORY
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    # CREATE A DIRECTORY
    required_dir = 'Models/RF'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _problem in prob_type:
        # CREATE A DIRECTORY
        required_results_dir = 'Models/RF/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    # SET EXPERIMENT SEED
    random.seed(42)

    # ----------------------- PREPARE ELA DATA -----------------------

    prob_list = prob_type
    algo_list = ng_algs
    dim_list = dims
    func_list = fids

    # all_files = []

    for prob in prob_list:

        # Defining parameters
        agg_algorithms = algo_list
        iteration_limit = 2000
        how_many_func = 0
        agg_data = []

        for algo in algo_list:
            for dim in dim_list:

                all_files = []

                for func in func_list:
                    how_many_func = how_many_func + 1

                    file_path = './ELA/' + prob + '/' + algo + '_' + str(dim) + 'D_F' + str(func) + '_ela.csv'
                    # print(file_path)
                    current = pd.read_csv(file_path)
                    all_files.append(current)

                df = pd.concat(all_files)
                # print('shape of df:', df.shape)
                # print(df.dtypes)
                # print('data: ', df.shape[0])

                # Save combined to a files
                # df.to_csv(algo + '_BBOB_' + str(dim) + 'D_ELA.csv', encoding='utf-8')

                # ----------------------- PREPROCESS FEATURES -----------------------

                # Replace Inf/-Inf with NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # y = df['final_best'].to_numpy()
                # y = df['final_best']
                y = df['current_best']

                groups = df['func_id'].to_numpy()
                # groups = df['func_id']

                # df = df.drop(['final_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id'], axis=1)
                # df = df.drop(['final_best', 'algo', 'dim', 'func_id', 'run_id'], axis=1)
                # df = df.drop(['current_best', 'algo', 'dim', 'func_id', 'run_id'], axis=1)
                # df = df.drop(['current_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id'], axis=1)
                # df = df.drop(['current_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id'], axis=1)
                # df = df.drop(['current_best', 'final_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id'], axis=1) # 25-3-2024
                # df = df.drop(
                #     ['current_best', 'final_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id'],
                #     axis=1)  # 25-3-2024
                df = df.drop(
                    ['current_best', 'final_best', 'algo', 'dim', 'func_id', 'ins_id', 'run_id', 'ic.costs_runtime'],
                    axis=1)  # 1-4-2024

                # print('shape of df:', df.shape)
                # print(df.dtypes)
                # replace infinite with nan. on train set. reflect on this later

                # Select columns to scale (all except the first one)
                columns_to_scale = df.columns[1:]

                # Apply StandardScaler
                # scaler = StandardScaler()
                scaler = MinMaxScaler()
                df_scaled = scaler.fit_transform(df[columns_to_scale])

                # Create a DataFrame from the scaled data
                df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)

                # Concatenate the first column (unchanged) with the scaled DataFrame
                df_final = pd.concat([df.iloc[:, :1].reset_index(drop=True), df_scaled], axis=1)

                # print(df_final)
                df = df_final

                # df = df.replace([np.inf, -np.inf], np.nan)
                # X = df.to_numpy()
                X = df

                gkf = None
                if prob == 'BBOB':
                    gkf = GroupKFold(n_splits=24)
                elif prob == 'PBO':
                    gkf = GroupKFold(n_splits=25)

                # print(X.shape, y.shape, groups.shape)

                # ----------------------- TRAIN META-MODEL -----------------------

                model_storage = []

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    result_exist_flag = os.path.exists(
                        './Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav')

                    # Skip or not?
                    if force_replace_flag[0] or not result_exist_flag:
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        # X_train_pre, X_test = X.iloc[train_index], X.iloc[test_index]
                        # y_train_pre, y_test = y.iloc[train_index], y.iloc[test_index]
                        #
                        # y_train_0 = pd.DataFrame(y_train_pre).sort_values(by=['current_best'], ascending=False)
                        # X_train_0 = pd.DataFrame(X_train_pre.iloc[y_train_0.index])
                        # y_train = y_train_0.reset_index(drop=True)
                        # X_train = X_train_0.reset_index(drop=True)

                        # Handle NaN/Inf in folds if necessary
                        # Similar preprocessing as Ste
                        # p 1

                        # model = RandomForestRegressor(random_state=42)
                        # model = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=15, random_state=42, bootstrap=False) # 25-3-2024
                        model = RandomForestRegressor(n_estimators=50, criterion="squared_error", min_samples_split=2,
                                                      min_samples_leaf=1,
                                                      max_features=76, random_state=42, bootstrap=True)  # 25-3-2024
                        # model.fit(X_train, y_train)
                        # y_pred = model.predict(X_test)

                        # imputer = SimpleImputer(strategy='mean')
                        imputer = SimpleImputer(strategy='constant', fill_value=-1)
                        pipeline = Pipeline([('imputer', imputer), ('regressor', model)])
                        pipeline.fit(X_train, y_train)
                        # y_pred = pipeline.predict(X_test)

                        # Store trained model
                        model_storage.append(pipeline)

                    # else:
                    #     print('SKIP: ' + './Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav')

                # ----------------------- STORE TRAINED META-MODEL -----------------------

                if len(model_storage) > 0:
                    i = 0
                    for model_obj in model_storage:
                        # print(model_obj)
                        # save the model to disk
                        filename = 'Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(
                            i) + '.sav'
                        pickle.dump(model_storage[i], open(filename, 'wb'))
                        i = i + 1

                    print(
                        showCurrentDateTime() + ' ' + 'COMPLETED 4/4: Models were created (' + algo + '_' + str(
                            dim) + 'D)')

                # ----------------------- TEST META-MODEL -----------------------

                fig, axs = plt.subplots(24, figsize=(10, 60))  # Adjust the size as needed

                # Set options to display a full DataFrame
                pd.set_option('display.max_rows', None)  # No limit on the number of rows displayed
                pd.set_option('display.max_columns', None)  # No limit on the number of columns
                pd.options.display.float_format = '{:.10f}'.format

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    # print('Train ', str(i), 'Group: ', groups[train_index],)
                    # print('Test ', str(i), 'Group: ', groups[test_index],)

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # X_train, X_test_pre = X.iloc[train_index], X.iloc[test_index]
                    # y_train, y_test_pre = y.iloc[train_index], y.iloc[test_index]
                    #
                    # # print(y_test_pre)
                    #
                    # y_test_0 = pd.DataFrame(y_test_pre).sort_values(by=['current_best'], ascending=False)
                    # X_test_0 = pd.DataFrame(X_test_pre.iloc[y_test_0.index])
                    # y_test = y_test_0.reset_index(drop=True)
                    # X_test = X_test_0.reset_index(drop=True)

                    # print(y_test)

                    filename = 'Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav'
                    loaded_model = pickle.load(open(filename, 'rb'))
                    pipeline = loaded_model
                    # pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    # print(y_pred.shape)

                    # # Reverse scaling
                    # y_pred_reverted = scaler.inverse_transform(pd.concat([X_test.iloc[:, :-1].reset_index(drop=True), pd.DataFrame(y_pred)], axis=1))
                    # y_test_reverted = scaler.inverse_transform(pd.concat([X_test.iloc[:, :-1].reset_index(drop=True), pd.DataFrame(y_test)], axis=1))

                    # print(y_pred)

                    y_test_0 = pd.DataFrame(y_test).sort_values(by=['current_best'], ascending=False)
                    # y_pred_0 = pd.DataFrame(y_pred_x.iloc[y_test_0.index])
                    y_test_vis = y_test_0.reset_index(drop=True)

                    y_pred_x = pd.DataFrame(y_pred, columns=['current_best'])
                    y_pred_0 = pd.DataFrame(y_pred_x).sort_values(by=['current_best'], ascending=False)
                    y_pred_vis = y_pred_0.reset_index(drop=True)

                    # print(y_pred_vis)

                    # Plot actual values
                    perf_r2 = r2_score(y_test_vis, y_pred_vis)
                    perf_rmse = root_mean_squared_error(y_test_vis, y_pred_vis)
                    # axs[i].plot(y_test.reset_index(drop=True), label='Actual', color='blue', linestyle='-')
                    axs[i].plot(y_test_vis, label='Actual', color='blue', linestyle='dotted')
                    axs[i].plot(y_pred_vis, label='Predicted', color='red', linestyle='dotted')

                    axs[i].set_title(
                        f'{algo} {dim}D F{groups[test_index[0]]}: actual and predicted values (RMSE: {perf_rmse})')  # Corrected
                    axs[i].set_xlabel('Sample Index')  # Corrected
                    axs[i].set_ylabel('Value')  # Corrected
                    axs[i].legend()
                    axs[i].set_yscale('log')
                    # axs[i].set_ylim(bottom=lb, top=ub)

                    # Collect data for aggregated visualization
                    for ind in y_test_vis.index:
                        if ind < iteration_limit:
                            agg_data.append({
                                'algo_name': algo,
                                'dim_size': dim,
                                'func_id': groups[test_index[0]],
                                'iteration': ind + 1,
                                'actual_val': y_test_vis['current_best'][ind],
                                'predicted_val': y_pred_vis['current_best'][ind]
                            })

                plt.tight_layout()
                plt.show()

                print(
                    showCurrentDateTime() + ' ' + 'COMPLETED 4/4: Models were tested (' + algo + '_' + str(dim) + 'D)')

        # Preparing data for plotting
        agg_df = pd.DataFrame(agg_data)
        # print(agg_df.head())

        # Melting the dataframe to have a long-form dataframe which seaborn prefers for relational plots
        agg_df_melted = agg_df.melt(id_vars=['algo_name', 'dim_size', 'func_id', 'iteration'],
                                    value_vars=['actual_val', 'predicted_val'],
                                    var_name='val_type', value_name='Error')

        # Plotting
        plt.figure(figsize=(15, 10))
        # sns.lineplot(data=agg_df_melted, x='iteration', y='Error', hue='algo_name', style='val_type',
        #              style_order=["actual_val", "predicted_val"],
        #              dashes={'actual_val': '', 'predicted_val': (2, 2)},
        #              markers=False, errorbar='sd', estimator='mean')
        sns.lineplot(data=agg_df_melted, x='iteration', y='Error', hue='algo_name', style='val_type',
                     style_order=["actual_val", "predicted_val"],
                     dashes={'actual_val': '', 'predicted_val': (2, 2)},
                     markers=False, errorbar=None, estimator='mean')

        plt.title(
            f'{how_many_func} functions aggregated view: Actual vs Predicted values over iterations for all comparators')
        plt.ylabel('Values')
        plt.xlabel('Iteration')
        plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.yscale('log')

        plt.show()


def step4_non_ela(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type = temp

    ###################### CREATE META-MODELS ######################

    # # !pip install --upgrade scikit-learn
    #
    # from sklearn.model_selection import GroupKFold
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error, r2_score
    # import matplotlib.pyplot as plt
    # from sklearn.pipeline import Pipeline
    # from sklearn.metrics import r2_score, root_mean_squared_error
    #
    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer, SimpleImputer
    # from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # import random
    # import pickle

    # CREATE A DIRECTORY
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    # CREATE A DIRECTORY
    required_dir = 'Models/RF'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _problem in prob_type:
        # CREATE A DIRECTORY
        required_results_dir = 'Models/RF/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    # SET EXPERIMENT SEED
    random.seed(42)

    # ----------------------- PREPARE FEATURE DATA -----------------------

    prob_list = prob_type
    algo_list = ng_algs
    dim_list = dims
    func_list = fids

    # all_files = []

    for prob in prob_list:

        # Defining parameters
        agg_algorithms = algo_list
        iteration_limit = 2000
        how_many_func = 0
        agg_data = []

        for algo in algo_list:
            for dim in dim_list:

                all_files = []

                for func in func_list:
                    how_many_func = how_many_func + 1

                    file_path = './NON_ELA/' + prob + '/' + algo + '_' + str(dim) + 'D_F' + str(func) + '_non_ela.csv'
                    # print(file_path)
                    current = pd.read_csv(file_path)
                    all_files.append(current)

                df = pd.concat(all_files)
                # print('shape of df:', df.shape)
                # print(df.dtypes)
                # print('data: ', df.shape[0])

                # Save combined to a files
                # df.to_csv(algo + '_BBOB_' + str(dim) + 'D_ELA.csv', encoding='utf-8')

                # ----------------------- PREPROCESS FEATURES -----------------------

                # Replace Inf/-Inf with NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # y = df['target']

                groups = df['func_id'].to_numpy()
                # groups = df['func_id']

                df = df.drop(
                    ['algo', 'dim', 'func_id', 'ins_id', 'run_id'], axis=1)  # 1-4-2024

                # print('shape of df:', df.shape)
                # print(df.dtypes)
                # replace infinite with nan. on train set. reflect on this later

                # Select columns to scale (all except the first one)
                # columns_to_scale = df.columns[1:]
                columns_to_scale = df.columns[0:]

                # Apply StandardScaler
                # scaler = StandardScaler()
                scaler = MinMaxScaler()
                df_scaled = scaler.fit_transform(df[columns_to_scale])

                # Create a DataFrame from the scaled data
                df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)

                # Concatenate the first column (unchanged) with the scaled DataFrame
                # df_final = pd.concat([df.iloc[:, :1].reset_index(drop=True), df_scaled], axis=1)
                X = df_scaled.iloc[:, :-1]
                y = df_scaled.iloc[:, -1]

                print(X)
                print(y)

                gkf = None
                if prob == 'BBOB':
                    gkf = GroupKFold(n_splits=24)
                elif prob == 'PBO':
                    gkf = GroupKFold(n_splits=25)

                # print(X.shape, y.shape, groups.shape)

                # ----------------------- TRAIN META-MODEL -----------------------

                model_storage = []

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    result_exist_flag = os.path.exists(
                        './Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav')

                    # Skip or not?
                    if force_replace_flag[0] or not result_exist_flag:
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        # X_train_pre, X_test = X.iloc[train_index], X.iloc[test_index]
                        # y_train_pre, y_test = y.iloc[train_index], y.iloc[test_index]
                        #
                        # y_train_0 = pd.DataFrame(y_train_pre).sort_values(by=['current_best'], ascending=False)
                        # X_train_0 = pd.DataFrame(X_train_pre.iloc[y_train_0.index])
                        # y_train = y_train_0.reset_index(drop=True)
                        # X_train = X_train_0.reset_index(drop=True)

                        # Handle NaN/Inf in folds if necessary
                        # Similar preprocessing as Ste
                        # p 1

                        # model = RandomForestRegressor(random_state=42)
                        # model = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=15, random_state=42, bootstrap=False) # 25-3-2024
                        model = RandomForestRegressor(n_estimators=50, criterion="squared_error", min_samples_split=2,
                                                      min_samples_leaf=1,
                                                      max_features=100, random_state=42, bootstrap=True)  # 25-3-2024
                        # model.fit(X_train, y_train)
                        # y_pred = model.predict(X_test)

                        # imputer = SimpleImputer(strategy='mean')
                        imputer = SimpleImputer(strategy='constant', fill_value=-1)
                        pipeline = Pipeline([('imputer', imputer), ('regressor', model)])
                        pipeline.fit(X_train, y_train)
                        # y_pred = pipeline.predict(X_test)

                        # Store trained model
                        model_storage.append(pipeline)

                    # else:
                    #     print('SKIP: ' + './Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav')

                # ----------------------- STORE TRAINED META-MODEL -----------------------

                if len(model_storage) > 0:
                    i = 0
                    for model_obj in model_storage:
                        # print(model_obj)
                        # save the model to disk
                        filename = 'Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(
                            i) + '.sav'
                        pickle.dump(model_storage[i], open(filename, 'wb'))
                        i = i + 1

                    print(
                        showCurrentDateTime() + ' ' + 'COMPLETED 4/4: Models were created (' + algo + '_' + str(
                            dim) + 'D)')

                # ----------------------- TEST META-MODEL -----------------------

                fig, axs = plt.subplots(24, figsize=(10, 60))  # Adjust the size as needed

                # Set options to display a full DataFrame
                pd.set_option('display.max_rows', None)  # No limit on the number of rows displayed
                pd.set_option('display.max_columns', None)  # No limit on the number of columns
                pd.options.display.float_format = '{:.10f}'.format

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    # print('Train ', str(i), 'Group: ', groups[train_index],)
                    # print('Test ', str(i), 'Group: ', groups[test_index],)

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # X_train, X_test_pre = X.iloc[train_index], X.iloc[test_index]
                    # y_train, y_test_pre = y.iloc[train_index], y.iloc[test_index]
                    #
                    # # print(y_test_pre)
                    #
                    # y_test_0 = pd.DataFrame(y_test_pre).sort_values(by=['current_best'], ascending=False)
                    # X_test_0 = pd.DataFrame(X_test_pre.iloc[y_test_0.index])
                    # y_test = y_test_0.reset_index(drop=True)
                    # X_test = X_test_0.reset_index(drop=True)

                    # print(y_test)

                    filename = 'Models/RF/' + prob + '/' + algo + '_' + str(dim) + 'D_' + 'metamodel_' + str(i) + '.sav'
                    loaded_model = pickle.load(open(filename, 'rb'))
                    pipeline = loaded_model
                    # pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    # print(y_pred.shape)

                    # # Reverse scaling
                    # y_pred_reverted = scaler.inverse_transform(pd.concat([X_test.iloc[:, :-1].reset_index(drop=True), pd.DataFrame(y_pred)], axis=1))
                    # y_test_reverted = scaler.inverse_transform(pd.concat([X_test.iloc[:, :-1].reset_index(drop=True), pd.DataFrame(y_test)], axis=1))

                    # print(y_pred)

                    # y_test_0 = pd.DataFrame(y_test).sort_values(by=['current_best'], ascending=False)
                    # # y_pred_0 = pd.DataFrame(y_pred_x.iloc[y_test_0.index])
                    # y_test_vis = y_test_0.reset_index(drop=True)
                    #
                    # y_pred_x = pd.DataFrame(y_pred, columns=['current_best'])
                    # y_pred_0 = pd.DataFrame(y_pred_x).sort_values(by=['current_best'], ascending=False)
                    # y_pred_vis = y_pred_0.reset_index(drop=True)

                    y_pred_vis = y_pred
                    y_test_vis = y_test

                    # print(y_pred_vis)

                    # Plot actual values
                    perf_r2 = r2_score(y_test_vis, y_pred_vis)
                    perf_rmse = root_mean_squared_error(y_test_vis, y_pred_vis)
                    # axs[i].plot(y_test.reset_index(drop=True), label='Actual', color='blue', linestyle='-')
                    axs[i].plot(y_test_vis, label='Actual', color='blue', linestyle='dotted')
                    axs[i].plot(y_pred_vis, label='Predicted', color='red', linestyle='dotted')

                    axs[i].set_title(
                        f'{algo} {dim}D F{groups[test_index[0]]}: actual and predicted values (RMSE: {perf_rmse})')  # Corrected
                    axs[i].set_xlabel('Sample Index')  # Corrected
                    axs[i].set_ylabel('Value')  # Corrected
                    axs[i].legend()
                    axs[i].set_yscale('log')
                    # axs[i].set_ylim(bottom=lb, top=ub)

                    # Collect data for aggregated visualization
                    for ind in y_test_vis.index:
                        if ind < iteration_limit:
                            agg_data.append({
                                'algo_name': algo,
                                'dim_size': dim,
                                'func_id': groups[test_index[0]],
                                'iteration': ind + 1,
                                'actual_val': y_test_vis['current_best'][ind],
                                'predicted_val': y_pred_vis['current_best'][ind]
                            })

                plt.tight_layout()
                plt.show()

                print(
                    showCurrentDateTime() + ' ' + 'COMPLETED 4/4: Models were tested (' + algo + '_' + str(dim) + 'D)')

        # Preparing data for plotting
        agg_df = pd.DataFrame(agg_data)
        # print(agg_df.head())

        # Melting the dataframe to have a long-form dataframe which seaborn prefers for relational plots
        agg_df_melted = agg_df.melt(id_vars=['algo_name', 'dim_size', 'func_id', 'iteration'],
                                    value_vars=['actual_val', 'predicted_val'],
                                    var_name='val_type', value_name='Error')

        # Plotting
        plt.figure(figsize=(15, 10))
        # sns.lineplot(data=agg_df_melted, x='iteration', y='Error', hue='algo_name', style='val_type',
        #              style_order=["actual_val", "predicted_val"],
        #              dashes={'actual_val': '', 'predicted_val': (2, 2)},
        #              markers=False, errorbar='sd', estimator='mean')
        sns.lineplot(data=agg_df_melted, x='iteration', y='Error', hue='algo_name', style='val_type',
                     style_order=["actual_val", "predicted_val"],
                     dashes={'actual_val': '', 'predicted_val': (2, 2)},
                     markers=False, errorbar=None, estimator='mean')

        plt.title(
            f'{how_many_func} functions aggregated view: Actual vs Predicted values over iterations for all comparators')
        plt.ylabel('Values')
        plt.xlabel('Iteration')
        plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.yscale('log')

        plt.show()


def showElapsedTime(start_time_inp):
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time_inp

    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    elapsed_time_formatted = f"{hours} hours, {minutes} minutes, and {seconds} seconds"
    print(showCurrentDateTime() + ' ' + elapsed_time_formatted)

    # return elapsed_time_formatted


def showCurrentDateTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def runPipeline(ng_algs_inp, fids_inp, iids_inp, dims_inp, bfacs_inp, force_replace_old_results_inp, repetition_inp,
                problem_type_inp):
    args = product(ng_algs_inp, fids_inp, iids_inp, dims_inp, bfacs_inp, [force_replace_old_results_inp],
                   [repetition_inp], problem_type_inp)
    args2 = (ng_algs_inp, fids_inp, iids_inp, dims_inp, bfacs_inp, [force_replace_old_results_inp], problem_type_inp)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # CREATE A DIRECTORY
    required_results_dir = 'Results'
    if not os.path.exists('./' + required_results_dir + '/'):
        os.makedirs(required_results_dir)
        print(required_results_dir + ' directory was created.')

    for _problem in problem_type_inp:
        # CREATE A DIRECTORY
        required_results_dir = 'Results/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')

    print(showCurrentDateTime() + ' ' + 'RUNNING...')

    # Execute the parallel function
    start_time = time.time()
    run_parallel_function(run_optimizer, args, pool_size)  # Perform Experiment
    showElapsedTime(start_time)

    for _problem in problem_type_inp:

        start_time = time.time()
        step2(args2)  # Format raw results
        showElapsedTime(start_time)

        start_time = time.time()
        if _problem == 'BBOB':
            step3_ela_feature(args2)  # Compute ELA features
        elif _problem == 'PBO':
            step3_non_ela_feature(args2)  # Prepare features
        showElapsedTime(start_time)

        start_time = time.time()
        if _problem == 'BBOB':
            step4_ela(args2)  # Build & Test ML meta-models
        elif _problem == 'PBO':
            step4_non_ela(args2)  # Build & Test ML meta-models
        showElapsedTime(start_time)

    print(showCurrentDateTime() + ' ' + 'COMPLETED')


# ---------------------- Main execution block ---------------------------

if __name__ == '__main__':
    freeze_support()

    # ALGORITHMS
    ng_algs = ['PSO', 'DCS']  # ['CMA', 'DE', 'PSO', 'DCS']

    # FUNCTIONS
    # fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # BBOB
    fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # PBO

    # INSTANCES
    # iids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iids = [1, 2, 3, 4, 5]

    # DIMENSIONS
    # dims = [2, 5, 10]  # [2, 5, 20, 40]  # BBOB
    dims = [16]  # [16, 64, 100, 625]  # PBO

    # BUDGETS FOR EACH DIMENSION
    # bfacs = [10000]  # BBOB
    bfacs = [1000]  # PBO

    # PARALLEL WORKERS
    pool_size = 26

    # RE-RUN THE EXPERIMENT
    force_replace_old_results = False

    # PER ALGORITHM / FUNCTION
    repetition = 5

    # PROBLEM TYPE: BBOB, PBO
    problem_type = ['PBO']  # ['BBOB', 'PBO']

    # Execute all procedures
    runPipeline(ng_algs, fids, iids, dims, bfacs, force_replace_old_results, repetition, problem_type)
