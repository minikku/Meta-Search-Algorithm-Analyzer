import os
import warnings
import time
from itertools import product

from .utilities import show_elapsed_time, show_current_date_time
from .algorithm_eval import run_optimizer, run_parallel_function
from .data_processing import step2
from .feature_computation import step3_ela_feature, step3_non_ela_feature
from .models import step4_ela, step4_non_ela
# from ..algorithms import *


def run_pipeline(ng_algs_inp, fids_inp, iids_inp, dims_inp, bfacs_inp, force_replace_old_results_inp, repetition_inp,
                problem_type_inp, pool_size_inp):

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

    print(show_current_date_time() + ' ' + 'RUNNING...')

    # Execute the parallel function
    start_time = time.time()
    run_parallel_function(run_optimizer, args, pool_size_inp)  # Perform Experiment
    show_elapsed_time(start_time)

    for _problem in problem_type_inp:

        start_time = time.time()
        step2(args2)  # Format raw results
        show_elapsed_time(start_time)

        start_time = time.time()
        if _problem == 'BBOB':
            step3_ela_feature(args2)  # Compute ELA features
        elif _problem == 'PBO':
            step3_non_ela_feature(args2)  # Prepare features
        show_elapsed_time(start_time)

        start_time = time.time()
        if _problem == 'BBOB':
            step4_ela(args2)  # Build & Test ML meta-models
        elif _problem == 'PBO':
            step4_non_ela(args2)  # Build & Test ML meta-models
        show_elapsed_time(start_time)

    print(show_current_date_time() + ' ' + 'COMPLETED')
