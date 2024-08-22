import os
from datetime import datetime
import time
import pflacco.classical_ela_features as pflacco_ela
import pandas as pd


def show_current_date_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def show_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    hours, minutes, seconds = int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60)
    return f"{hours} hours, {minutes} minutes, and {seconds} seconds"


def compute_ela(X, y, min_y, max_y, lower_bound, upper_bound):
    # y_rescale = (max(y) - y) / (max(y) - min(y) + 1e-20)
    y_rescale = (max_y - y) / (max_y - min_y + 1e-30)
    # y_rescale = (max(y) - y) / (max(y) - min(y) + 1e-30)
    # y_rescale = y
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


def create_new_results_directory(problem_type):
    # Create result directories
    required_results_dir = 'Results'
    if not os.path.exists('./' + required_results_dir + '/'):
        os.makedirs(required_results_dir)
        print(required_results_dir + ' directory was created.')

    for _problem in problem_type:
        # CREATE A DIRECTORY
        required_results_dir = 'Results/' + _problem
        if not os.path.exists('./' + required_results_dir + '/'):
            os.makedirs(required_results_dir)
            print(required_results_dir + ' directory was created.')


def create_new_models_directory(problem_type, models):
    # CREATE MODEL DIRECTORIES
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    for _model in models:
        required_dir = 'Models/' + str(_model).split('(')[0]
        if not os.path.exists('./' + required_dir + '/'):
            os.makedirs(required_dir)
            print(required_dir + ' directory was created')

        for _problem in problem_type:
            required_results_dir = 'Models/' + str(_model).split('(')[0] + '/' + _problem
            if not os.path.exists('./' + required_results_dir + '/'):
                os.makedirs(required_results_dir)
                print(required_results_dir + ' directory was created.')
