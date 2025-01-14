import os
import random

from matplotlib.lines import lineStyles

os.environ["OPENBLAS_NUM_THREADS"] = "6"

from multiprocessing import freeze_support
import warnings
import time
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from msaa.core.algorithm_eval import run_parallel_function, run_optimizer
from msaa.core.data_processing import format_raw_result
from msaa.core.feature_computation import ela_feature_minimize, non_ela_feature_maximize
from msaa.core.meta_model import MetaModel
from msaa.core.visualization import per_function_plot, aggregated_plot
from msaa.core.utilities import show_elapsed_time, show_current_date_time, create_new_results_directory, \
    create_new_models_directory

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from msaa.models.custom_random_forest import CustomRandomForestRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

random.seed(42)

if __name__ == '__main__':
    freeze_support()

    # ============== CONFIGURATIONS ==============

    # ALGORITHMS
    ng_algs = ['DCS']

    # FUNCTIONS
    fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 16, 17, 18, 19, 20, 21 ,22, 23, 24]

    # INSTANCES
    iids = [1]  # [1,2,3,4,5,6,7,8,9,10]

    # DIMENSIONS
    dims = [2]

    # BUDGETS FOR EACH DIMENSION
    bfacs = [10000]

    # PARALLEL WORKERS
    pool_size = 1

    # RE-RUN THE EXPERIMENT
    force_replace_old_results = False

    # PER ALGORITHM / FUNCTION
    repetition = 1

    # PROBLEM TYPE: BBOB, PBO
    problem_type = ['BBOB']

    # ML MODEL OPTIONS
    # models = [
    #     LinearRegression(),
    #     Ridge(),
    #     Lasso(),
    #     ElasticNet(),
    #     KNeighborsRegressor(),
    #     DecisionTreeRegressor(),
    #     RandomForestRegressor(),
    #     GradientBoostingRegressor(),
    # ]

    models = [
        RandomForestRegressor(),
        CustomRandomForestRegressor()
    ]

    # WINDOW SIZE
    window_size = 100

    # PREPARE CONFIG BUNDLES
    args1 = product(ng_algs, fids, iids, dims, bfacs, [force_replace_old_results],
                    [repetition], problem_type)
    args2 = (ng_algs, fids, iids, dims, bfacs, [force_replace_old_results], problem_type, window_size)

    # ============== EXPERIMENT ==============

    # Run the experiment (algorithms evaluation)
    start_time = time.time()

    print(show_current_date_time() + ' ' + 'RUNNING...')

    # Create new directories
    create_new_results_directory(problem_type)

    # Execute parallel runs
    run_parallel_function(run_optimizer, args1, pool_size)
    show_elapsed_time(start_time)

    # ============== RESULT FORMATTING ==============

    # Format raw results
    for _problem in problem_type:
        start_time = time.time()
        format_raw_result(args2)
        show_elapsed_time(start_time)

    # ============== COMPUTE FEATURES ==============

    # Compute ELA/Non-ELA features
    for _problem in problem_type:
        start_time = time.time()
        if _problem == 'BBOB':
            ela_feature_minimize(args2)
        elif _problem == 'PBO':
            non_ela_feature_maximize(args2)
        show_elapsed_time(start_time)

    # ============== META-MODEL ==============

    # Build & Test ML meta-models
    start_time = time.time()

    # Create new directories
    create_new_models_directory(problem_type, models)

    model_names = []
    for model in models:
        model_names.append(model.__str__())

    prob_list = problem_type
    algo_list = ng_algs
    dim_list = dims
    func_list = fids

    for prob in prob_list:

        # Collecting model errors
        error_agg_data = [0] * len(models)

        for algo in algo_list:
            for dim in dim_list:

                # ----------------------- LOAD DATA -----------------------

                all_files = []

                for func in func_list:
                    file_path = './ELA/' + prob + '/' + algo + '_' + str(dim) + 'D_F' + str(func) + '_ela.csv'
                    current = pd.read_csv(file_path)
                    all_files.append(current)

                df = pd.concat(all_files, ignore_index=True)

                # ----------------------- PREPROCESS FEATURES -----------------------

                # Replace Inf/-Inf with NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Drop columns that contain any NaN values
                df = df.dropna(axis=1, how='any')

                # Defining a target values
                # y = df['current_best']
                y = df['final_best']

                groups = df['func_id'].to_numpy()

                df = df.drop(
                    ['algo', 'dim', 'func_id', 'ins_id', 'run_id',
                     'ic.costs_runtime'],
                    axis=1)

                # Select columns to scale (all except the first one)
                columns_to_scale = df.columns[1:]

                scaler = MinMaxScaler()
                df_scaled = scaler.fit_transform(df[columns_to_scale])

                # Create a DataFrame from the scaled data
                df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale)

                # Concatenate the first column (unchanged) with the scaled DataFrame
                df_final = pd.concat([df.iloc[:, :1].reset_index(drop=True), df_scaled], axis=1)

                X = df_final

                gkf = GroupKFold(n_splits=len(fids))

                # ----------------------- BUILD META-MODEL -----------------------


                # Train on different models
                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    target_func_no = groups[test_index[0]]

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Lists to store predictions and errors
                    predict_vals = []
                    error_vals = []
                    actual_vals = []

                    # Collect errors from predictions
                    func_error_agg_data = []

                    model_idx = 0
                    # Train and predict with each model
                    for model in models:

                        filename = 'Models/' + str(model).split('(')[0] + '/' + prob + '/' + algo + '_' + str(
                            dim) + 'D_' + 'F' + str(target_func_no) + '_metamodel.sav'

                        result_exist_flag = os.path.exists(filename)

                        # Declare MetaModel instance
                        meta_model = MetaModel(model, X_train)
                        selected_input_features = X_train.columns[:-1]
                        target_feature = X_train.columns[-1]

                        # ----------------------- TRAIN META-MODEL -----------------------
                        if force_replace_old_results or not result_exist_flag:
                            meta_model.fit_model(selected_input_features, target_feature)

                            # save the model to disk
                            pickle.dump(meta_model, open(filename, 'wb'))

                        # ----------------------- TEST META-MODEL -----------------------
                        loaded_model = pickle.load(open(filename, 'rb'))
                        meta_model = loaded_model
                        y_pred = meta_model.make_prediction(X_test[selected_input_features])
                        predict_vals.append(y_pred)

                        # Calculate SE
                        squared_errors = (X_test[target_feature] - y_pred) ** 2
                        func_error_agg_data.append(squared_errors)

                        # # Plotting
                        # plt.figure(figsize=(10, 6))
                        # plt.scatter(range(len(X_test[target_feature])), X_test[target_feature], label='Actual', s=2)
                        # plt.scatter(range(len(y_pred)), y_pred, label='Predicted', s=2)
                        # plt.title('Actual vs Predicted Values (F' + str(target_func_no) + ' : ' + model.__str__() + ')')
                        # plt.ylabel('Final Best Values')
                        # plt.xlabel('Iterations')
                        # # plt.yscale('log', base=2)
                        # plt.grid(True, linestyle='--', alpha=0.7)
                        # # plt.tight_layout()
                        # plt.legend()
                        # plt.show()

                        model_idx = model_idx + 1

                    # Plotting
                    # plt.figure(figsize=(16, 10))
                    plt.scatter(range(len(X_test[target_feature])), X_test[target_feature], label='Actual', s=2,
                                marker='*')
                    for m_idx in range(len(predict_vals)):
                        plt.scatter(range(len(predict_vals[m_idx])), predict_vals[m_idx], label='Predicted (' + models[m_idx].__str__()[:-2] + ')', s=6, marker='o')
                    plt.title('Actual vs Predicted Values (F' + str(target_func_no) + ')')
                    plt.ylabel('Final Best Values')
                    plt.xlabel('Iterations')
                    # plt.yscale('log', base=2)
                    plt.grid(True, linestyle=':', alpha=0.5)
                    plt.tight_layout()
                    plt.legend()
                    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                    plt.show()

                    # Plotting
                    # plt.figure(figsize=(10, 6))
                    plt.boxplot(func_error_agg_data)
                    plt.xticks(ticks=np.arange(1, len(model_names) + 1), labels=model_names, rotation=45, ha='right')
                    plt.title('Distribution of Squared Errors (F' + str(target_func_no) + ')')
                    plt.ylabel('Squared Error')
                    plt.yscale('log', base=2)
                    plt.grid(True, linestyle='-', alpha=0.7)
                    plt.tight_layout()
                    plt.show()

                print(
                    show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were trained and tested (' + algo + '_' + str(
                        dim) + 'D)')

        # Plotting
        # aggregated_plot(fids, error_agg_data, models)

    show_elapsed_time(start_time)

print(show_current_date_time() + ' ' + 'COMPLETED')
