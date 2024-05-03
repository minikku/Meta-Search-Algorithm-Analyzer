import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from .utilities import show_current_date_time
from .visualization import per_function_plot, aggregated_plot


def step4_ela_minimize(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type, ml_model_options = temp

    ###################### CREATE META-MODELS ######################

    # CREATE A DIRECTORY
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    # CREATE A DIRECTORY
    for _model in ml_model_options:
        required_dir = 'Models/' + str(_model).split('(')[0]
        if not os.path.exists('./' + required_dir + '/'):
            os.makedirs(required_dir)
            print(required_dir + ' directory was created')

        for _problem in prob_type:
            # CREATE A DIRECTORY
            required_results_dir = 'Models/' + str(_model).split('(')[0] + '/' + _problem
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
        error_agg_data = [0] * len(ml_model_options)

        for algo in algo_list:
            for dim in dim_list:

                all_files = []

                for func in func_list:

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

                gkf = GroupKFold(n_splits=len(fids))

                # print(X.shape, y.shape, groups.shape)

                # ----------------------- BUILD META-MODEL -----------------------

                # Train different models
                models = ml_model_options

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    target_func_no = groups[test_index[0]]

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Lists to store predictions and errors
                    predict_vals = []
                    error_vals = []
                    actual_vals = []

                    model_idx = 0
                    # Train and predict with each model
                    for model in models:

                        filename = 'Models/' + str(model).split('(')[0] + '/' + prob + '/' + algo + '_' + str(
                            dim) + 'D_' + 'F' + str(target_func_no) + '_metamodel.sav'

                        result_exist_flag = os.path.exists(filename)

                        # ----------------------- TRAIN META-MODEL -----------------------
                        if force_replace_flag[0] or not result_exist_flag:

                            imputer = SimpleImputer(strategy='constant', fill_value=-1)
                            pipeline = Pipeline([('imputer', imputer), ('regressor', model)])
                            pipeline.fit(X_train, y_train)

                            # save the model to disk
                            pickle.dump(pipeline, open(filename, 'wb'))

                        # ----------------------- TEST META-MODEL -----------------------
                        loaded_model = pickle.load(open(filename, 'rb'))
                        pipeline = loaded_model
                        y_pred = pipeline.predict(X_test)

                        # Pre-processing
                        y_test_0 = pd.DataFrame(y_test).sort_values(by=['current_best'], ascending=False)
                        # y_pred_0 = pd.DataFrame(y_pred_x.iloc[y_test_0.index])
                        y_test_vis = y_test_0.reset_index(drop=True)

                        y_pred_x = pd.DataFrame(y_pred, columns=['current_best'])
                        y_pred_0 = pd.DataFrame(y_pred_x).sort_values(by=['current_best'], ascending=False)
                        y_pred_vis = y_pred_0.reset_index(drop=True)

                        predict_vals.append(y_pred_vis)
                        error_vals.append(root_mean_squared_error(y_test_vis, y_pred_vis))
                        if len(actual_vals) == 0:
                            actual_vals.append(y_test_vis)

                        # Collect data for aggregated visualization
                        error_agg_data[model_idx] += error_vals[model_idx]
                        model_idx = model_idx + 1

                    # Plotting
                    per_function_plot(algo, dim, actual_vals, predict_vals, error_vals, target_func_no, ml_model_options)

                print(show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were trained and tested (' + algo + '_' + str(dim) + 'D)')

        # Plotting
        aggregated_plot(fids, error_agg_data, ml_model_options)


def step4_non_ela_minimize(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type, ml_model_options = temp

    ###################### CREATE META-MODELS ######################

    # CREATE A DIRECTORY
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    # CREATE A DIRECTORY
    for _model in ml_model_options:
        required_dir = 'Models/' + str(_model).split('(')[0]
        if not os.path.exists('./' + required_dir + '/'):
            os.makedirs(required_dir)
            print(required_dir + ' directory was created')

        for _problem in prob_type:
            # CREATE A DIRECTORY
            required_results_dir = 'Models/' + str(_model).split('(')[0] + '/' + _problem
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
        error_agg_data = [0] * len(ml_model_options)

        for algo in algo_list:
            for dim in dim_list:

                all_files = []

                for func in func_list:

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

                # print(X)
                # print(y)

                gkf = GroupKFold(n_splits=len(fids))

                # print(X.shape, y.shape, groups.shape)

                # ----------------------- TRAIN META-MODEL -----------------------

                # Train different models
                models = ml_model_options

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    target_func_no = groups[test_index[0]]

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Lists to store predictions and errors
                    predict_vals = []
                    error_vals = []
                    actual_vals = []

                    model_idx = 0
                    # Train and predict with each model
                    for model in models:

                        filename = 'Models/' + str(model).split('(')[0] + '/' + prob + '/' + algo + '_' + str(
                            dim) + 'D_' + 'F' + str(target_func_no) + '_metamodel.sav'

                        result_exist_flag = os.path.exists(filename)

                        # ----------------------- TRAIN META-MODEL -----------------------
                        if force_replace_flag[0] or not result_exist_flag:
                            imputer = SimpleImputer(strategy='constant', fill_value=-1)
                            pipeline = Pipeline([('imputer', imputer), ('regressor', model)])
                            pipeline.fit(X_train, y_train)

                            # save the model to disk
                            pickle.dump(pipeline, open(filename, 'wb'))

                        # ----------------------- TEST META-MODEL -----------------------
                        loaded_model = pickle.load(open(filename, 'rb'))
                        pipeline = loaded_model
                        y_pred = pipeline.predict(X_test)

                        # Pre-processing
                        y_pred_vis = pd.DataFrame(y_pred, columns=['target'])
                        y_test_vis = pd.DataFrame(y_test).reset_index(drop=True)

                        predict_vals.append(y_pred_vis)
                        error_vals.append(root_mean_squared_error(y_test_vis, y_pred_vis))
                        if len(actual_vals) == 0:
                            actual_vals.append(y_test_vis)

                        # Collect data for aggregated visualization
                        error_agg_data[model_idx] += error_vals[model_idx]
                        model_idx = model_idx + 1

                    # Plotting
                    per_function_plot(algo, dim, actual_vals, predict_vals, error_vals, target_func_no,
                                          ml_model_options)

                print(show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were trained and tested (' + algo + '_' + str(dim) + 'D)')

        # Plotting
        aggregated_plot(fids, error_agg_data, ml_model_options)


def step4_non_ela_maximize(temp):
    ng_algs, fids, iids, dims, bfacs, force_replace_flag, prob_type, ml_model_options = temp

    ###################### CREATE META-MODELS ######################

    # CREATE A DIRECTORY
    required_dir = 'Models'
    if not os.path.exists('./' + required_dir + '/'):
        os.makedirs(required_dir)
        print(required_dir + ' directory was created')

    # CREATE A DIRECTORY
    for _model in ml_model_options:
        required_dir = 'Models/' + str(_model).split('(')[0]
        if not os.path.exists('./' + required_dir + '/'):
            os.makedirs(required_dir)
            print(required_dir + ' directory was created')

        for _problem in prob_type:
            # CREATE A DIRECTORY
            required_results_dir = 'Models/' + str(_model).split('(')[0] + '/' + _problem
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
        agg_data = []
        agg_df_melted = []
        error_agg_data = [0] * len(ml_model_options)

        for algo in algo_list:
            for dim in dim_list:

                all_files = []

                for func in func_list:

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

                # print(X)
                # print(y)

                gkf = GroupKFold(n_splits=len(fids))

                # print(X.shape, y.shape, groups.shape)

                # ----------------------- TRAIN META-MODEL -----------------------

                # Train different models
                models = ml_model_options

                for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

                    target_func_no = groups[test_index[0]]

                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Lists to store predictions and errors
                    predict_vals = []
                    error_vals = []
                    actual_vals = []

                    model_idx = 0
                    # Train and predict with each model
                    for model in models:

                        filename = 'Models/' + str(model).split('(')[0] + '/' + prob + '/' + algo + '_' + str(
                            dim) + 'D_' + 'F' + str(target_func_no) + '_metamodel.sav'

                        result_exist_flag = os.path.exists(filename)

                        # ----------------------- TRAIN META-MODEL -----------------------
                        if force_replace_flag[0] or not result_exist_flag:
                            imputer = SimpleImputer(strategy='constant', fill_value=-1)
                            pipeline = Pipeline([('imputer', imputer), ('regressor', model)])
                            pipeline.fit(X_train, y_train)

                            # save the model to disk
                            pickle.dump(pipeline, open(filename, 'wb'))

                        # ----------------------- TEST META-MODEL -----------------------
                        loaded_model = pickle.load(open(filename, 'rb'))
                        pipeline = loaded_model
                        y_pred = pipeline.predict(X_test)

                        # Pre-processing
                        y_pred_vis = pd.DataFrame(y_pred, columns=['target'])
                        y_test_vis = pd.DataFrame(y_test).reset_index(drop=True)

                        predict_vals.append(y_pred_vis)
                        error_vals.append(root_mean_squared_error(y_test_vis, y_pred_vis))
                        if len(actual_vals) == 0:
                            actual_vals.append(y_test_vis)

                        # Collect data for aggregated visualization
                        error_agg_data[model_idx] += error_vals[model_idx]
                        model_idx = model_idx + 1

                    # Plotting
                    per_function_plot(algo, dim, actual_vals, predict_vals, error_vals, target_func_no,
                                      ml_model_options)

                print(
                    show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were trained and tested (' + algo + '_' + str(
                        dim) + 'D)')

        # Plotting
        aggregated_plot(fids, error_agg_data, ml_model_options)
