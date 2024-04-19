import os

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import random
import pandas as pd
import numpy as np
import seaborn as sns
from .utilities import show_current_date_time


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
        agg_data = []

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
                        show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were created (' + algo + '_' + str(
                            dim) + 'D)')

                # ----------------------- TEST META-MODEL -----------------------

                fig, axs = plt.subplots(len(fids), figsize=(10, 60))  # Adjust the size as needed

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
                    show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were tested (' + algo + '_' + str(dim) + 'D)')

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
            f'{len(fids)} functions aggregated view: Actual vs Predicted values over iterations for all comparators')
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
        agg_data = []

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
                        show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were created (' + algo + '_' + str(
                            dim) + 'D)')

                # ----------------------- TEST META-MODEL -----------------------

                fig, axs = plt.subplots(len(fids), figsize=(10, 60))  # Adjust the size as needed

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

                    y_pred_vis = pd.DataFrame(y_pred, columns=['target'])
                    y_test_vis = pd.DataFrame(y_test).reset_index(drop=True)

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
                                'actual_val': y_test_vis['target'][ind],
                                'predicted_val': y_pred_vis['target'][ind]
                            })

                plt.tight_layout()
                plt.show()

                print(
                    show_current_date_time() + ' ' + 'COMPLETED 4/4: Models were tested (' + algo + '_' + str(dim) + 'D)')

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
            f'{len(fids)} functions aggregated view: Actual vs Predicted values over iterations for all comparators')
        plt.ylabel('Values')
        plt.xlabel('Iteration')
        plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.yscale('log')

        plt.show()