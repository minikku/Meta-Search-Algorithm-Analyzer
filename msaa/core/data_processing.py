import csv
import json
import os
from .utilities import show_current_date_time

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
                                # print(base_entries)

                                sub_entry = base_folder + '/' + base_entries[0]
                                base_entries_index = 0
                                target_file_a = base_folder + '/' + base_entries[1]
                                if os.path.isfile(sub_entry):
                                    sub_entry = base_folder + '/' + base_entries[1]
                                    base_entries_index = 1
                                    target_file_a = base_folder + '/' + base_entries[0]
                                # print(sub_entry)

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

                                    with open(target_file_a, 'r') as json_file:
                                        # Read json summary file
                                        json_data = json.load(json_file)
                                        # print(json_data)

                                        # Iterate over each file in the file list
                                        # Open the current raw data file for reading

                                        dir_scan_b = os.listdir(sub_entry)
                                        target_file_b = sub_entry + '/' + dir_scan_b[0]
                                        # print(target_file_b)

                                        with open(target_file_b, 'r') as file:
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

    print(show_current_date_time() + ' ' + 'COMPLETED 2/4: Raw results were formatted')
