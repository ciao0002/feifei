import os
import pandas as pd
import numpy as np
import json
import shutil
import copy


def get_metrics(duration_list, traffic_name, total_summary_metrics, num_of_out):
    # calculate the mean final 10 rounds
    validation_duration_length = 10
    duration_list = np.array(duration_list)
    validation_duration = duration_list[-validation_duration_length:]
    validation_through = num_of_out[-validation_duration_length:]
    final_through = np.round(np.mean(validation_through), decimals=2)
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    total_summary_metrics["traffic"].append(traffic_name.split(".json")[0])
    total_summary_metrics["final_duration"].append(final_duration)
    total_summary_metrics["final_duration_std"].append(final_duration_std)
    total_summary_metrics["final_through"].append(final_through)

    return total_summary_metrics


def summary_detail_RL(memo_rl, total_summary_rl):
    """
    Used for test RL results
    """
    records_dir = os.path.join("records", memo_rl)

    # Check if there's a traffic_env.conf file in the main directory
    traffic_env_conf_path = os.path.join(records_dir, "traffic_env.conf")
    if os.path.exists(traffic_env_conf_path):
        # Original logic for directories with traffic_env.conf
        for traffic_file in os.listdir(records_dir):
            if ".json" not in traffic_file:
                continue
            print(traffic_file)

            traffic_env_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')
            dic_traffic_env_conf = json.load(traffic_env_conf)
            run_counts = dic_traffic_env_conf["RUN_COUNTS"]
            num_intersection = dic_traffic_env_conf['NUM_INTERSECTIONS']
            duration_each_round_list = []
            num_of_vehicle_in = []
            num_of_vehicle_out = []
            test_round_dir = os.path.join(records_dir, traffic_file, "test_round")
            try:
                round_files = os.listdir(test_round_dir)
            except:
                print("no test round in {}".format(traffic_file))
                continue
            round_files = [f for f in round_files if "round" in f]
            round_files.sort(key=lambda x: int(x[6:]))
            for round_rl in round_files:
                df_vehicle_all = []
                for inter_index in range(num_intersection):
                    try:
                        round_dir = os.path.join(test_round_dir, round_rl)
                        df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                       sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                       names=["vehicle_id", "enter_time", "leave_time"])


                        # [leave_time_origin, leave_time, enter_time, duration]
                        df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                        df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)
                        df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                                       df_vehicle_inter["enter_time"].values
                        tmp_idx = []
                        for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                            if "shadow" in v:
                                tmp_idx.append(i)
                        df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                        ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                        print("------------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                        df_vehicle_all.append(df_vehicle_inter)
                    except:
                        print("======= Error occurred during reading vehicle_inter_{}.csv")

                if len(df_vehicle_all) == 0:
                    print("====================================EMPTY")
                    continue

                df_vehicle_all = pd.concat(df_vehicle_all)
                # calculate the duration through the entire network
                vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
                ave_duration = vehicle_duration.mean()  # mean among all the vehicle

                duration_each_round_list.append(ave_duration)

                num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
                num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

                print("==== round: {0}\tave_duration: {1}\tnum_of_vehicle_in:{2}\tnum_of_vehicle_out:{2}"
                      .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))
                duration_flow = vehicle_duration.reset_index()
                duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x: x.split('_')[1])
                duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
                print(duration_flow_ave)
            result_dir = os.path.join("summary", memo_rl, traffic_file)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _res = {
                "duration": duration_each_round_list,
                "vehicle_in": num_of_vehicle_in,
                "vehicle_out": num_of_vehicle_out
            }
            result = pd.DataFrame(_res)
            result.to_csv(os.path.join(result_dir, "test_results.csv"))
            total_summary_rl = get_metrics(duration_each_round_list, traffic_file, total_summary_rl, num_of_vehicle_out)
            total_result = pd.DataFrame(total_summary_rl)
            total_result.to_csv(os.path.join("summary", memo_rl, "total_test_results.csv"))
    else:
        # New logic for directories without traffic_env.conf in the main directory
        # Look for the traffic_env.conf file in subdirectories or infer from other config files

        # For PPO-style directories, look for a config file or use the cityflow.config
        cityflow_config_path = os.path.join(records_dir, "cityflow.config")
        if os.path.exists(cityflow_config_path):
            with open(cityflow_config_path, 'r') as f:
                cityflow_conf = json.load(f)

            # Extract traffic file name from the flowFile field in cityflow.config
            traffic_file = cityflow_conf.get("flowFile", "default_traffic.json")
            print(traffic_file)

            # Use default values since we don't have traffic_env.conf
            # We'll try to infer these from the directory structure
            run_counts = 3600  # Default value
            num_intersection = 12  # For 3x4 grid

            duration_each_round_list = []
            num_of_vehicle_in = []
            num_of_vehicle_out = []

            test_round_dir = os.path.join(records_dir, "test_round")
            try:
                round_files = os.listdir(test_round_dir)
            except:
                print("no test round in {}".format(records_dir))
                return

            round_files = [f for f in round_files if "round" in f]
            round_files.sort(key=lambda x: int(x.replace("round_", "")) if x.replace("round_", "").isdigit() else 0)

            for round_rl in round_files:
                df_vehicle_all = []
                for inter_index in range(num_intersection):
                    try:
                        round_dir = os.path.join(test_round_dir, round_rl)
                        vehicle_file_path = os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index))

                        if os.path.exists(vehicle_file_path):
                            df_vehicle_inter = pd.read_csv(vehicle_file_path,
                                                           sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                           names=["vehicle_id", "enter_time", "leave_time"])

                            # [leave_time_origin, leave_time, enter_time, duration]
                            df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                            df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)
                            df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                                           df_vehicle_inter["enter_time"].values
                            tmp_idx = []
                            for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                                if "shadow" in v:
                                    tmp_idx.append(i)
                            df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                            ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                            print("------------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                            df_vehicle_all.append(df_vehicle_inter)
                    except Exception as e:
                        print(f"======= Error occurred during reading vehicle_inter_{inter_index}.csv: {e}")

                if len(df_vehicle_all) == 0:
                    print("====================================EMPTY")
                    continue

                df_vehicle_all = pd.concat(df_vehicle_all)
                # calculate the duration through the entire network
                vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
                ave_duration = vehicle_duration.mean()  # mean among all the vehicle

                duration_each_round_list.append(ave_duration)

                num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
                num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

                print("==== round: {0}\tave_duration: {1}\tnum_of_vehicle_in:{2}\tnum_of_vehicle_out:{3}"
                      .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))
                try:
                    duration_flow = vehicle_duration.reset_index()
                    duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x: x.split('_')[1] if '_' in x else 'default')
                    duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
                    print(duration_flow_ave)
                except:
                    print("Could not calculate direction-based durations")

            result_dir = os.path.join("summary", memo_rl)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _res = {
                "duration": duration_each_round_list,
                "vehicle_in": num_of_vehicle_in,
                "vehicle_out": num_of_vehicle_out
            }
            result = pd.DataFrame(_res)
            result.to_csv(os.path.join(result_dir, "test_results.csv"))
            total_summary_rl = get_metrics(duration_each_round_list, traffic_file, total_summary_rl, num_of_vehicle_out)
            total_result = pd.DataFrame(total_summary_rl)
            total_result.to_csv(os.path.join("summary", memo_rl, "total_test_results.csv"))


def summary_detail_conventional(memo_cv):
    """
    Used for test conventional results.
    """
    total_summary_cv = []
    records_dir = os.path.join("records", memo_cv)
    for traffic_file in os.listdir(records_dir):
        if "anon" not in traffic_file:
            continue
        traffic_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')

        dic_traffic_env_conf = json.load(traffic_conf)
        run_counts = dic_traffic_env_conf["RUN_COUNTS"]

        print(traffic_file)
        train_dir = os.path.join(records_dir, traffic_file)
        use_all = True
        if use_all:
            with open(os.path.join(records_dir, traffic_file, 'agent.conf'), 'r') as agent_conf:
                dic_agent_conf = json.load(agent_conf)

            df_vehicle_all = []
            NUM_OF_INTERSECTIONS = int(traffic_file.split('_')[1]) * int(traffic_file.split('_')[2])

            for inter_id in range(int(NUM_OF_INTERSECTIONS)):
                vehicle_csv = "vehicle_inter_{0}.csv".format(inter_id)

                df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, vehicle_csv),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                # [leave_time_origin, leave_time, enter_time, duration]
                df_vehicle_inter_0['leave_time_origin'] = df_vehicle_inter_0['leave_time']
                df_vehicle_inter_0['leave_time'].fillna(run_counts, inplace=True)
                df_vehicle_inter_0['duration'] = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0[
                    "enter_time"].values

                tmp_idx = []
                for i, v in enumerate(df_vehicle_inter_0["vehicle_id"]):
                    if "shadow" in v:
                        tmp_idx.append(i)
                df_vehicle_inter_0.drop(df_vehicle_inter_0.index[tmp_idx], inplace=True)

                ave_duration = df_vehicle_inter_0['duration'].mean(skipna=True)
                print("------------- inter_index: {0}\tave_duration: {1}".format(inter_id, ave_duration))
                df_vehicle_all.append(df_vehicle_inter_0)

            df_vehicle_all = pd.concat(df_vehicle_all, axis=0)
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()
            num_of_vehicle_in = len(df_vehicle_all['vehicle_id'].unique())
            num_of_vehicle_out = len(df_vehicle_all.dropna()['vehicle_id'].unique())
            save_path = os.path.join('records', memo_cv, traffic_file).replace("records", "summary")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # duration.to_csv(os.path.join(save_path, 'flow.csv'))
            total_summary_cv.append(
                [traffic_file, ave_duration, num_of_vehicle_in, num_of_vehicle_out, dic_agent_conf["FIXED_TIME"]])
        else:
            shutil.rmtree(train_dir)
    total_summary_cv = pd.DataFrame(total_summary_cv)
    total_summary_cv.sort_values([0], ascending=[True], inplace=True)
    total_summary_cv.columns = ['TRAFFIC', 'DURATION', 'CAR_NUMBER_in', 'CAR_NUMBER_out', 'CONFIG']
    total_summary_cv.to_csv(os.path.join("records", memo_cv,
                                         "total_baseline_results.csv").replace("records", "summary"),
                            sep='\t', index=False)


if __name__ == "__main__":
    """Only use these data"""
    total_summary = {
        "traffic": [],
        "final_duration": [],
        "final_duration_std": [],
        "final_through": [],
    }

    # Check existing directories in records
    records_dir = "records"
    if os.path.exists(records_dir):
        available_dirs = [d for d in os.listdir(records_dir)
                         if os.path.isdir(os.path.join(records_dir, d))]

        print(f"Available directories in records: {available_dirs}")

        # Look for directories that contain test_round and vehicle_inter_*.csv files
        for dir_name in available_dirs:
            dir_path = os.path.join(records_dir, dir_name)

            # Check if this directory has the expected structure
            test_round_path = os.path.join(dir_path, "test_round")
            if os.path.exists(test_round_path):
                # Find round directories
                round_dirs = [rd for rd in os.listdir(test_round_path)
                             if os.path.isdir(os.path.join(test_round_path, rd)) and "round" in rd]

                if round_dirs:
                    # Check if any round has vehicle_inter_*.csv files
                    for round_dir in round_dirs:
                        round_path = os.path.join(test_round_path, round_dir)
                        vehicle_files = [vf for vf in os.listdir(round_path)
                                       if vf.startswith("vehicle_inter_") and vf.endswith(".csv")]

                        if vehicle_files:
                            print(f"Analyzing data from: {dir_name}")
                            try:
                                summary_detail_RL(dir_name, copy.deepcopy(total_summary))
                            except Exception as e:
                                print(f"Error processing {dir_name}: {e}")
                            break  # Process only the first valid directory per parent

    # Also check the original CoLight directories if they exist
    memo_jinan = "colight_jinan_real_60rds"
    jinan_path = os.path.join("records", memo_jinan)
    if os.path.exists(jinan_path):
        print("Analyzing Jinan CoLight data...")
        summary_detail_RL(memo_jinan, copy.deepcopy(total_summary))

    memo_hangzhou = "colight_hangzhou_real_60rds"
    hangzhou_path = os.path.join("records", memo_hangzhou)
    if os.path.exists(hangzhou_path):
        print("Analyzing Hangzhou CoLight data...")
        summary_detail_RL(memo_hangzhou, copy.deepcopy(total_summary))

    # summary_detail_conventional(memo)
