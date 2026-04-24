import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
from collections import deque
from multiprocessing import Process
import heapq


class Intersection:
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, lanes_length_dict):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.lane_length = lanes_length_dict
        self.obs_length = dic_traffic_env_conf["OBS_LENGTH"]

        self.list_approachs = ["W", "E", "N", "S"]
        # corresponding exiting lane for entering lanes
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approachs}
        self.list_phases = dic_traffic_env_conf["PHASE"]

        # generate all lanes
        self.list_entering_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + "_" + str(i) for i in
                                         range(lane_number)]
        self.list_exiting_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + "_" + str(i) for i in
                                        range(lane_number)]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]
        self.topology_vector = list(
            light_id_dict.get(
                "topology_vector",
                [0.0] * int(self.dic_traffic_env_conf.get("INTERSECTION_TOPOLOGY_DIM", 8))
            )
        )

        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second
        self.kalman_pressure_estimate = np.zeros(12, dtype=np.float32)
        self.kalman_pressure_variance = np.ones(12, dtype=np.float32)
        self.avg_vehicle_length = float(self.dic_traffic_env_conf.get("AVG_VEHICLE_LENGTH", 7.5))
        self.max_vehicle_speed = float(self.dic_traffic_env_conf.get("MAX_VEHICLE_SPEED", 11.11))
        self.max_wait_time = float(self.dic_traffic_env_conf.get("MAX_WAIT_TIME", 45.0))
        self.stop_speed_threshold = float(self.dic_traffic_env_conf.get("STOP_SPEED_THRESHOLD", 0.1))
        self.wait_time_increment = float(self.dic_traffic_env_conf.get("WAIT_TIME_INCREMENT", 1.0))
        self.dic_vehicle_waiting_time = {}
        self.trend_window_steps = max(
            1,
            int(self.dic_traffic_env_conf.get("TREND_WINDOW_STEPS", self.dic_traffic_env_conf["MIN_ACTION_TIME"])),
        )
        self.lane_queue_count_history = {
            lane: deque(maxlen=self.trend_window_steps + 1) for lane in self.list_entering_lanes
        }
        self.lane_vehicle_history = {
            lane: deque(maxlen=self.trend_window_steps + 1) for lane in self.list_entering_lanes
        }

        # =========== signal info set ================
        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        os.makedirs(path_to_log, exist_ok=True)
        path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode="a", header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time, path_to_log):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                os.makedirs(path_to_log, exist_ok=True)
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                # self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment
                self.next_phase_to_set_index = action + 1
            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                os.makedirs(path_to_log, exist_ok=True)
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step
        self.dic_feature_previous_step = self.dic_feature

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state["get_lane_vehicles"][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]
        self._update_vehicle_waiting_times()

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step_in)
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step_in) - set(self.list_lane_vehicle_previous_step_in))
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(set(self.list_lane_vehicle_current_step_in) -
                                         set(self.list_lane_vehicle_previous_step_in))
        else:
            list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step_in) -
                                         set(self.list_lane_vehicle_current_step_in))
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()
        prev_efficient_pressure = self.dic_feature.get(
            "traffic_movement_pressure_queue_efficient",
            [0.0] * 12,
        )
        ema_alpha = float(self.dic_traffic_env_conf.get("EMA_ALPHA", 0.4))
        kalman_q = float(self.dic_traffic_env_conf.get("KALMAN_PRESSURE_Q", 0.05))
        kalman_r = float(self.dic_traffic_env_conf.get("KALMAN_PRESSURE_R", 1.0))
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        phase_elapsed_base = float(self.dic_traffic_env_conf.get("PHASE_ELAPSED_NORM_BASE", 15.0))
        dic_feature["phase_elapsed"] = [float(self.current_phase_duration) / max(1.0, phase_elapsed_base)]
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle_entring()
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream()
        dic_feature["delta_lane_num_vehicle"] = [dic_feature["lane_num_vehicle"][i] -
                                                 dic_feature["lane_num_vehicle_downstream"][i]
                                                 for i in range(12)]
        dic_feature["lane_num_waiting_vehicle_in"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_waiting_vehicle_out"] = self._get_lane_queue_length(self.list_exiting_lanes)

        dic_feature["traffic_movement_pressure_queue"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["traffic_movement_pressure_queue_efficient"] = self._get_traffic_movement_pressure_efficient(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])
        dic_feature["delta_pressure"] = [
            float(dic_feature["traffic_movement_pressure_queue_efficient"][i] - prev_efficient_pressure[i])
            for i in range(12)
        ]

        dic_feature["traffic_movement_pressure_num"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_vehicle"], dic_feature["lane_num_vehicle_downstream"])

        (
            tmp_part_n,
            tmp_part_q,
            tmp_efficient_part,
            enter_running_part,
            lepq,
            enter_vehicle_close,
        ) = self._get_part_traffic_movement_features()

        dic_feature["lane_enter_running_part"] = list(enter_running_part)
        dic_feature["lane_far_approaching_part"] = self._get_lane_far_approaching_part()
        dic_feature["lane_num_vehicle_close"] = list(enter_vehicle_close)
        dic_feature["lane_average_speed_effective"] = self._get_lane_average_speed_effective()
        dic_feature["downstream_saturation_movement"] = self._get_downstream_saturation_movement(
            dic_feature["lane_num_waiting_vehicle_out"]
        )
        qgr, qdr = self._get_queue_growth_decay_movement(dic_feature["lane_num_waiting_vehicle_in"])
        dic_feature["queue_growth_rate_movement"] = qgr
        dic_feature["queue_decay_rate_movement"] = qdr
        dic_feature["weighted_accumulated_wait_movement"] = self._get_weighted_accumulated_wait_movement()
        dic_feature["intersection_topology_vector"] = list(self.topology_vector)
        # ATS/QDSE-style state expects bounded, comparable inputs. Keep the
        # original raw features for baseline runs, and expose normalized aliases
        # so ATS feature sets do not mix raw counts with [0,1] trends.
        dic_feature["ats_effective_pressure"] = self._normalize_entering_by_capacity(
            dic_feature["traffic_movement_pressure_queue_efficient"], signed=True
        )
        dic_feature["ats_effective_running_demand"] = self._normalize_entering_by_capacity(
            dic_feature["lane_enter_running_part"], signed=False
        )
        dic_feature["ats_far_approaching_demand"] = self._normalize_entering_by_capacity(
            dic_feature["lane_far_approaching_part"], signed=False
        )
        # Lane Average Speed (LAS): average speed of vehicles inside the effective
        # range L=v_max*action_time near the stop line. We normalize by max speed
        # so ATS-style state components remain bounded and comparable.
        dic_feature["ats_lane_average_speed"] = [
            float(np.clip(float(v) / max(self.max_vehicle_speed, 1.0), 0.0, 1.0))
            for v in dic_feature["lane_average_speed_effective"]
        ]
        prev_pressure_ema = np.asarray(
            self.dic_feature.get(
                "traffic_movement_pressure_queue_efficient_ema",
                dic_feature["traffic_movement_pressure_queue_efficient"],
            ),
            dtype=np.float32,
        )
        prev_running_ema = np.asarray(
            self.dic_feature.get(
                "lane_enter_running_part_ema",
                dic_feature["lane_enter_running_part"],
            ),
            dtype=np.float32,
        )
        raw_pressure = np.asarray(dic_feature["traffic_movement_pressure_queue_efficient"], dtype=np.float32)
        raw_running = np.asarray(dic_feature["lane_enter_running_part"], dtype=np.float32)
        dic_feature["traffic_movement_pressure_queue_efficient_ema"] = list(
            ema_alpha * raw_pressure + (1.0 - ema_alpha) * prev_pressure_ema
        )
        pred_var = self.kalman_pressure_variance + kalman_q
        kalman_gain = pred_var / (pred_var + kalman_r)
        self.kalman_pressure_estimate = self.kalman_pressure_estimate + kalman_gain * (
            raw_pressure - self.kalman_pressure_estimate
        )
        self.kalman_pressure_variance = (1.0 - kalman_gain) * pred_var
        dic_feature["traffic_movement_pressure_queue_efficient_kalman"] = list(
            self.kalman_pressure_estimate.astype(np.float32)
        )
        dic_feature["lane_enter_running_part_ema"] = list(
            ema_alpha * raw_running + (1.0 - ema_alpha) * prev_running_ema
        )

        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        # IFDG: Ideal-Factual Distance Gap — sum of (1 - v_actual/v_max) over all
        # entering-lane vehicles. Proven to be an unbiased estimator of ATT.
        # v_max = 11.11 m/s (≈40 km/h, CityFlow default road speed limit).
        V_MAX = 11.11
        ifdg = 0.0
        for lane in self.list_entering_lanes:
            for vehicle in self.dic_lane_vehicle_current_step.get(lane, []):
                if "shadow" in vehicle:
                    continue
                v_actual = self.dic_vehicle_speed_current_step.get(vehicle, 0.0)
                ifdg += (V_MAX - min(v_actual, V_MAX)) / V_MAX
        dic_feature["ifdg"] = ifdg

        dic_feature["num_in_seg_attend"] = self._orgnize_several_segments_attend(dic_feature["lane_num_waiting_vehicle_in"],
                                                                                 dic_feature["lane_num_waiting_vehicle_out"])
        self.dic_feature = dic_feature

    def _orgnize_several_segments_attend(self, queue_in, queue_out):
        part1, part2, part3 = self._get_several_segments_attend(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                                vehicle_distance=self.dic_vehicle_distance_current_step,
                                                                vehicle_speed=self.dic_vehicle_speed_current_step,
                                                                lane_length=self.lane_length,
                                                                list_lanes=self.list_lanes)
        run_in_part1 = [float(len(part1[lane])) for lane in self.list_entering_lanes]
        run_in_part2 = [float(len(part2[lane])) for lane in self.list_entering_lanes]
        run_in_part3 = [float(len(part3[lane])) for lane in self.list_entering_lanes]

        run_out_part1 = [float(len(part1[lane])) for lane in self.list_exiting_lanes]
        run_out_part2 = [float(len(part2[lane]))for lane in self.list_exiting_lanes]
        run_out_part3 = [float(len(part3[lane])) for lane in self.list_exiting_lanes]

        total_in, total_out = [], []
        for i in range(12):
            total_in.extend([run_in_part1[i], run_in_part2[i], run_in_part3[i], queue_in[i]])
            total_out.extend([run_out_part1[i], run_out_part2[i], run_out_part3[i], queue_out[i]])
        return total_in + total_out

    def _get_several_segments_attend(self, lane_vehicles, vehicle_distance, vehicle_speed,
                                           lane_length, list_lanes):
        obs_length = 100
        part1, part2, part3 = {}, {}, {}
        for lane in list_lanes:
            part1[lane], part2[lane], part3[lane] = [], [], []
            for vehicle in lane_vehicles[lane]:
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                    continue
                if vehicle_speed[vehicle] > 0.1:
                    temp_v_distance = vehicle_distance[vehicle]
                    if temp_v_distance > lane_length[lane] - obs_length:
                        part1[lane].append(vehicle)
                    elif lane_length[lane] - 2 * obs_length < temp_v_distance <= lane_length[lane] - obs_length:
                        part2[lane].append(vehicle)
                    elif lane_length[lane] - 3 * obs_length < temp_v_distance <= lane_length[lane] - 2 * obs_length:
                        part3[lane].append(vehicle)
        return part1, part2, part3

    @staticmethod
    def _get_traffic_movement_pressure_general(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] for j in range(12)]
        return t_m_p

    @staticmethod
    def _get_traffic_movement_pressure_efficient(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]]/3 for j in range(12)]
        return t_m_p

    def _get_part_traffic_movement_features(self):
        """
        return: part_traffic_movement_pressure_num:     both the end and the beginning of the lane
                part_patrric_movement_pressure_queue:   all at the end of the road
                part_entering_running_vehicles:         part obs of the running vehicles
        """
        f_p_num, l_p_num, l_p_q = self._get_part_observations(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                              vehicle_distance=self.dic_vehicle_distance_current_step,
                                                              vehicle_speed=self.dic_vehicle_speed_current_step,
                                                              lane_length=self.lane_length,
                                                              obs_length=self.obs_length,
                                                              list_lanes=self.list_lanes)
        """calculate traffic_movement_pressure with part queue"""
        list_entering_part_queue = [len(l_p_q[lane]) for lane in self.list_entering_lanes]
        list_exiting_part_queue = [len(l_p_q[lane]) for lane in self.list_exiting_lanes]
        tmp_queue_efficient_part = self._get_traffic_movement_pressure_efficient(list_entering_part_queue,
                                                                                 list_exiting_part_queue)
        tmp_queue_part = self._get_traffic_movement_pressure_general(list_entering_part_queue,
                                                                     list_exiting_part_queue)

        """calculate traffic_movement_pressure with part num vehicle"""
        # entering
        list_entering_num_f = [len(f_p_num[lane]) for lane in self.list_entering_lanes]
        list_entering_num_l = [len(l_p_num[lane]) for lane in self.list_entering_lanes]
        entering_num = np.array(list_entering_num_f) + np.array(list_entering_num_l)
        # exiting
        list_exiting_num_f = [len(f_p_num[lane]) for lane in self.list_exiting_lanes]
        list_exiting_num_l = [len(l_p_num[lane]) for lane in self.list_exiting_lanes]
        exiting_num = np.array(list_exiting_num_f) + np.array(list_exiting_num_l)
        traffic_movement_pressure_nums = self._get_traffic_movement_pressure_general(entering_num, exiting_num)
        # part of entering running vehicles, all at the end of the road
        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)

        return (
            traffic_movement_pressure_nums,
            tmp_queue_part,
            tmp_queue_efficient_part,
            part_entering_running,
            list_entering_part_queue,
            list_entering_num_l,
        )

    @staticmethod
    def _get_part_observations(lane_vehicles, vehicle_distance, vehicle_speed,
                               lane_length, obs_length, list_lanes):
        """
            Input: lane_vehicles :      Dict{lane_id    :   [vehicle_ids]}
                   vehicle_distance:    Dict{vehicle_id :   float(dist)}
                   vehicle_speed:       Dict{vehicle_id :   float(speed)}
                   lane_length  :       Dict{lane_id    :   float(length)}
                   obs_length   :       The part observation length
                   list_lanes   :       List[lane_ids at the intersection]
        :return:
                    part_vehicles:      Dict{ lane_id, [vehicle_ids]}
        """
        # get vehicle_ids and speeds
        first_part_num_vehicle = {}
        first_part_queue_vehicle = {}  # useless, at the begin of lane, there is no waiting vechiles
        last_part_num_vehicle = {}
        last_part_queue_vehicle = {}

        for lane in list_lanes:
            first_part_num_vehicle[lane] = []
            first_part_queue_vehicle[lane] = []
            last_part_num_vehicle[lane] = []
            last_part_queue_vehicle[lane] = []
            last_part_obs_length = lane_length[lane] - obs_length
            for vehicle in lane_vehicles[lane]:
                """ get the first part of obs
                    That is vehicle_distance <= obs_length 
                """
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        first_part_queue_vehicle[lane].append(vehicle)

                """ get the last part of obs
                    That is  lane_length-obs_length <= vehicle_distance <= lane_length 
                """
                if temp_v_distance >= last_part_obs_length:
                    last_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle_entring(self):
        """
        vehicle number for each lane
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_entering_lanes]

    def _get_lane_num_vehicle_downstream(self):
        """
        vehicle number for each lane, exiting
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_exiting_lanes]

    def _lane_capacity(self, lane):
        lane_length = float(self.lane_length.get(lane, 0.0))
        return max(lane_length / max(self.avg_vehicle_length, 1.0), 1.0)

    def _normalize_entering_by_capacity(self, values, signed=False):
        normalized = []
        for lane, value in zip(self.list_entering_lanes, values):
            cap = self._lane_capacity(lane)
            v = float(value) / cap
            if signed:
                normalized.append(float(np.clip(v, -1.0, 1.0)))
            else:
                normalized.append(float(np.clip(v, 0.0, 1.0)))
        return normalized

    def _stopped_vehicle_set_for_lane(self, lane, lane_vehicle_dic, speed_dic):
        stopped = set()
        for vehicle in lane_vehicle_dic.get(lane, []):
            if "shadow" in vehicle:
                continue
            if speed_dic.get(vehicle, 0.0) <= self.stop_speed_threshold:
                stopped.add(vehicle)
        return stopped

    def _update_vehicle_waiting_times(self):
        current_vehicle_ids = set()
        for vehicle, speed in self.dic_vehicle_speed_current_step.items():
            if "shadow" in vehicle:
                continue
            current_vehicle_ids.add(vehicle)
            if speed <= self.stop_speed_threshold:
                self.dic_vehicle_waiting_time[vehicle] = self.dic_vehicle_waiting_time.get(vehicle, 0.0) + self.wait_time_increment
            else:
                self.dic_vehicle_waiting_time[vehicle] = 0.0
        for vehicle in list(self.dic_vehicle_waiting_time.keys()):
            if vehicle not in current_vehicle_ids:
                self.dic_vehicle_waiting_time.pop(vehicle, None)

    def _get_lane_far_approaching_part(self):
        """
        Count running vehicles on each entering lane within (L, 2L] from the stop line,
        where L = vmax * action_time.
        """
        action_horizon = self.max_vehicle_speed * float(self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        far_counts = []
        for lane in self.list_entering_lanes:
            lane_length = float(self.lane_length.get(lane, 0.0))
            near_cut = lane_length - action_horizon
            far_cut = lane_length - 2.0 * action_horizon
            if near_cut <= 0.0:
                far_counts.append(0.0)
                continue
            cnt = 0.0
            for vehicle in self.dic_lane_vehicle_current_step.get(lane, []):
                if "shadow" in vehicle:
                    continue
                speed = self.dic_vehicle_speed_current_step.get(vehicle, 0.0)
                if speed <= self.stop_speed_threshold:
                    continue
                dist = self.dic_vehicle_distance_current_step.get(vehicle, 0.0)
                if far_cut < dist <= near_cut:
                    cnt += 1.0
            far_counts.append(cnt)
        return far_counts

    def _get_lane_average_speed_effective(self):
        """
        Lane Average Speed (LAS) within the effective range L from the stop line.
        For each entering lane, average the speeds of vehicles whose distance lies in
        (lane_length - L, lane_length]. When the effective range is empty, return 0.
        """
        action_horizon = self.max_vehicle_speed * float(self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        las = []
        for lane in self.list_entering_lanes:
            lane_length = float(self.lane_length.get(lane, 0.0))
            near_cut = lane_length - action_horizon
            speeds = []
            for vehicle in self.dic_lane_vehicle_current_step.get(lane, []):
                if "shadow" in vehicle:
                    continue
                dist = float(self.dic_vehicle_distance_current_step.get(vehicle, 0.0))
                if dist <= near_cut:
                    continue
                speeds.append(float(self.dic_vehicle_speed_current_step.get(vehicle, 0.0)))
            las.append(float(np.mean(speeds)) if speeds else 0.0)
        return las

    def _get_downstream_saturation_movement(self, exiting_waiting):
        """
        Map downstream queue saturation back to 12 movement slots.
        Each movement uses the average saturation of the target exiting approach.
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11],
        }
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]

        approach_sat = {}
        for approach in list_approachs:
            lane_indices = index_maps[approach]
            sat_values = []
            for idx in lane_indices:
                lane = self.list_exiting_lanes[idx]
                cap = self._lane_capacity(lane)
                sat = float(exiting_waiting[idx]) / cap
                sat_values.append(min(max(sat, 0.0), 1.0))
            approach_sat[approach] = float(np.mean(sat_values)) if sat_values else 0.0

        return [approach_sat[turn_maps[j]] for j in range(12)]

    def _get_queue_growth_decay_movement(self, current_waiting):
        qgr = []
        qdr = []
        for lane, current_wait in zip(self.list_entering_lanes, current_waiting):
            history = self.lane_queue_count_history[lane]
            history.append(float(current_wait))
            
            # --- QDSE Logic: Real N_in and N_out caching ---
            veh_history = self.lane_vehicle_history[lane]
            veh_history.append(list(self.dic_lane_vehicle_current_step.get(lane, [])))
            
            curr_veh_list = veh_history[-1] if len(veh_history) > 0 else []
            pre_veh_list = veh_history[0] if len(veh_history) > 0 else curr_veh_list
            
            # Remove shadows generated by CityFlow
            curr_veh_clean = [v for v in curr_veh_list if "shadow" not in v]
            pre_veh_clean = [v for v in pre_veh_list if "shadow" not in v]

            # Set difference perfectly identifies arrived and left vehicles
            n_in = len(set(curr_veh_clean) - set(pre_veh_clean))
            n_out = len(set(pre_veh_clean) - set(curr_veh_clean))

            # Normalize by trend_window_steps (approx max flow over the window is 1 veh/sec)
            max_val = max(1.0, float(self.trend_window_steps))
            qgr.append(min(max(float(n_in) / max_val, 0.0), 1.0))
            qdr.append(min(max(float(n_out) / max_val, 0.0), 1.0))

        return qgr, qdr

    def _get_weighted_accumulated_wait_movement(self):
        waw = []
        for lane in self.list_entering_lanes:
            stopped = self._stopped_vehicle_set_for_lane(
                lane, self.dic_lane_vehicle_current_step, self.dic_vehicle_speed_current_step
            )
            if not stopped:
                waw.append(0.0)
                continue
            avg_wait = float(np.mean([self.dic_vehicle_waiting_time.get(v, 0.0) for v in stopped]))
            waw.append(min(max(avg_wait / max(self.max_wait_time, 1.0), 0.0), 1.0))
        return waw

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = dict()
        for state_feature_name in list_state_features:
            if state_feature_name == "downstream_congestion":
                dic_state[state_feature_name] = float(
                    np.sum(self.dic_feature.get("lane_num_vehicle_downstream", np.zeros(12)))
                )
            elif state_feature_name == "cur_phase_previous_step":
                dic_state[state_feature_name] = [self.previous_phase_index]
            elif state_feature_name.endswith("_previous_step"):
                base_name = state_feature_name[: -len("_previous_step")]
                if base_name in self.dic_feature_previous_step:
                    dic_state[state_feature_name] = self.dic_feature_previous_step[base_name]
                else:
                    dic_state[state_feature_name] = self.dic_feature.get(base_name, [0.0] * 12)
            else:
                dic_state[state_feature_name] = self.dic_feature[state_feature_name]
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        # dic_reward["sum_lane_queue_length"] = None
        waiting = np.asarray(self.dic_feature["lane_num_waiting_vehicle_in"], dtype=np.float32)
        if waiting.size > 0:
            waiting_sorted = np.sort(waiting)[::-1]
            queue_top2_mean = float(np.mean(waiting_sorted[: min(2, waiting_sorted.size)]))
            queue_top3_mean = float(np.mean(waiting_sorted[: min(3, waiting_sorted.size)]))
            queue_pnorm = float(np.linalg.norm(waiting, ord=4))
        else:
            queue_top2_mean = 0.0
            queue_top3_mean = 0.0
            queue_pnorm = 0.0
        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"]))
        advanced_pressure_list = (
            np.asarray(self.dic_feature["traffic_movement_pressure_queue_efficient"], dtype=np.float32)
            + np.asarray(self.dic_feature["lane_enter_running_part"], dtype=np.float32)
        )
        dic_reward["pressure"] = float(np.sum(np.abs(np.asarray(self.dic_feature["pressure"], dtype=np.float32))))
        dic_reward["advanced_pressure"] = float(np.sum(np.abs(advanced_pressure_list)))
        dic_reward["queue_length"] = np.absolute(np.sum(self.dic_feature["lane_num_waiting_vehicle_in"]))
        dic_reward["regional_queue"] = float(
            np.absolute(
                np.sum(self.dic_feature["lane_num_waiting_vehicle_in"])
                + np.sum(self.dic_feature["lane_num_waiting_vehicle_out"])
            )
        )
        dic_reward["queue_max"] = float(np.max(waiting)) if waiting.size > 0 else 0.0
        dic_reward["queue_balance"] = float(np.std(waiting)) if waiting.size > 0 else 0.0
        dic_reward["queue_top2_mean"] = queue_top2_mean
        dic_reward["queue_top3_mean"] = queue_top3_mean
        dic_reward["queue_pnorm"] = queue_pnorm
        dic_reward["global_queue_length"] = dic_reward["queue_length"]
        dic_reward["downstream_congestion"] = float(
            np.sum(self.dic_feature.get("lane_num_vehicle_downstream", np.zeros(12)))
        )
        dic_reward["ifdg"] = float(self.dic_feature.get("ifdg", 0.0))
        dic_reward["switch_penalty"] = 0.0
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class CityFlowEnv:

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            """ include the yellow time in action time """
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")
        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": int(self.dic_traffic_env_conf.get("SEED", 0)),
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",
            "replayLogFile": "frontend/web/replayLogFile.txt"
        }
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # get lane length
        _, self.lane_length = self.get_lane_length()
        self._attach_topology_vectors()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                               self.path_to_log,
                                               self.lane_length)
                                  for i in range(self.dic_traffic_env_conf["NUM_COL"])
                                  for j in range(self.dic_traffic_env_conf["NUM_ROW"])]
        self.list_inter_log = [[] for _ in range(self.dic_traffic_env_conf["NUM_COL"] *
                                                 self.dic_traffic_env_conf["NUM_ROW"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_COL"]):
            for j in range(self.dic_traffic_env_conf["NUM_ROW"]):
                self.id_to_index["intersection_{0}_{1}".format(i+1, j+1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        state, done = self.get_state()
        return state

    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if i == 0:
                print("time: {0}".format(instant_time))
                    
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = False
        return list_state, done

    def get_reward(self):
        reward_info = self.dic_traffic_env_conf["DIC_REWARD_INFO"]
        list_reward = [inter.get_reward(reward_info) for inter in self.list_intersection]
        if reward_info.get("global_queue_length", 0) != 0:
            global_queue = float(
                np.sum(
                    [
                        np.sum(inter.dic_feature["lane_num_waiting_vehicle_in"])
                        for inter in self.list_intersection
                    ]
                )
            )
            global_reward = float(reward_info["global_queue_length"]) * global_queue
            list_reward = [global_reward for _ in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
            
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            non_virtual_ids = set()
            for inter in net["intersections"]:
                if not inter["virtual"]:
                    non_virtual_ids.add(inter["id"])
                    traffic_light_node_dict[inter["id"]] = {"location": {"x": float(inter["point"]["x"]),
                                                                         "y": float(inter["point"]["y"])},
                                                            "total_inter_num": None, "adjacency_row": None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            inter_graph = {}
            weighted_graph = {}
            max_lane_speed = 0.0

            def _polyline_length(points):
                if not points or len(points) < 2:
                    return 0.0
                total = 0.0
                for idx in range(len(points) - 1):
                    p1 = points[idx]
                    p2 = points[idx + 1]
                    dx = float(p1["x"]) - float(p2["x"])
                    dy = float(p1["y"]) - float(p2["y"])
                    total += float(np.sqrt(dx * dx + dy * dy))
                return total

            for road in net["roads"]:
                if road["id"] not in edge_id_dict.keys():
                    edge_id_dict[road["id"]] = {}
                edge_id_dict[road["id"]]["from"] = road["startIntersection"]
                edge_id_dict[road["id"]]["to"] = road["endIntersection"]
                for lane in road.get("lanes", []):
                    max_lane_speed = max(max_lane_speed, float(lane.get("maxSpeed", 0.0)))
                start = road["startIntersection"]
                end = road["endIntersection"]
                if start in non_virtual_ids and end in non_virtual_ids:
                    inter_graph.setdefault(start, set()).add(end)
                    inter_graph.setdefault(end, set()).add(start)
                    road_len = _polyline_length(road.get("points", []))
                    if road_len <= 0.0:
                        start_loc = traffic_light_node_dict[start]["location"]
                        end_loc = traffic_light_node_dict[end]["location"]
                        road_len = self._cal_distance(start_loc, end_loc)
                    weighted_graph.setdefault(start, {})[end] = min(
                        float(road_len),
                        weighted_graph.get(start, {}).get(end, float("inf")),
                    )
                    weighted_graph.setdefault(end, {})[start] = min(
                        float(road_len),
                        weighted_graph.get(end, {}).get(start, float("inf")),
                    )

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            hop_enabled = bool(self.dic_traffic_env_conf.get("ADJ_MASK_BY_HOP", False))
            max_hop = int(self.dic_traffic_env_conf.get("MAX_HOP_DISTANCE", -1))
            distance_topk_mode = bool(self.dic_traffic_env_conf.get("DISTANCE_TOPK_MODE", False))
            distance_topk_k = int(self.dic_traffic_env_conf.get("DISTANCE_TOPK_K", top_k))
            static_delay_mode = bool(self.dic_traffic_env_conf.get("STATIC_DELAY_CANDIDATE_MODE", False))
            static_delay_multiplier = float(self.dic_traffic_env_conf.get("STATIC_DELAY_MULTIPLIER", 1.0))
            static_delay_rmax = int(self.dic_traffic_env_conf.get("STATIC_DELAY_CANDIDATE_RMAX", top_k))
            static_delay_min_external = max(0, int(self.dic_traffic_env_conf.get("STATIC_DELAY_MIN_EXTERNAL", 0)))
            static_delay_use_shortest_path = bool(self.dic_traffic_env_conf.get("STATIC_DELAY_USE_SHORTEST_PATH", True))
            static_delay_padding = str(self.dic_traffic_env_conf.get("STATIC_DELAY_PADDING", "self"))
            mask_farthest_count = max(0, int(self.dic_traffic_env_conf.get("MASK_FARTHEST_COUNT", 0)))
            max_vehicle_speed = float(max_lane_speed) if max_lane_speed > 0.0 else float(
                self.dic_traffic_env_conf.get("MAX_VEHICLE_SPEED", 11.11)
            )
            delay_threshold = static_delay_multiplier * float(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15))

            def shortest_path_lengths(src):
                dist = {src: 0.0}
                heap = [(0.0, src)]
                while heap:
                    cur_dist, cur = heapq.heappop(heap)
                    if cur_dist > dist.get(cur, float("inf")):
                        continue
                    for nxt, weight in weighted_graph.get(cur, {}).items():
                        nd = cur_dist + float(weight)
                        if nd < dist.get(nxt, float("inf")):
                            dist[nxt] = nd
                            heapq.heappush(heap, (nd, nxt))
                return dist

            shortest_path_cache = {}

            def bfs_hop(src):
                dist = {src: 0}
                queue = [src]
                head = 0
                while head < len(queue):
                    cur = queue[head]
                    head += 1
                    cur_dist = dist[cur]
                    if hop_enabled and max_hop >= 0 and cur_dist >= max_hop:
                        continue
                    for nxt in inter_graph.get(cur, []):
                        if nxt not in dist:
                            dist[nxt] = cur_dist + 1
                            queue.append(nxt)
                return dist

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]["location"]

                row = np.array([0]*total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]["location"]
                    dist = self._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if distance_topk_mode:
                    k_total = max(1, distance_topk_k)
                    keep_external = max(0, (k_total - 1) - mask_farthest_count)
                    external_sorted = sorted(
                        [
                            inter_id_to_index[j]
                            for j in traffic_light_node_dict.keys()
                            if j != i
                        ],
                        key=lambda idx: row[idx]
                    )
                    adjacency_row_unsorted = external_sorted[:keep_external]
                    while len(adjacency_row_unsorted) < max(0, k_total - 1):
                        adjacency_row_unsorted.append(inter_id_to_index[i])
                elif hop_enabled and max_hop >= 0:
                    hop_dist = bfs_hop(i)
                    allowed = [
                        inter_id_to_index[j]
                        for j in traffic_light_node_dict.keys()
                        if j != i and hop_dist.get(j, max_hop + 1) <= max_hop
                    ]
                    allowed = sorted(allowed, key=lambda idx: row[idx])
                    need = max(0, top_k - 1)
                    adjacency_row_unsorted = allowed[:need]
                    if len(adjacency_row_unsorted) < need:
                        adjacency_row_unsorted.extend([inter_id_to_index[i]] * (need - len(adjacency_row_unsorted)))
                elif static_delay_mode:
                    if i not in shortest_path_cache:
                        shortest_path_cache[i] = shortest_path_lengths(i) if static_delay_use_shortest_path else {i: 0.0}
                    shortest_paths = shortest_path_cache[i]
                    external_limit = max(0, int(static_delay_rmax) - 1)
                    allowed = []
                    nearest_all = []
                    for j in traffic_light_node_dict.keys():
                        if j == i:
                            continue
                        if j not in shortest_paths:
                            continue
                        nearest_all.append((float(shortest_paths[j]), inter_id_to_index[j]))
                        travel_time = float(shortest_paths[j]) / max(max_vehicle_speed, 1e-6)
                        if travel_time <= delay_threshold:
                            allowed.append((float(shortest_paths[j]), inter_id_to_index[j]))
                    allowed.sort(key=lambda item: item[0])
                    nearest_all.sort(key=lambda item: item[0])
                    adjacency_row_unsorted = [idx for _, idx in allowed[:external_limit]]
                    if len(adjacency_row_unsorted) < min(external_limit, static_delay_min_external):
                        existing = set(adjacency_row_unsorted)
                        for _, idx in nearest_all:
                            if idx in existing:
                                continue
                            adjacency_row_unsorted.append(idx)
                            existing.add(idx)
                            if len(adjacency_row_unsorted) >= min(external_limit, static_delay_min_external):
                                break
                    pad_value = inter_id_to_index[i]
                    if static_delay_padding != "self" and adjacency_row_unsorted:
                        pad_value = adjacency_row_unsorted[-1]
                    if len(adjacency_row_unsorted) < external_limit:
                        adjacency_row_unsorted.extend([pad_value] * (external_limit - len(adjacency_row_unsorted)))
                elif len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                if not (distance_topk_mode or (hop_enabled and max_hop >= 0) or static_delay_mode):
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]["adjacency_row"] = [inter_id_to_index[i]]+adjacency_row_unsorted
                traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

            if static_delay_mode:
                candidate_counts = []
                for inter_id, payload in traffic_light_node_dict.items():
                    row = payload.get("adjacency_row", [])
                    unique_ext = len({idx for idx in row[1:] if idx != row[0]})
                    candidate_counts.append(unique_ext)
                if candidate_counts:
                    print(
                        "[StaticDelayCandidates] vmax={:.3f}m/s threshold={:.3f}s rmax={} avg={:.2f} min={} max={}".format(
                            max_vehicle_speed,
                            delay_threshold,
                            static_delay_rmax,
                            float(np.mean(candidate_counts)),
                            int(np.min(candidate_counts)),
                            int(np.max(candidate_counts)),
                        )
                    )

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
                traffic_light_node_dict[i]["neighbor_ENWS"] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    if edge_id_dict[road_id]["to"] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                    else:
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(edge_id_dict[road_id]["to"])

        return traffic_light_node_dict

    def _attach_topology_vectors(self):
        if not self.traffic_light_node_dict or self.lane_length is None:
            return

        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)

        non_virtual_ids = set(self.traffic_light_node_dict.keys())
        incoming_deg = {iid: 0 for iid in non_virtual_ids}
        outgoing_deg = {iid: 0 for iid in non_virtual_ids}
        in_lengths = {iid: [] for iid in non_virtual_ids}
        out_lengths = {iid: [] for iid in non_virtual_ids}

        for road in net["roads"]:
            start = road["startIntersection"]
            end = road["endIntersection"]
            if start not in non_virtual_ids or end not in non_virtual_ids:
                continue
            road_points = road["points"]
            road_length = abs(
                road_points[0]["x"] + road_points[0]["y"] - road_points[1]["x"] - road_points[1]["y"]
            )
            outgoing_deg[start] += 1
            incoming_deg[end] += 1
            out_lengths[start].append(float(road_length))
            in_lengths[end].append(float(road_length))

        xs = [self.traffic_light_node_dict[i]["location"]["x"] for i in non_virtual_ids]
        ys = [self.traffic_light_node_dict[i]["location"]["y"] for i in non_virtual_ids]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_span = max(max_x - min_x, 1e-6)
        y_span = max(max_y - min_y, 1e-6)

        max_in_deg = max(max(incoming_deg.values()), 1)
        max_out_deg = max(max(outgoing_deg.values()), 1)
        lane_lengths = np.asarray(list(self.lane_length.values()), dtype=np.float32)
        mean_lane_scale = float(np.mean(lane_lengths)) if lane_lengths.size > 0 else 1.0
        mean_lane_scale = max(mean_lane_scale, 1e-6)

        for inter_id, node in self.traffic_light_node_dict.items():
            x = float(node["location"]["x"])
            y = float(node["location"]["y"])
            x_norm = (x - min_x) / x_span
            y_norm = (y - min_y) / y_span
            in_deg_norm = float(incoming_deg.get(inter_id, 0)) / float(max_in_deg)
            out_deg_norm = float(outgoing_deg.get(inter_id, 0)) / float(max_out_deg)
            mean_in_len = float(np.mean(in_lengths.get(inter_id, []) or [0.0])) / mean_lane_scale
            mean_out_len = float(np.mean(out_lengths.get(inter_id, []) or [0.0])) / mean_lane_scale
            is_boundary = 1.0 if (x_norm < 1e-6 or x_norm > 1.0 - 1e-6 or y_norm < 1e-6 or y_norm > 1.0 - 1e-6) else 0.0
            is_arterial_like = 1.0 if max(incoming_deg.get(inter_id, 0), outgoing_deg.get(inter_id, 0)) <= 2 else 0.0
            node["topology_vector"] = [
                float(x_norm),
                float(y_norm),
                float(in_deg_norm),
                float(out_deg_norm),
                float(mean_in_len),
                float(mean_out_len),
                float(is_boundary),
                float(is_arterial_like),
            ]

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for i in range(3):
                lane_id = road['id'] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict
