import copy
import os
import pickle
import random
import sys
from typing import Dict, List

import numpy as np


def _ensure_coslight_onpolicy_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    coslight_pkg = os.path.join(repo_root, "CoSLight", "CoSLight")
    if coslight_pkg not in sys.path:
        sys.path.insert(0, coslight_pkg)


_ensure_coslight_onpolicy_path()
from onpolicy.envs.sumo_files_marl.env.sim_env import TSCSimulator


class SUMOEnv:
    """
    SUMO wrapper with the same high-level interface as CityFlowEnv used by pipeline.
    It keeps training/test logging format compatible with ConstructSample/summary.py.
    """

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        os.makedirs(self.path_to_log, exist_ok=True)

        self.sim = None
        self.current_time = 0.0
        self._raw_time_offset = 0.0
        self.list_inter_log = []
        self._features = []
        self._states = []

        self._tls: List[str] = []
        self._tl_to_idx: Dict[str, int] = {}
        self._adjacency_rows: List[List[int]] = []

        self._last_agent_action: List[int] = []
        self._phase_elapsed: List[float] = []
        self._prev_intersection_vehicles: List[set] = []
        self._vehicle_arrive_leave: List[Dict[str, Dict[str, float]]] = []

        self.min_action_time = int(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15))
        self.yellow_time = int(self.dic_traffic_env_conf.get("YELLOW_TIME", 5))

    def _resolve_sumocfg(self):
        sumocfg = self.dic_traffic_env_conf.get("SUMO_CFG_FILE")
        if not sumocfg:
            raise ValueError("SUMO_CFG_FILE is required when SIMULATOR=sumo")
        if not os.path.isabs(sumocfg):
            sumocfg = os.path.join(self.path_to_work_directory, sumocfg)
        sumocfg = os.path.abspath(sumocfg)
        if not os.path.exists(sumocfg):
            raise FileNotFoundError("SUMO_CFG_FILE not found: {}".format(sumocfg))
        return sumocfg

    def _build_sim_config(self):
        iter_duration = self.min_action_time - self.yellow_time
        if iter_duration <= 0:
            raise ValueError("MIN_ACTION_TIME must be > YELLOW_TIME for SUMOEnv")
        record_tripinfo = bool(self.dic_traffic_env_conf.get("SUMO_RECORD_TRIPINFO", False))

        return {
            "name": str(self.dic_traffic_env_conf.get("SUMO_NAME", "sumo_exp")),
            "seed": int(self.dic_traffic_env_conf.get("seed", self.dic_traffic_env_conf.get("SEED", 0))),
            "agent": self.dic_traffic_env_conf.get("MODEL_NAME", "AdvancedColight"),
            "is_libsumo": bool(self.dic_traffic_env_conf.get("SUMO_USE_LIBSUMO", True)),
            "yellow_duration": self.yellow_time,
            "iter_duration": iter_duration,
            "episode_length_time": int(self.dic_traffic_env_conf.get("RUN_COUNTS", 3600)),
            "gui": bool(self.dic_traffic_env_conf.get("SUMO_GUI", False)),
            "is_record": record_tripinfo,
            "output_path": self.path_to_log + os.sep if record_tripinfo else None,
            "action_type": "select_phase",
            "state_key": ["car_num", "queue_length", "stop_car_num", "pressure"],
            "reward_type": ["queue_len"],
            "is_neighbor_reward": False,
            "adjacency_top_k": int(self.dic_traffic_env_conf.get("TOP_K_ADJACENCY", 5)),
            "sumocfg_file": self._resolve_sumocfg(),
        }

    def _to_env_time(self, raw_time):
        return float(raw_time) - float(self._raw_time_offset)

    def _get_action_mapping(self, num_actions):
        mapping = self.dic_traffic_env_conf.get("SUMO_ACTION_MAPPING")
        if mapping:
            mapping = [int(x) for x in mapping]
        else:
            if num_actions <= 4:
                mapping = [0, 2, 4, 6][:num_actions]
            else:
                mapping = list(range(num_actions))
        if len(mapping) < num_actions:
            mapping.extend([mapping[-1]] * (num_actions - len(mapping)))
        return mapping

    @staticmethod
    def _expand8_to_12(v8):
        # input order from CoSLight state is [N_l,N_s,W_l,W_s,S_l,S_s,E_l,E_s]
        n_l, n_s, w_l, w_s, s_l, s_s, e_l, e_s = v8
        return [
            float(w_l), float(w_s), float(w_s),
            float(e_l), float(e_s), float(e_s),
            float(n_l), float(n_s), float(n_s),
            float(s_l), float(s_s), float(s_s),
        ]

    @staticmethod
    def _safe_get(state_list, key):
        out = []
        for d in state_list:
            try:
                out.append(float(d.get(key, 0.0)))
            except Exception:
                out.append(0.0)
        while len(out) < 8:
            out.append(0.0)
        return out[:8]

    def _build_adjacency_rows(self, obs_raw):
        num_intersections = len(self._tls)
        top_k = min(int(self.dic_traffic_env_conf.get("TOP_K_ADJACENCY", 5)), num_intersections)
        rows = []
        for idx, tl in enumerate(self._tls):
            near = obs_raw.get(tl, (None, None, None))[2]
            candidates = []
            if near and len(near) > 0 and len(near[0]) > 0:
                for nei in near[0][0]:
                    try:
                        nei_i = int(nei)
                    except Exception:
                        continue
                    if 0 <= nei_i < num_intersections and nei_i != idx:
                        if nei_i not in candidates:
                            candidates.append(nei_i)
            row = [idx] + candidates
            row = row[:top_k]
            while len(row) < top_k:
                row.append(idx)
            rows.append(row)
        return rows

    def _capture_intersection_vehicles(self):
        all_sets = []
        for tl in self._tls:
            cross = self.sim._crosses[tl]
            ids = set()
            for lane in cross._incoming_lanes:
                try:
                    vids = self.sim.sim.lane.getLastStepVehicleIDs(lane)
                except Exception:
                    vids = cross._lane_vehicle_dict.get(lane, [])
                ids.update(vids)
            all_sets.append(ids)
        return all_sets

    def _init_vehicle_tracking(self):
        self._prev_intersection_vehicles = self._capture_intersection_vehicles()
        self._vehicle_arrive_leave = [dict() for _ in self._tls]
        for i, s in enumerate(self._prev_intersection_vehicles):
            for vid in s:
                self._vehicle_arrive_leave[i][vid] = {
                    "enter_time": float(self.current_time),
                    "leave_time": np.nan,
                }

    def _update_vehicle_tracking(self):
        cur_sets = self._capture_intersection_vehicles()
        ts = float(self.current_time)
        for i in range(len(self._tls)):
            prev = self._prev_intersection_vehicles[i]
            cur = cur_sets[i]
            arrive = cur - prev
            leave = prev - cur
            rec = self._vehicle_arrive_leave[i]
            for vid in arrive:
                if vid not in rec:
                    rec[vid] = {"enter_time": ts, "leave_time": np.nan}
            for vid in leave:
                if vid in rec and np.isnan(rec[vid]["leave_time"]):
                    rec[vid]["leave_time"] = ts
            self._prev_intersection_vehicles[i] = cur

    def _build_feature_from_raw(self, obs_raw):
        feature_list = []
        for idx, tl in enumerate(self._tls):
            state_main = obs_raw.get(tl, ([], None, None))[0]
            q8 = self._safe_get(state_main, "queue_length")
            car8 = self._safe_get(state_main, "car_num")
            stop8 = self._safe_get(state_main, "stop_car_num")
            pressure8 = self._safe_get(state_main, "pressure")
            run8 = [max(c - s, 0.0) for c, s in zip(car8, stop8)]

            q12 = self._expand8_to_12(q8)
            car12 = self._expand8_to_12(car8)
            run12 = self._expand8_to_12(run8)
            pressure12 = self._expand8_to_12(pressure8)

            feat = {
                "cur_phase": [int(self._last_agent_action[idx]) + 1],
                "time_this_phase": [float(self._phase_elapsed[idx])],
                "phase_elapsed": [float(self._phase_elapsed[idx])],
                "lane_num_vehicle": car12,
                "lane_num_waiting_vehicle_in": q12,
                "traffic_movement_pressure_queue_efficient": pressure12,
                "lane_enter_running_part": run12,
                "downstream_congestion": [float(np.mean(q8))],
                "pressure": pressure12,
                "adjacency_matrix": self._adjacency_rows[idx],
            }
            feature_list.append(feat)

        return feature_list

    def _select_state(self):
        state_keys = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        return [{k: feat[k] for k in state_keys} for feat in self._features]

    def _compute_reward(self):
        reward_info = self.dic_traffic_env_conf.get("DIC_REWARD_INFO", {})
        queue_w = float(reward_info.get("queue_length", 0.0))
        pressure_w = float(reward_info.get("pressure", 0.0))

        rewards = []
        for feat in self._features:
            queue = float(np.sum(feat.get("lane_num_waiting_vehicle_in", [])))
            pressure = float(np.abs(np.sum(feat.get("pressure", []))))
            rewards.append(queue_w * queue + pressure_w * pressure)
        return rewards

    def _log_action_window(self, cur_time, before_action_feature, action):
        if not self.dic_traffic_env_conf.get("saveReplay", False):
            return

        for i in range(self.min_action_time):
            if i == 0:
                action_i = action
            else:
                action_i = [-1] * len(action)
            self.log(cur_time + i, before_action_feature, action_i)

    def _map_action(self, action):
        action_arr = np.asarray(action).astype(int).reshape(-1).tolist()
        num_intersections = len(self._tls)
        if len(action_arr) < num_intersections:
            action_arr.extend([0] * (num_intersections - len(action_arr)))
        action_arr = action_arr[:num_intersections]

        num_actions = len(self.dic_traffic_env_conf.get("PHASE", {}))
        num_actions = max(num_actions, 1)
        action_map = self._get_action_mapping(num_actions)

        tl_action = {}
        effective_action = []
        for idx, tl in enumerate(self._tls):
            a = int(action_arr[idx]) % num_actions
            cross = self.sim._crosses[tl]
            green_slots = len(cross.green_phases)
            pref_slot = int(action_map[a]) % green_slots
            unava = set(cross.unava_index)
            if pref_slot in unava:
                avail = [s for s in range(green_slots) if s not in unava]
                pref_slot = avail[0] if avail else 0

            raw_phase = cross.green_phases[pref_slot]
            tl_action[tl] = raw_phase

            eff_a = a
            for i, slot in enumerate(action_map[:num_actions]):
                if int(slot) % green_slots == pref_slot:
                    eff_a = i
                    break
            effective_action.append(eff_a)

        return tl_action, effective_action

    def _touch_inter_pickles(self):
        for inter_ind in range(len(self._tls)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            with open(path_to_log_file, "wb"):
                pass

    def reset(self):
        if self.sim is not None:
            try:
                self.sim.terminate()
            except Exception:
                pass

        cfg = self._build_sim_config()
        port_start = int(self.dic_traffic_env_conf.get("SUMO_PORT_START", 20000)) + random.randint(0, 500)
        self.sim = TSCSimulator(cfg, port_start)
        obs_raw = self.sim.reset()
        self._raw_time_offset = float(self.sim._current_time)
        self.current_time = 0.0

        self._tls = list(self.sim.all_tls)
        self._tl_to_idx = {tl: i for i, tl in enumerate(self._tls)}

        num_intersections = len(self._tls)
        # Keep conf aligned with true env size so loops and logging stay consistent.
        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = num_intersections

        self.list_inter_log = [[] for _ in range(num_intersections)]
        self._last_agent_action = [0 for _ in range(num_intersections)]
        self._phase_elapsed = [0.0 for _ in range(num_intersections)]
        self._adjacency_rows = self._build_adjacency_rows(obs_raw)
        self._touch_inter_pickles()

        self._features = self._build_feature_from_raw(obs_raw)
        self._states = self._select_state()
        self._init_vehicle_tracking()

        return self._states

    def step(self, action):
        before_action_feature = copy.deepcopy(self._features)
        cur_time = int(self.get_current_time())

        tl_action, effective_action = self._map_action(action)
        obs_raw, _, done, _ = self.sim.step(tl_action)
        self.current_time = self._to_env_time(self.sim._current_time)

        for i, a in enumerate(effective_action):
            if a == self._last_agent_action[i]:
                self._phase_elapsed[i] += float(self.min_action_time)
            else:
                self._phase_elapsed[i] = float(self.min_action_time)
            self._last_agent_action[i] = a

        self._features = self._build_feature_from_raw(obs_raw)
        self._states = self._select_state()
        reward = self._compute_reward()

        self._update_vehicle_tracking()
        self._log_action_window(cur_time, before_action_feature, effective_action)

        average_reward_action_list = reward
        return self._states, reward, done, average_reward_action_list

    def get_feature(self):
        return self._features

    def get_state(self):
        return self._states, False

    def get_current_time(self):
        return float(self.current_time)

    def log(self, cur_time, before_action_feature, action):
        if not self.dic_traffic_env_conf.get("saveReplay", False):
            return

        for inter_ind in range(len(self.list_inter_log)):
            self.list_inter_log[inter_ind].append(
                {
                    "time": int(cur_time),
                    "state": before_action_feature[inter_ind],
                    "action": int(action[inter_ind]),
                }
            )

    def batch_log_2(self):
        for inter_ind in range(len(self._tls)):
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self._vehicle_arrive_leave[inter_ind]

            with open(path_to_log_file, "w") as f:
                f.write("vehicle_id,enter_time,leave_time\n")
                for vid, times in dic_vehicle.items():
                    enter = float(times["enter_time"])
                    leave = float(times["leave_time"])
                    leave_str = str(leave) if not np.isnan(leave) else "nan"
                    f.write("{0},{1},{2}\n".format(vid, enter, leave_str))

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self._vehicle_arrive_leave[inter_ind]
            with open(path_to_log_file, "w") as f:
                f.write("vehicle_id,enter_time,leave_time\n")
                for vid, times in dic_vehicle.items():
                    enter = float(times["enter_time"])
                    leave = float(times["leave_time"])
                    leave_str = str(leave) if not np.isnan(leave) else "nan"
                    f.write("{0},{1},{2}\n".format(vid, enter, leave_str))

        for inter_ind in range(start, stop):
            try:
                path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
                with open(path_to_log_file, "wb") as f:
                    pickle.dump(self.list_inter_log[inter_ind], f)
            except Exception as e:
                print("Warning: Failed to pickle log for inter {}: {}".format(inter_ind, e))

    def bulk_log_multi_process(self, batch_size=100):
        del batch_size
        print("Logging {} intersections sequentially...".format(len(self._tls)))
        self.batch_log(0, len(self._tls))
        print("Logging finished.")

    def end_cityflow(self):
        try:
            if self.sim is not None:
                self.sim.terminate()
        except Exception:
            pass
        print("============== sumo process end ===============")
