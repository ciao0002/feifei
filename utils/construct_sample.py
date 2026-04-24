import numpy as np
import pickle
import os
import traceback


def get_reward_from_features(rs):
    waiting = np.asarray(rs["lane_num_waiting_vehicle_in"], dtype=np.float32)
    if waiting.size > 0:
        waiting_sorted = np.sort(waiting)[::-1]
        queue_top2_mean = float(np.mean(waiting_sorted[: min(2, waiting_sorted.size)]))
        queue_top3_mean = float(np.mean(waiting_sorted[: min(3, waiting_sorted.size)]))
        queue_pnorm = float(np.linalg.norm(waiting, ord=4))
    else:
        queue_top2_mean = 0.0
        queue_top3_mean = 0.0
        queue_pnorm = 0.0
    reward = {"queue_length": np.sum(rs["lane_num_waiting_vehicle_in"]),
              "regional_queue": float(
                  np.sum(rs["lane_num_waiting_vehicle_in"]) + np.sum(rs.get("lane_num_waiting_vehicle_out", 0.0))
              ),
              "queue_max": float(np.max(waiting)) if waiting.size > 0 else 0.0,
              "queue_balance": float(np.std(waiting)) if waiting.size > 0 else 0.0,
              "queue_top2_mean": queue_top2_mean,
              "queue_top3_mean": queue_top3_mean,
              "queue_pnorm": queue_pnorm,
              "pressure": float(np.sum(np.abs(np.asarray(rs["pressure"], dtype=np.float32)))),
              "advanced_pressure": float(np.sum(np.abs(
                  np.asarray(rs.get("traffic_movement_pressure_queue_efficient", np.zeros(12)), dtype=np.float32)
                  + np.asarray(rs.get("lane_enter_running_part", np.zeros(12)), dtype=np.float32)
              ))),
              "downstream_congestion": float(rs.get(
                  "downstream_congestion",
                  np.sum(rs.get("lane_num_vehicle_downstream", np.zeros(12))),
              )),
              "ifdg": float(rs.get("ifdg", 0.0)),
              "switch_penalty": 0.0}
    return reward


def get_global_queue_reward_from_system(logging_data_list_per_gen, time):
    total_queue = 0.0
    for i in range(len(logging_data_list_per_gen)):
        rs = logging_data_list_per_gen[i][time]
        assert time == rs["time"]
        total_queue += float(np.sum(rs["state"]["lane_num_waiting_vehicle_in"]))
    return total_queue


def cal_reward(rs, rewards_components):
    r = 0
    for component, weight in rewards_components.items():
        if weight == 0:
            continue
        if component not in rs.keys():
            continue
        if rs[component] is None:
            continue
        r += rs[component] * weight
    return r


class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]

    def _effective_reward_components(self):
        reward_info = dict(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
        schedule_mode = self.dic_traffic_env_conf.get("REWARD_SCHEDULE_MODE", "none")
        if schedule_mode != "staged_ifdg":
            return reward_info

        warmup_rounds = int(self.dic_traffic_env_conf.get("REWARD_WARMUP_ROUNDS", 20))
        ramp_rounds = max(1, int(self.dic_traffic_env_conf.get("REWARD_RAMP_ROUNDS", 20)))
        target_ifdg_weight = float(reward_info.get("ifdg", 0.0))
        if self.cnt_round < warmup_rounds:
            reward_info["ifdg"] = 0.0
            return reward_info

        alpha = min(1.0, float(self.cnt_round - warmup_rounds + 1) / float(ramp_rounds))
        reward_info["ifdg"] = target_ifdg_weight * alpha
        return reward_info

    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    def load_data_for_system(self, folder):
        self.logging_data_list_per_gen = []
        print("Load data for system in ", folder)
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    def construct_state(self, features, time, i):
        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection

    def _reward_at_time(self, rewards_components, time, i, action=None):
        rs = self.logging_data_list_per_gen[i][time]
        assert time == rs["time"]
        feat_reward = get_reward_from_features(rs['state'])
        if "global_queue_length" in rewards_components:
            feat_reward["global_queue_length"] = get_global_queue_reward_from_system(
                self.logging_data_list_per_gen, time
            )
        if action is not None and "switch_penalty" in rewards_components:
            cur_phase = rs["state"].get("cur_phase", [0])[0]
            feat_reward["switch_penalty"] = 1.0 if int(action) + 1 != int(cur_phase) else 0.0
        return cal_reward(feat_reward, rewards_components)

    def construct_reward(self, rewards_components, time, i, action):
        rs_end_t = time + self.measure_time - 1
        r_instant = self._reward_at_time(rewards_components, rs_end_t, i, action=action)

        list_r = []
        for t in range(time, time + self.measure_time):
            list_r.append(self._reward_at_time(rewards_components, t, i, action=action))
        r_average = np.average(list_r)
        return r_instant, r_average

    def judge_action(self, time, i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']

    def make_reward(self, folder, i):
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []
        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))
        list_samples = []
        nstep = max(1, int(self.dic_traffic_env_conf.get("NSTEP", 1)))
        gamma = float(self.dic_traffic_env_conf.get("GAMMA", 0.8))
        rewards_components = self._effective_reward_components()
        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            for time in range(0, total_time - self.measure_time + 1, self.interval):
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                action = self.judge_action(time, i)
                reward_instant, reward_average = self.construct_reward(rewards_components, time, i, action)
                if nstep > 1:
                    # Accumulate n-step reward: r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
                    r_nstep = reward_average
                    for k in range(1, nstep):
                        t_k = time + k * self.interval
                        if t_k > total_time - self.measure_time:
                            break
                        _, r_k = self.construct_reward(rewards_components, t_k, i, self.judge_action(t_k, i))
                        r_nstep += (gamma ** k) * r_k
                    reward_average = r_nstep
                    next_time = min(time + nstep * self.interval, total_time - 1)
                else:
                    next_time = min(time + self.interval, total_time - 1)

                next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                  next_time, i)
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)

            self.samples_all_intersection[i].extend(list_samples)
            return 1
        except:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def make_reward_for_system(self):
        for folder in os.listdir(self.path_to_samples):
            print(folder)
            if "generator" not in folder:
                continue
            if not self.load_data_for_system(folder):
                continue
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i], "inter_{0}".format(i))

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"), "ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)
