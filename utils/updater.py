from .config import DIC_AGENTS
import pickle
import os
import time
import traceback
import json
import random
import numpy as np


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None
        self.use_global_per = bool(dic_agent_conf.get("USE_GLOBAL_PER", False))
        self.use_priority_candidate_pool = bool(dic_agent_conf.get("USE_PER", False))

        warmup_rounds = int(dic_agent_conf.get("REDQ_UTD_WARMUP_ROUNDS", 0) or 0)
        warmup_utd = int(dic_agent_conf.get("REDQ_UTD_WARMUP_VALUE", dic_agent_conf.get("REDQ_UTD", 1)) or dic_agent_conf.get("REDQ_UTD", 1))
        after_utd = int(dic_agent_conf.get("REDQ_UTD_AFTER_VALUE", dic_agent_conf.get("REDQ_UTD", 1)) or dic_agent_conf.get("REDQ_UTD", 1))
        if warmup_rounds > 0:
            effective_utd = warmup_utd if cnt_round < warmup_rounds else after_utd
            self.dic_agent_conf["REDQ_UTD"] = effective_utd
            print(
                "[REDQ UTD schedule] round {} -> UTD={} (warmup_rounds={}, warmup_value={}, after_value={})".format(
                    cnt_round, effective_utd, warmup_rounds, warmup_utd, after_utd
                )
            )

        print("Number of agents: ", dic_traffic_env_conf['NUM_AGENTS'])

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample_with_forget(self, i):
        sample_set = []
        try:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                cur_sample_set = []
                while True:
                    chunk = pickle.load(sample_file)
                    if chunk is None:
                        print("skip None chunk when loading samples for inter {0}".format(i))
                        continue
                    if isinstance(chunk, list):
                        cur_sample_set += chunk
                    else:
                        try:
                            cur_sample_set += list(chunk)
                        except TypeError:
                            print("skip malformed chunk type {0} for inter {1}".format(type(chunk), i))
            except EOFError:
                print("===== load samples finished =====")
                sample_file.close()

            ind_end = len(cur_sample_set)
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            # forget
            memory_after_forget = cur_sample_set[ind_sta: ind_end]
            print("==== memory size after forget ====:", len(memory_after_forget))
            if self.cnt_round % self.dic_traffic_env_conf["FORGET_ROUND"] == 0:
                with open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                       "total_samples_inter_{0}".format(i) + ".pkl"), "wb+") as f:
                    pickle.dump(memory_after_forget, f, -1)
            # sample the memory
            if self.use_global_per:
                sample_set = memory_after_forget
                print("==== global PER candidate size =====:", len(sample_set))
            else:
                sample_size = int(self.dic_agent_conf["SAMPLE_SIZE"])
                if self.use_priority_candidate_pool:
                    sample_size *= max(1, int(self.dic_agent_conf.get("PER_POOL_MULT", 1)))
                sample_size = min(sample_size, len(memory_after_forget))
                if (
                    self.sample_indexes is None
                    or len(self.sample_indexes) != sample_size
                    or max(self.sample_indexes) >= len(memory_after_forget)
                ):
                    self.sample_indexes = random.sample(range(len(memory_after_forget)), sample_size)
                sample_set = [memory_after_forget[k] for k in self.sample_indexes]
                if self.use_priority_candidate_pool:
                    print("==== priority candidate samples number =====:", sample_size)
                else:
                    print("==== memory samples number =====:", sample_size)

        except:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i % 100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set

    def load_sample_for_agents(self):
        start_time = time.time()
        print("Start load samples at", start_time)
        if self.dic_traffic_env_conf['MODEL_NAME'] in ["REDQ"]:
            samples_list = []
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample_with_forget(i)
                # [sample1, sample2, ...]
                samples_list.append(sample_set)
            self.agents[0].prepare_Xs_Y(samples_list)

    def update_network(self, i):
        print('update agent %d' % i)
        self._write_updater_status(stage="train_start", agent=i)
        self.agents[i].train_network()
        self._write_updater_status(stage="train_end", agent=i)

        save_name = "round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id)
        print("save_network starts", save_name)
        self._write_updater_status(stage="save_start", agent=i, file_name=save_name)
        try:
            self.agents[i].save_network(save_name)
        except Exception as e:
            self._write_updater_status(
                stage="save_failed",
                agent=i,
                file_name=save_name,
                error=repr(e),
                traceback=traceback.format_exc(),
            )
            print("save_network failed", save_name, repr(e))
            raise
        print("save_network ends", save_name)
        self._write_updater_status(stage="save_end", agent=i, file_name=save_name)

    def update_network_for_agents(self):
        print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            self.update_network(i)

    def _write_updater_status(self, **payload):
        status_path = os.path.join(
            self.dic_path["PATH_TO_WORK_DIRECTORY"],
            "updater_status.json",
        )
        data = {
            "cnt_round": self.cnt_round,
            "timestamp": time.time(),
        }
        data.update(payload)
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
