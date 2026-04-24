from .generator import Generator
from .construct_sample import ConstructSample
from .updater import Updater
from . import model_test
import json
import shutil
import os
import time
import traceback
from multiprocessing import Process


def path_check(dic_path, resume=False):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if resume:
            pass
        elif dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if resume:
            pass
        elif dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):
    try:
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return
    except Exception:
        with open(os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "generator_wrapper_error.log"), "a") as f:
            f.write("round={} gen={}\n".format(cnt_round, cnt_gen))
            f.write(traceback.format_exc())
            f.write("\n")
        raise


def updater_wrapper(cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):
    try:
        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path
        )
        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return
    except Exception:
        with open(os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "updater_wrapper_error.log"), "a") as f:
            f.write("round={}\n".format(cnt_round))
            f.write(traceback.format_exc())
            f.write("\n")
        raise


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.initialize()

    def initialize(self):
        path_check(self.dic_path, resume=bool(self.dic_traffic_env_conf.get("RESUME", False)))
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def _write_round_status(self, cnt_round, stage, **extra):
        status_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "round_status.json")
        data = {
            "cnt_round": cnt_round,
            "stage": stage,
            "timestamp": time.time(),
        }
        data.update(extra)
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _update_dynamic_masking(self, cnt_round):
        """
        Update masking configuration before each round.

        Supports a simple annealing schedule for distance-topk runs:
        MASK_SCHEDULE_COUNTS = [4, 3, 2, 1, 0]
        The schedule is split evenly across NUM_ROUNDS.
        """
        mask_schedule = self.dic_traffic_env_conf.get("MASK_SCHEDULE_COUNTS", None)
        if not mask_schedule:
            return
        schedule = [max(0, int(v)) for v in mask_schedule]
        if not schedule:
            return
        num_rounds = max(1, int(self.dic_traffic_env_conf.get("NUM_ROUNDS", len(schedule))))
        stage = min(len(schedule) - 1, cnt_round * len(schedule) // num_rounds)
        self.dic_traffic_env_conf["MASK_FARTHEST_COUNT"] = schedule[stage]
        print(
            "dynamic mask schedule: round {} -> MASK_FARTHEST_COUNT={}".format(
                cnt_round, self.dic_traffic_env_conf["MASK_FARTHEST_COUNT"]
            )
        )

    def run(self, multi_process=False):
        start_round = 0
        resume = bool(self.dic_traffic_env_conf.get("RESUME", False))
        if resume:
            test_round_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round")
            if os.path.isdir(test_round_dir):
                round_dirs = [d for d in os.listdir(test_round_dir) if d.startswith("round_")]
                if round_dirs:
                    start_round = max(int(d.split("_")[1]) for d in round_dirs) + 1
            if not os.path.exists(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv")):
                f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
                f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
                f_time.close()
        else:
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
            f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
            f_time.close()

        for cnt_round in range(start_round, self.dic_traffic_env_conf["NUM_ROUNDS"]):
            self._update_dynamic_masking(cnt_round)
            print("round %d starts" % cnt_round)
            self._write_round_status(cnt_round, "round_start")
            round_start_time = time.time()
            process_list = []

            print("==============  generator =============")
            self._write_round_status(cnt_round, "generator_start")
            generator_start_time = time.time()
            if multi_process:
                print("-------------- use multi-process for generator -------------")
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    p = Process(target=generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path,
                                      self.dic_agent_conf, self.dic_traffic_env_conf)
                                )
                    print("before")
                    p.start()
                    print("end")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                    if p.exitcode != 0:
                        raise RuntimeError("generator {} exited with code {}".format(i, p.exitcode))
                print("end join")
            else:
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    generator_wrapper(cnt_round=cnt_round,
                                      cnt_gen=cnt_gen,
                                      dic_path=self.dic_path,
                                      dic_agent_conf=self.dic_agent_conf,
                                      dic_traffic_env_conf=self.dic_traffic_env_conf)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time
            self._write_round_status(cnt_round, "generator_end", generator_time=generator_total_time)

            print("==============  make samples =============")
            # make samples and determine which samples are good
            self._write_round_status(cnt_round, "sample_start")
            making_samples_start_time = time.time()
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)
            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time
            self._write_round_status(cnt_round, "sample_end", sample_time=making_samples_total_time)

            print("==============  update network =============")
            self._write_round_status(cnt_round, "update_start")
            update_network_start_time = time.time()
            if self.dic_traffic_env_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                    if p.exitcode != 0:
                        raise RuntimeError("updater exited with code {}".format(p.exitcode))
                else:
                    updater_wrapper(cnt_round=cnt_round,
                                    dic_agent_conf=self.dic_agent_conf,
                                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                                    dic_path=self.dic_path)

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time
            self._write_round_status(cnt_round, "update_end", update_time=update_network_total_time)

            print("==============  test evaluation =============")
            self._write_round_status(cnt_round, "test_start")
            test_evaluation_start_time = time.time()
            model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round,
                            self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time
            self._write_round_status(cnt_round, "test_end", test_time=test_evaluation_total_time)

            print("Generator time: ", generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            self._write_round_status(
                cnt_round,
                "round_end",
                total_time=time.time() - round_start_time,
                generator_time=generator_total_time,
                sample_time=making_samples_total_time,
                update_time=update_network_total_time,
                test_time=test_evaluation_total_time,
            )
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"], exist_ok=True)
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time, making_samples_total_time,
                                                            update_network_total_time, test_evaluation_total_time,
                                                            time.time()-round_start_time))
            f_time.close()
