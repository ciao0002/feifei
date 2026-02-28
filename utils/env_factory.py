from .cityflow_env import CityFlowEnv


def get_env_class(dic_traffic_env_conf):
    simulator = str(dic_traffic_env_conf.get("SIMULATOR", "cityflow")).lower()
    if simulator == "sumo":
        from .sumo_env import SUMOEnv

        return SUMOEnv
    return CityFlowEnv


def is_sumo(dic_traffic_env_conf):
    return str(dic_traffic_env_conf.get("SIMULATOR", "cityflow")).lower() == "sumo"
