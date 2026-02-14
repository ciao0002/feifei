"""
Vectorized CityFlow Environment for PPO.
Runs multiple CityFlow instances in parallel for efficient data collection.
"""
import os
import json
import shutil
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from .cityflow_env import CityFlowEnv


def worker_process(conn: Connection, env_id: int, path_to_log: str, 
                   path_to_work_directory: str, dic_traffic_env_conf: dict):
    """
    Worker process that runs a single CityFlow environment.
    Communicates via pipe with the main process.
    """
    # Create unique work directory for this worker
    worker_work_dir = f"{path_to_work_directory}_worker_{env_id}"
    worker_log_dir = f"{path_to_log}_worker_{env_id}"
    
    os.makedirs(worker_work_dir, exist_ok=True)
    os.makedirs(worker_log_dir, exist_ok=True)
    
    # Copy necessary files
    traffic_file = dic_traffic_env_conf.get("TRAFFIC_FILE")
    roadnet_file = dic_traffic_env_conf.get("ROADNET_FILE")
    for f in [traffic_file, roadnet_file]:
        if not f: continue
        src = os.path.join(path_to_work_directory, f)
        dst = os.path.join(worker_work_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    
    env = CityFlowEnv(
        path_to_log=worker_log_dir,
        path_to_work_directory=worker_work_dir,
        dic_traffic_env_conf=dic_traffic_env_conf
    )
    
    while True:
        cmd, data = conn.recv()
        
        if cmd == 'reset':
            state = env.reset()
            conn.send(('obs', state))
        
        elif cmd == 'step':
            action = data
            next_state, reward, done, info = env.step(action)
            conn.send(('step_result', (next_state, reward, done, info)))
        
        elif cmd == 'close':
            conn.close()
            break
        
        elif cmd == 'get_feature':
            feature = env.get_feature()
            conn.send(('feature', feature))


class VecCityFlowEnv:
    """
    Vectorized wrapper for multiple CityFlow environments.
    Uses multiprocessing for true parallelism.
    """
    
    def __init__(self, 
                 n_envs: int,
                 path_to_log: str,
                 path_to_work_directory: str,
                 dic_traffic_env_conf: dict):
        self.n_envs = n_envs
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        
        self.workers = []
        self.conns = []
        
        # Start worker processes
        for i in range(n_envs):
            parent_conn, child_conn = Pipe()
            worker = Process(
                target=worker_process,
                args=(child_conn, i, path_to_log, path_to_work_directory, dic_traffic_env_conf)
            )
            worker.start()
            self.workers.append(worker)
            self.conns.append(parent_conn)
    
    def reset(self) -> List:
        """Reset all environments and return initial observations."""
        for conn in self.conns:
            conn.send(('reset', None))
        
        states = []
        for conn in self.conns:
            _, state = conn.recv()
            states.append(state)
        
        return states
    
    def step(self, actions: List) -> Tuple[List, List, List, List]:
        """
        Step all environments with given actions.
        
        Args:
            actions: List of actions, one per environment
        
        Returns:
            next_states: List of next observations
            rewards: List of rewards
            dones: List of done flags
            infos: List of info dicts
        """
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        
        next_states, rewards, dones, infos = [], [], [], []
        for conn in self.conns:
            _, (next_state, reward, done, info) = conn.recv()
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return next_states, rewards, dones, infos
    
    def step_single(self, env_id: int, action) -> Tuple:
        """Step a single environment."""
        self.conns[env_id].send(('step', action))
        _, (next_state, reward, done, info) = self.conns[env_id].recv()
        return next_state, reward, done, info
    
    def close(self):
        """Close all worker processes."""
        for conn in self.conns:
            conn.send(('close', None))
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()


class SimpleVecEnv:
    """
    Simple synchronous vectorized environment (no multiprocessing).
    Easier to debug and works well for small N.
    """
    
    def __init__(self,
                 n_envs: int,
                 path_to_log: str,
                 path_to_work_directory: str,
                 dic_traffic_env_conf: dict,
                 data_dir: str = None):
        self.n_envs = n_envs
        self.envs = []
        
        # Get filenames from config
        traffic_file = dic_traffic_env_conf.get("TRAFFIC_FILE", "anon_4_4_hangzhou_real.json")
        roadnet_file = dic_traffic_env_conf.get("ROADNET_FILE", "roadnet_4_4.json")
        needed_files = [traffic_file, roadnet_file]
        
        for i in range(n_envs):
            # Create unique directories for each env (as subdirectories)
            env_log = os.path.join(path_to_log, f"env_{i}")
            env_work = os.path.join(path_to_work_directory, f"worker_{i}")
            
            os.makedirs(env_log, exist_ok=True)
            os.makedirs(env_work, exist_ok=True)

            
            # Copy data files - try work_directory first, then data_dir
            for fname in needed_files:
                dst = os.path.join(env_work, fname)
                if os.path.exists(dst):
                    continue
                
                # Try main work directory
                src = os.path.join(path_to_work_directory, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    continue
                
                # Try data_dir if provided
                if data_dir:
                    src = os.path.join(data_dir, fname)
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        continue
                
                print(f"Warning: Could not find {fname} for env {i}")
            
            env = CityFlowEnv(
                path_to_log=env_log,
                path_to_work_directory=env_work,
                dic_traffic_env_conf=dic_traffic_env_conf
            )
            self.envs.append(env)

    
    def reset(self) -> List:
        """Reset all environments."""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List) -> Tuple[List, List, List, List]:
        """Step all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return next_states, rewards, dones, infos
    
    def close(self):
        """Cleanup (no-op for simple env)."""
        pass
