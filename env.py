from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.handlers.central import MComCentralHandler
import numpy as np

import random


class CustomEnv(MComCore):
    # overwrite the default config
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            # 10 steps per episode
            "EP_MAX_TIME": 10,
            # identical episodes
            "seed": 1234,
            'reset_rng_episode': True,
        })
        # faster user movement
        config["ue"].update({
            "velocity": 10,
        })
        return config

    # configure users and cells in the constructor
    def __init__(self, config={}, users_num=3, stations_num=3, render_mode=None):
        # load default config defined above; overwrite with custom params
        env_config = self.default_config()
        env_config.update(config)
        self.stations_num = stations_num
        self.users_num = users_num

        # stations
        stations = []
        for i in range(self.stations_num):
            pos = (random.randint(0, 100), random.randint(0, 100))
            baseStation = BaseStation(bs_id=i, pos=pos, **env_config['bs'])
            stations.append(baseStation)

        # users
        users = []
        for i in range(self.users_num):
            user = UserEquipment(ue_id=i, **env_config['ue'])
            users.append(user)
        super().__init__(stations, users, config, render_mode)
