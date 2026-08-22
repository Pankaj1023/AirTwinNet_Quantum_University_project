import os
import yaml


class ConfigurationManager:

    def __init__(self, config_file_path: str = "config/config.yaml"):

        self.config_file_path = config_file_path

    def read_config(self):

        with open(self.config_file_path, "r") as file:
            config = yaml.safe_load(file)

        return config