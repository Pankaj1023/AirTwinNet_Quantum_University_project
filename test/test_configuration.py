from airtwinnet.configuration.configuration import ConfigurationManager


config_manager = ConfigurationManager()

config = config_manager.read_config()


print("Project Name:", config["project"]["name"])
print("Project Version:", config["project"]["version"])

print("Raw Data Path:", config["data"]["raw_data_path"])
print("Processed Data Path:", config["data"]["processed_data_path"])

print("Model Path:", config["artifacts"]["model_path"])
print("Logs Path:", config["logs"]["path"])