import yaml
import os

def load_config_from_yaml(dataset_name):
    """
    Loads configuration from a YAML file based on the dataset name and converts relative paths to absolute paths.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'SWaT', 'BATADAL', 'HAI').

    Returns:
        dict: The corresponding configuration dictionary. Returns None if the dataset is not found.
    """
    # Determine the project root directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)

        if dataset_name in all_configs:
            dataset_config = all_configs[dataset_name]

            # Normalize paths using os.path.normpath()
            for key, value in dataset_config.items():
                if isinstance(value, str) and 'path' in key.lower():
                    # Construct absolute path and normalize it immediately
                    full_path = os.path.join(project_root, value)
                    dataset_config[key] = os.path.normpath(full_path)

            return dataset_config
        else:
            print(f"Error: Dataset '{dataset_name}' not found in YAML configuration.")
            return None
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Error parsing YAML file: {e}")
        return None

class Config:
    """
    Simple configuration object that converts dictionary keys to object attributes.
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ', '.join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({attrs})"

def get_config(dataset_name):
    """
    Wrapper function to get a Config object for a specific dataset.
    """
    config_dict = load_config_from_yaml(dataset_name)
    if config_dict:
        return Config(config_dict)
    return None

if __name__ == "__main__":
    # Test the configuration loader
    config = get_config('HAI')
    if config:
        print(config)
        print(f"Absolute path to training data: {config.train_data_path}")
        print(f"Absolute path to benign model: {config.benign_model_path}")