import yaml
import os

def load_config_from_yaml(dataset_name):
    """
    根据数据集名称从 YAML 文件中加载配置，并将相对路径转换为绝对路径。

    Args:
        dataset_name (str): 数据集名称。

    Returns:
        dict: 对应的配置字典。如果找不到，则返回 None。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)

        if dataset_name in all_configs:
            dataset_config = all_configs[dataset_name]

            # 使用 os.path.normpath() 规范化路径
            for key, value in dataset_config.items():
                if isinstance(value, str) and 'path' in key.lower():
                    # 拼接路径后，立即使用 os.path.normpath() 规范化
                    full_path = os.path.join(project_root, value)
                    dataset_config[key] = os.path.normpath(full_path)

            return dataset_config
        else:
            print(f"错误: 在 YAML 配置中找不到数据集 '{dataset_name}'。")
            return None
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{config_path}'。")
        return None
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件时出错: {e}")
        return None

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ', '.join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({attrs})"

def get_config(dataset_name):
    config_dict = load_config_from_yaml(dataset_name)
    if config_dict:
        return Config(config_dict)
    return None

if __name__ == "__main__":
    config = get_config('HAI')
    if config:
        print(config)
        print(f"训练数据绝对路径: {config.train_data_path}")
        print(f"良性模型绝对路径: {config.benign_model_path}")