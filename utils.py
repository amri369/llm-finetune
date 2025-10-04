from typing import Dict
import json

def get_config(config_file: str) -> Dict:
    with open(config_file) as f:
        config = json.load(f)
        return config
