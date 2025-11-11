import yaml

DEFAULT_CFG_PATH='../default.yaml'
with open(DEFAULT_CFG_PATH) as f: DEFAULT_CFG_DICT=yaml.load(f, Loader=yaml.SafeLoader)
