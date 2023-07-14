"""此文件用于管理配置文件，不需多次加载，调用该文件的args变量即为字典文件
"""

import yaml
import os

 # 读取配置文件
path = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(path, 'config.yaml')
# print(configPath)
cfg = yaml.load(open(configPath, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

cfg['path_root'] = os.path.abspath(os.path.join(path, cfg['path_root']))
cfg['path_flash_engine'] = os.path.join(cfg['path_root'], cfg['path_flash_engine'])
cfg['path_flash_game'] = os.path.join(cfg['path_root'], cfg['path_flash_game'])

class sta:
    __UI_STA_LIST = cfg['sample']['UI']['status']
    UI_TITLE = __UI_STA_LIST.index('title')
    UI_END   = __UI_STA_LIST.index('end')
    UI_SHOP  = __UI_STA_LIST.index('shop')
    UI_PLAY  = __UI_STA_LIST.index('play')
    UI_OTHER = __UI_STA_LIST.index('other')
    
    __NUM_STA_LIST = cfg['sample']['number']['class']
    NUM_MONEY  = __NUM_STA_LIST.index('money')
    NUM_TARGET = __NUM_STA_LIST.index('target')
    NUM_TIME   = __NUM_STA_LIST.index('time')
    NUM_LEVEL  = __NUM_STA_LIST.index('level')

# 打印参数
# print("\033[34m[CONFIG] config: " + str(cfg) + "\033[0m" )



   

