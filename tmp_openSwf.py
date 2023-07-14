
import sys
import os
from threading import Thread

  
def main():    
    game = os.path.join(_FLASH_PATH, 'game.swf')
    if os.path.exists(game) is False:
        print('Game is not exist!')
        return
    engine = os.path.join(_FLASH_PATH, 'flashplayer' if sys.platform == 'linux' else 'flashplayer.exe')
    args = (f'{engine} \"{game}\"', )
    Thread(target=os.popen, args=args).start()


if __name__ == '__main__':
    main()