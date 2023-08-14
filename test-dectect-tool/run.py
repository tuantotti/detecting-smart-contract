from slither import Slither
import slither
import os
from solc_select import solc_select
import re

FOLDER_DIR = '/home/tuan/Downloads/Authentication through tx.origin/'
FILE_LIST = os.listdir(FOLDER_DIR)
solidity_versions = solc_select.installed_versions()
AVAILABLE_VERSION = list(solc_select.get_available_versions().keys())
# bytecode = '60 80 60 40 52 34 80 15 61 00 10 57 60 00 80 fd 5b 50 60 c6 80 61 00 1f 60 00 39 60 00 f3 fe 60 80 60 40 52 60 04 36 10 60 1f 57 60 00 35 60 e0 1c 80 63 6b 85 09 9b 14 60 2a 57 60 25 56 5b 36 60 25 57 00 5b 60 00 80 fd 5b 34 80 15 60 35 57 60 00 80 fd 5b 50 60 75 60 04 80 36 03 60 20 81 10 15 60 4a 57 60 00 80 fd 5b 81 01 90 80 80 35 73 ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff 16 90 60 20 01 90 92 91 90 50 50 50 60 77 56 5b 00 5b 80 73 ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff 16 ff fe a2 64 69 70 66 73 58 22 12 20 46 f7 6a f5 1c ee cb c4 1f bd 13 58 63 a0 1d d6 8f 19 a3 c5 a2 4e 77 d2 14 b1 94 0b 1f cb 35 ca 64 73 6f 6c 63 43 00 06 01 00 33'

def detect_version(file_dir):
    with open(file=file_dir) as file:
        line = file.readline()
        regex_version = r'(?:pragma\s+solidity\s*(?:>=([\d.]+)\s*<([\d.]+)|\^([\d.]+)|([\d.]+));)'
        true_version = solc_select.get_latest_release()
        range_version = [AVAILABLE_VERSION[0], AVAILABLE_VERSION[-1]]
        need_get_version = True

        while line:
            match = re.search(regex_version, line)
            if match:
                # case like pragma solidity >=0.4.0 <0.6.0
                if match.group(1) and match.group(2):
                    version_start = match.group(1)
                    version_end = match.group(2)
                    if range_version is None:
                        range_version = [version_start, version_end]
                    else:
                        if range_version[0] < version_start:
                            range_version[0] = version_start
                        if range_version[1] > version_end:
                            range_version[1] = version_end
                    # print('1: ',version_start, ' ', version_end)

                # case like pragma solidity ^0.4.0
                elif match.group(3):
                    true_version = match.group(3)
                    need_get_version = False
                    # if range_version[0] < temp_verion:
                    #     range_version[0] = temp_verion
                    # print('2: ',match, '-->',true_version)
                
                # case like pragma solidity 0.6.0
                elif match.group(4):
                    true_version = match.group(4)
                    need_get_version = False
                    # print('3: ',match, '-->',true_version)
                    break
                    
            line = file.readline()

        if need_get_version:
            for i in AVAILABLE_VERSION:
                if i > range_version[0] and i < range_version[1]:
                    true_version = i
                    break
            
    return true_version


def run():
    # for file_name in FILE_LIST:
    file_name = FILE_LIST[0]
    file_dir = FOLDER_DIR + file_name
    print(file_dir)
    version = detect_version(file_dir)
    print('Detect version: ', version)
    current_version = solc_select.current_version()[0]
    if version != current_version:
        solc_select.switch_global_version(version=version, always_install=True)
    try:
        slitherObj = Slither(file_dir)
        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    run()
    # v = detect_version('/home/tuan/Downloads/Authentication through tx.origin/0x0ba2e75fe1368d8d517be1db5c39ca50a1429441.sol')
    # print(v)
    # print(AVAILABLE_VERSION)