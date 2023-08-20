from slither import Slither
import slither
import os
from solc_select import solc_select
import re

FOLDER_DIR = '/home/tuan/Downloads/Authentication through tx.origin/'
FILE_LIST = os.listdir(FOLDER_DIR)
solidity_versions = solc_select.installed_versions()
AVAILABLE_VERSION = list(solc_select.get_available_versions().keys())

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