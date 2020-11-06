import argparse
import sys
import logging
import pandas as pd
# import pathlib
# import hashlib
# import requests
# import tqdm

# from . import file_reader
from . import built_in_desc

logging.basicConfig(format='[autodatasets:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s')
logging.getLogger().setLevel(logging.INFO)

# pd.set_option('display.max_columns', 500)

# pd.set_option('display.width', 1000)


TASK_TYPES = ['image_classification', 'object_detection']

def main():

#     parser = argparse.ArgumentParser(description='''
# Auto Datasets: download ml datasets.

# Run autodatasets command -h to get the help message for each command.
# ''')
#     parser.add_argument('command', nargs=1, choices=['list', 'download', 'update'])
#     args = parser.parse_args(sys.argv[1:2])

    cmd = sys.argv[1]
    if cmd == 'gen_desc':
        for tt in TASK_TYPES:
            built_in_desc.generate(tt)

if __name__ == "__main__":
    main()
