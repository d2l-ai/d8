import argparse
import sys
import logging
import pandas as pd
import pathlib
import logging
import importlib

TASK_TYPES = ['image_classification', 'object_detection', 'tabular_classification']

message = '''
```{.python .input}
#@hide
# DO NOT EDIT THIS NOTEBOOK.

# This notebook is automatically generated from the `template` notebook in this
# folder by running `d8 gen_desc`
```
'''

def generate_built_in_desc(task_type: str):
    dir = pathlib.Path(__file__).parent.parent/task_type/'built_in'
    template = dir/'_template.md'
    if not template.exists():
        logging.warning(f'Not found {template}')
        return
    with template.open('r') as f:
        lines = f.readlines()
    mod = importlib.import_module('d8.'+task_type)
    names = mod.Dataset.list() # type: ignore
    for name in names:
        tgt = dir/(name+'.md')
        if tgt.exists() and tgt.stat().st_mtime > template.stat().st_mtime:
            logging.info(f'skip to generate {tgt} as it is newer than the template file')
            continue
        lines[0] = f'# `{name}`\n{message}\n'
        for i, l in enumerate(lines):
            if 'name = ' in l:
                lines[i] = f'name = "{name}"\n'
        logging.info(f'write to {tgt}')
        with tgt.open('w') as f:
            f.writelines(lines)

    with (dir/'index.md').open('w') as f:
        f.write(f'# Built-in Datasets\n:label:`{task_type}_built_in`\n\n```toc\n\n')
        for name in sorted(names):
            f.write(name+'\n')
        f.write('\n\n```')

def main():

#     parser = argparse.ArgumentParser(description='''
# Auto Datasets: download ml datasets.

# Run d8 command -h to get the help message for each command.
# ''')
#     parser.add_argument('command', nargs=1, choices=['list', 'download', 'update'])
#     args = parser.parse_args(sys.argv[1:2])

    cmd = sys.argv[1]
    if cmd == 'gen_desc':
        for tt in TASK_TYPES:
            generate_built_in_desc(tt)

if __name__ == "__main__":
    main()
