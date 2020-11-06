"""Generate descriptions about built-in datasets"""
import pathlib
import logging
import importlib

message = '''
```{.python .input}
#@hide
# DO NOT EDIT THIS NOTEBOOK.

# This notebook is automatically generated from the `template` notebook in this
# folder by running `autodatasets gen_desc`
```
'''

def generate(task_type: str):
    dir = pathlib.Path(__file__).parent.parent/task_type/'built_in'
    template = dir/'_template.md'
    if not template.exists():
        logging.warning(f'Not found {template}')
        return
    with template.open('r') as f:
        lines = f.readlines()
    mod = importlib.import_module('autodatasets.'+task_type)
    names = mod.Dataset.list()
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
