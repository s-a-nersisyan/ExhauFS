import os
import re


test_files = [
    os.path.join(dp, f)
    for dp, dn, fn in os.walk('test')
    for f in fn if re.match(r'test_.*.py$', f)
]

for f in test_files:
    print('testing ', f)
    os.system(f'python3 -W ignore -m {f.replace("/", ".").replace(".py", "")}')
