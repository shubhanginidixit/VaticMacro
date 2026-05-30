import os
import py_compile

errors = []
for root, dirs, files in os.walk('.'):
    # skip .venv and hidden folders
    if any(part.startswith('.') for part in root.split(os.sep)) and root != '.':
        continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                py_compile.compile(path, doraise=True)
            except Exception as e:
                errors.append((path, str(e)))

if not errors:
    print('COMPILE_OK')
else:
    print('COMPILE_ERRORS')
    for p, e in errors:
        print(p)
        print(e)
