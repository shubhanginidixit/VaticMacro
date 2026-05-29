import os, joblib

for f in sorted(os.listdir('models')):
    if f.endswith('.pkl'):
        path = os.path.join('models', f)
        try:
            obj = joblib.load(path)
        except Exception as e:
            print(f, 'load_error', e)
            continue
        print('\nFile:', f)
        print('Type:', type(obj))
        if isinstance(obj, dict):
            print('Keys:', list(obj.keys()))
            if 'pipeline' in obj:
                print('Pipeline type:', type(obj['pipeline']))
        else:
            print('Object repr:', repr(obj)[:200])
