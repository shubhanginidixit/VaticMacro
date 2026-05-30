import joblib
from pathlib import Path

models = [Path('models') / 'best_model.pkl', Path('models') / 'ridge.pkl']
for m in models:
    print('\n---', m, 'exists=', m.exists())
    if not m.exists():
        continue
    try:
        a = joblib.load(m)
        print('TYPE', type(a))
        if isinstance(a, dict):
            print('keys:', list(a.keys()))
            feat = a.get('feature_columns')
            print('n_features:', len(feat) if feat else None)
            pipe = a.get('pipeline')
        else:
            pipe = a
        print('PIPELINE TYPE:', type(pipe))
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(pipe, Pipeline):
                final = pipe.named_steps.get('model') or pipe.steps[-1][1]
                print('final step type:', type(final))
                est = getattr(final, 'estimator_', final)
                print('estimator type:', type(est))
                for attr in ['intercept_', 'coef_', 'feature_importances_']:
                    if hasattr(est, attr):
                        val = getattr(est, attr)
                        if hasattr(val, '__len__'):
                            print(attr, list(val)[:10])
                        else:
                            print(attr, val)
        except Exception as e:
            print('Error inspecting pipeline:', e)
    except Exception as e:
        print('Error loading', m, e)
