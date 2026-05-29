import joblib
import traceback

paths = ['models/best_model.pkl', 'models/xgboost.pkl', 'models/linear_regression.pkl']
for p in paths:
    try:
        m = joblib.load(p)
        print(p, '->', type(m))
        try:
            s = repr(m)
            print(s[:1000])
        except Exception:
            pass
    except Exception as e:
        print(p, 'ERROR', repr(e))
        traceback.print_exc()
