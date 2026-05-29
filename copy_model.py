import joblib
import traceback

try:
    lm = joblib.load('models/linear_regression.pkl')
    print('Loaded linear_regression.pkl type:', type(lm))
    joblib.dump(lm, 'models/best_model.pkl')
    print('Copied to models/best_model.pkl')
except Exception as e:
    print('Error:', repr(e))
    traceback.print_exc()
