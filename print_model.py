import joblib
art = joblib.load('models/best_model.pkl')
print(type(art))
if isinstance(art, dict):
    print('keys:', art.keys())
    pipe = art.get('pipeline')
    print('model_name:', art.get('model_name'))
else:
    pipe = art

print('pipeline type:', type(pipe))
try:
    model = pipe.named_steps['model']
    print('model step type:', type(model))
    if hasattr(model,'get_params'):
        print('model params sample:', {k: v for k, v in list(model.get_params().items())[:5]})
except Exception as e:
    print('error reading pipeline/model:', e)
