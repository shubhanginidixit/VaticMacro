from app import app

with app.test_client() as c:
    resp = c.get('/models')
    print('Status code:', resp.status_code)
    # print a short excerpt to confirm model name is present
    text = resp.get_data(as_text=True)
    start = text.find('<p class="text-3xl')
    print(text[start:start+200])
