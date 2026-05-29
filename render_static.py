import os
from app import app

OUT_DIR = os.path.join(os.path.dirname(__file__), 'dist')
os.makedirs(OUT_DIR, exist_ok=True)

pages = ['/', '/dashboard', '/analysis', '/models', '/predict', '/forecast', '/about']

with app.app_context():
    for p in pages:
        # Use test_request_context so view functions render templates normally
        with app.test_request_context(p):
            try:
                html = app.view_functions[app.url_map.bind('').match(p)[0]]()
            except Exception:
                # fallback: try to call the endpoint by matching rules
                try:
                    endpoint = app.view_functions.get(app.url_map._rules_by_endpoint.get(p))
                    html = ''
                except Exception:
                    html = ''

            # If direct call didn't work, try resolving using the request path
            if not html:
                try:
                    html = app.full_dispatch_request().get_data(as_text=True)
                except Exception:
                    html = f"<!-- Could not render {p} -->"

            # create a filename
            name = p.strip('/').replace('/', '_') or 'index'
            out_path = os.path.join(OUT_DIR, f"{name}.html")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Wrote {out_path}")

print("Static render complete. Open the files from the 'dist' folder.")
