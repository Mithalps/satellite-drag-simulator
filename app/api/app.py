from flask import Flask
from app.api.routes import api_bp

app = Flask(__name__)

# Register API routes
app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)