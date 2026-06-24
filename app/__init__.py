import os

from flask import Flask, jsonify
from flask import render_template

from app.config import Config, DATA_DIR
from app.database import db


def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(DATA_DIR, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        # Import models so SQLAlchemy registers them before create_all()
        from app import models  # noqa: F401

        db.create_all()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # Register API routes
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    from app.api.analysis import analysis_bp
    app.register_blueprint(analysis_bp)
    
    @app.route("/dashboard")
    def dashboard():
        return render_template("index.html")

    return app
