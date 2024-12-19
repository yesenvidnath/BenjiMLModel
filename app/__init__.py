from flask import Flask

def create_app():
    """
    Factory function to create a Flask application instance.
    """
    app = Flask(__name__)

    # Import and initialize routes
    from app.routes import init_routes
    init_routes(app)

    return app
