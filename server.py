from flask import Flask
from pyngrok import ngrok
from app.routes import init_routes

# Initialize Flask app
app = Flask(__name__)
init_routes(app)

# Set up Ngrok tunnel
public_url = ngrok.connect(5000).public_url
print(f" * Ngrok URL: {public_url}")

if __name__ == "__main__":
    # Run Flask server
    app.run(port=5000)
