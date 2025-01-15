from flask import request, jsonify
from app.controllers.user_analysis import analyze_user
from flask import Flask, request, jsonify, render_template
from app.controllers.user_analysis import analyze_user
from app.controllers.all_users_controller import fetch_all_users

def init_routes(app):
    @app.route("/")
    def home():
        return render_template("index.html")
    
    @app.route("/analyze", methods=["POST"])
    def analyze():
        user_id = request.json.get("user_id")
        if not isinstance(user_id, int):
            return jsonify({"status": "error", "message": "Invalid User ID"}), 400

        result = analyze_user(user_id)
        return jsonify(result)
    
    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        try:
            # Get User ID from the request
            request_data = request.get_json()
            user_id = request_data.get("user_id")

            # Validate User ID
            if not isinstance(user_id, int):
                return jsonify({"status": "error", "message": "Invalid User ID. Must be an integer."}), 400

            # Analyze User
            result = analyze_user(user_id)
            return jsonify(result)

        except Exception as e:
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500
            
    @app.route("/all_users", methods=["POST"])
    def update_and_fetch_all_users():
        try:
            # Extract new data from the request
            new_data = request.json.get("data")
            if not new_data:
                return jsonify({"status": "error", "message": "No data provided"}), 400

            # Call controller function to update and fetch data
            result = fetch_all_users(new_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
