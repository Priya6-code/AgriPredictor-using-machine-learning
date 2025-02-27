from flask import Flask, render_template, send_from_directory, request
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Serve assets (CSS, JS, images) from the 'assets/' folder
@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('static/assets', path)

# Load the Machine Learning model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)  # Load only the model (no feature names)
except FileNotFoundError:
    model = None
    print("❌ Error: Model file 'model.pkl' not found. Make sure it's in the correct directory.")

# Home Page (Index)
@app.route("/")
def index():
    return render_template("index.html")  # Render landing page (index.html)

# Form Page (User Input)
@app.route("/inspect", methods=["GET", "POST"])
def inspect():
    return render_template("inspect.html")  # Render input form (inspect.html)


@app.route("/contact", methods=["POST"])
def contact():
    # Retrieve form data
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    # Process the form data (e.g., save to a database or send an email)
    print(f"Name: {name}, Email: {email}, Message: {message}")

    # Return a success response
    return "Message sent successfully!"


# Output Page (Prediction Results)
@app.route("/output", methods=["POST"])
def output():
    if request.method == "POST":
        try:
            # Retrieve form data in the correct order (same as the order used during model training)
            input_data = [
                float(request.form.get("nitrogen")),
                float(request.form.get("phosphorus")),
                float(request.form.get("potassium")),
                float(request.form.get("temperature")),
                float(request.form.get("humidity")),
                float(request.form.get("ph_value")),
                float(request.form.get("rainfall")),
                float(request.form.get("land area"))

            ]
        except (ValueError, TypeError):
            return "❌ Invalid input. Please ensure all fields are filled correctly."

        # Check if the model was loaded successfully
        if model is None:
            return "❌ Model not loaded. Please ensure 'model.pkl' is in the correct directory."

        # Prepare input data for prediction
        try:
            prediction = model.predict([input_data])[0]  # Get prediction (no feature names needed)
        except Exception as e:
            print(f"❌ Prediction Error: {e}")
            return f"Error during prediction: {e}"

        # Render the output page with the prediction result
        print(prediction[0])
        return render_template("output.html", prediction=prediction)
    else:
        # Handle non-POST requests
        return render_template("output.html", prediction="No prediction made yet.")

if __name__ == "__main__":
    app.run(debug=True)
