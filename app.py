from flask import Flask, render_template, request, redirect, url_for
from style_transfer import run_style_transfer  # Import the style transfer function from your Python script
import os

# Create a Flask web application
app = Flask(__name__)

# Set the folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Define the main route for the web page ("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    # If the user submits the form (POST request)
    if request.method == 'POST':
        # Get the uploaded content and style images from the form
        content_img = request.files['content_image']
        style_img = request.files['style_image']

        # Save the uploaded images to the upload folder
        content_path = os.path.join(UPLOAD_FOLDER, 'content.jpg')
        style_path = os.path.join(UPLOAD_FOLDER, 'style.jpg')

        content_img.save(content_path)
        style_img.save(style_path)

        # Call the style transfer function and get the path to the output image
        output_path = run_style_transfer(content_path, style_path)

        # Show the result on the web page
        return render_template('index.html', output_image=output_path)

    # If it's a GET request, just load the page without any image
    return render_template('index.html', output_image=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
