import os
from imgsrcheng.inference import image_search
from flask import Flask, request, render_template, send_from_directory

# Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define Flask app
app = Flask(__name__, static_url_path="/static")

# Define apps home page
@app.route("/")
def index():
    return render_template("index.html")

# Define upload function
@app.route("/upload", methods=["POST"])
def upload():

    upload_dir = os.path.join(APP_ROOT, "uploads/")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = os.path.join(upload_dir, img_name)
        img.save(destination)

    # inference
    result = image_search(os.path.join(upload_dir, img_name))
    result_final = []
    for img in result:
        result_final.append("images/" + img.split("/")[-1])

    return render_template("result.html", image_name=img_name, result_paths=result_final)

@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
