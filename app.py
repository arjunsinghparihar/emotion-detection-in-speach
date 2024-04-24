from flask import Flask, render_template, redirect, request
import EmotionDetection
from io import BytesIO
import librosa
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

@app.route('/')
def hello():
    return  render_template("index.html")

@app.route('/', methods=["POST"])
def emotion():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)
        label = EmotionDetection.detect(path)
        return render_template("result.html", label=label)



if __name__ == '__main__':
    app.run(debug=True)