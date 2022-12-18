import numpy as np
from flask import Flask, request, render_template
from tensorflow import keras
from PIL import Image


app = Flask(__name__)

classes  = ["COVID-19","Lung-Opacity","Normal","Viral Pneumonia","Tuberculosis"]

model = keras.models.load_model("models\model.h5")

def model_pipeline(image):
    
    X = np.asarray(Image.open(image).resize((224,224)))

    X = np.expand_dims(X, axis = 0)

    y_hat = model.predict(X)

    y_hat = [round(i * 100,3) for i in y_hat[0]]

    return y_hat

@app.route('/')
def principal():

    return render_template("index.html")

@app.route('/classifier',methods = ["POST","GET"])
def classififier():

    y_hat = [0,0,0,0]
    
    if request.files:

        image = request.files["image"]

        try:

            img = Image.open(image)

            img.save(f"static/cache/cache_img.png")

            y_hat = model_pipeline(image)

        except Exception:

            y_hat = model_pipeline("static/cache/cache_img.png")

    return render_template(template_name_or_list = "gui.html",
                           pred = y_hat,
                           labels = classes)


if __name__ == "__main__":
    
    app.run(debug = True,
            host = "localhost",
            port = "5000")