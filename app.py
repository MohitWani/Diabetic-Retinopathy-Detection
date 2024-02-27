from flask import Flask, render_template, request, url_for
from Diabetic_Retinopathy import detection, model

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message= 'No files selected')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message = 'No image selected')
        
        if file:

            models = model.build_model()
            #detection.load_weight(models, 'model.hdf5')
            models.load_weights('Diabetic_Retinopathy\model.hdf5')

            img_array = detection.preprocess_image(file)
            prediction = detection.detection(models, img_array)
            labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}
            return render_template("index.html", message=labels[prediction])
    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)