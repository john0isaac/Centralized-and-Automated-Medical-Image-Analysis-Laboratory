import os
from flask import Flask, request, render_template, redirect, abort, jsonify, flash, url_for
from flask_cors import CORS
from flask_mail import Mail, Message
from sqlalchemy import or_
from models import setup_db
import numpy as np
import json
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import send_from_directory
from keras.preprocessing import image
from tensorflow import keras
import matplotlib.pyplot as plt


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.dtypes.cast(y_true, dtype = tf.float32)
    y_pred = tf.dtypes.cast(y_pred, dtype = tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__)
    app.config.from_pyfile('settings.py')
    # setup_db(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    mail = Mail(app)
    # CORS Headers
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization,true')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE, OPTIONS')
        return response

    @app.route("/")
    def landing_page():
        return render_template("pages/index.html")
    
    @app.route("/ar")
    def landing_ar_page():
        return render_template("pages/index-ar.html")
    
    @app.route("/about")
    def about_page():
        return render_template("pages/about.html")
    
    @app.route("/ar/about")
    def about_ar_page():
        return render_template("pages/about-ar.html")
    
    @app.route("/contact", methods=["GET", "POST"])
    def contact_page():
        if request.method == 'POST':
            body = request.get_json()
            name = body.get('name', None)
            email = body.get('email', None)
            phone = body.get('phone', None)
            message = body.get('message', None)
            subject = 'New Message From '+ email +' Via Your Webstie'
            body = "Hello,\n"\
            "This is "+name+ " from your website.\n\n"\
            "My Email: " +email+'.\n'\
            "My Message: "+ message
            try:
                msg = Message(subject, sender='johnaziz269@gmail.com', recipients=['johnaziz269@gmail.com'])
                msg.body = body
                mail.send(msg)
                return jsonify({
                'success': True 
                })
            except:
                return jsonify({
                    'success': False 
                })
        return render_template("pages/contact.html")

    @app.route("/ar/contact", methods=["GET", "POST"])
    def contact_ar_page():
        if request.method == 'POST':
            body = request.get_json()
            name = body.get('name', None)
            email = body.get('email', None)
            phone = body.get('phone', None)
            message = body.get('message', None)
            subject = 'New Message From '+ email +' Via Your Webstie'
            body = "Hello,\n"\
            "This is "+name+ " from your website.\n\n"\
            "My Email: " +email+'.\n'\
            "My Message: "+ message
            try:
                msg = Message(subject, sender='johnaziz269@gmail.com', recipients=['johnaziz269@gmail.com'])
                msg.body = body
                mail.send(msg)
                return jsonify({
                'success': True 
                })
            except:
                return jsonify({
                    'success': False 
                })
        return render_template("pages/contact-ar.html")

    @app.route("/newsletter-subscribe", methods=["POST"])
    def subscribe_to_newsletter():
        if request.method == 'POST':
            body = request.get_json()
            email = body.get('email', None)
            filePath = app.config['EMAIL_FILE_PATH']
            with open(filePath, "a") as f:
                f.write(email+','+'\n')
            
            subject = 'You Have Sucessfully Subscribed to our Newsletter'
            body = "Hello from CovDec Team,\n\n"\
            "Thank you for subscribing to our monthly newsletter"+'.\n\n'\
            "Regards,"
            try:
                msg = Message(subject, sender='johnaziz269@gmail.com', recipients=[email])
                msg.body = body
                mail.send(msg)
                return jsonify({
                'success': True 
                })
            except:
                return jsonify({
                    'success': False 
                })
            
        return jsonify({
                    'success': False 
                }), 405
    
    @app.route("/ar/newsletter-subscribe", methods=["POST"])
    def subscribe_to_newsletter_ar():
        if request.method == 'POST':
            body = request.get_json()
            email = body.get('email', None)
            filePath = app.config['EMAIL_FILE_PATH']
            with open(filePath, "a") as f:
                f.write(email+','+'\n')
            
            subject = 'You Have Sucessfully Subscribed to our Newsletter'
            body = "Hello from CovDec Team,\n\n"\
            "Thank you for subscribing to our monthly newsletter"+'.\n\n'\
            "Regards,"
            try:
                msg = Message(subject, sender='johnaziz269@gmail.com', recipients=[email])
                msg.body = body
                mail.send(msg)
                return jsonify({
                'success': True 
                })
            except:
                return jsonify({
                    'success': False 
                })
            
        return jsonify({
                    'success': False 
                }), 405

    @app.route("/faq")
    def faq_page():
        return render_template("pages/faq.html")

    @app.route("/ar/faq")
    def faq_ar_page():
        return render_template("pages/faq-ar.html")

    @app.route("/prevention")
    def prevention_page():
        return render_template("pages/prevention.html")

    @app.route("/ar/prevention")
    def prevention_ar_page():
        return render_template("pages/prevention-ar.html")

    @app.route("/search")
    def search_page():
        return render_template("pages/search.html")

    @app.route("/ar/search")
    def search_ar_page():
        return render_template("pages/search-ar.html")

    @app.route("/symptom")
    def symptom_page():
        return render_template("pages/symptom.html")

    @app.route("/ar/symptom")
    def symptom_ar_page():
        return render_template("pages/symptom-ar.html")
   
    @app.route("/symptom-checker-lung")
    def symptom_checker_lung_page():
        return render_template("pages/symptom-checker-lung.html")
    
    @app.route("/ar/symptom-checker-lung")
    def symptom_checker_lung_ar_page():
        return render_template("pages/symptom-checker-lung-ar.html")

    @app.route("/symptom-checker-covid")
    def symptom_checker_covid_page():
        return render_template("pages/symptom-checker-covid.html")
    
    @app.route("/ar/symptom-checker-covid")
    def symptom_checker_covid_ar_page():
        return render_template("pages/symptom-checker-covid-ar.html")
    
    @app.route("/symptom-checker-pneumonia")
    def symptom_checker__pneumonia_page():
        return render_template("pages/symptom-checker-pneumonia.html")
    
    @app.route("/ar/symptom-checker-pneumonia")
    def symptom_checker__pneumonia_ar_page():
        return render_template("pages/symptom-checker-pneumonia-ar.html")

    @app.route("/virus-checker")
    def virus_checker_page():
        return render_template("pages/virus-checker.html")
    
    @app.route("/ar/virus-checker")
    def virus_checker_ar_page():
        return render_template("pages/virus-checker-ar.html")

    @app.route("/tracker")
    def tracker_page():
        return render_template("pages/tracker.html")
    
    @app.route("/ar/tracker")
    def tracker_ar_page():
        return render_template("pages/tracker-ar.html")
    
    @app.route("/prediction-covid", methods=["POST"])
    def prediction_covid_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['COVID19_PATH'])
            #model =  tf.keras.models.load_model(model_path)
            
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(200, 200))
            img=image.img_to_array(img)
            img /= 255
            img=np.expand_dims(img, axis=0)
            img = np.vstack([img])
            
            #classes = model.predict(img, batch_size=10)


            uri = app.config['COVID19_URI']
            api_key = app.config['COVID19_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            classes = np.array(data, dtype=np.float32)

            percentage = round(classes[0][0] * 100, 2)
            if classes[0]>0.5:
                prediction = "Positive"
            else:
                prediction = "Negative"
                percentage = 100 - percentage
            return jsonify({
                'prediction': prediction,
                'success': True,
                'percentage': percentage
                }), 200
    
    @app.route("/ar/prediction-covid", methods=["POST"])
    def prediction_covid_ar_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['COVID19_PATH'])
            #model =  tf.keras.models.load_model(model_path)
            
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(200, 200))
            img=image.img_to_array(img)
            img /= 255
            img=np.expand_dims(img, axis=0)
            img = np.vstack([img])
            
            #classes = model.predict(img, batch_size=10)


            uri = app.config['COVID19_URI']
            api_key = app.config['COVID19_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            classes = np.array(data, dtype=np.float32)
            
            percentage = round(classes[0][0] * 100, 2)
            if classes[0]>0.5:
                prediction = "ايجابي"
            else:
                prediction = "سلبي"
                percentage = 100 - percentage
            return jsonify({
                'prediction': prediction,
                'success': True,
                'percentage': percentage
                }), 200

    @app.route("/prediction-lung-cancer", methods=["POST"])
    def prediction_lung_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['LUNG_CANCER_PATH'])
            #model =  tf.keras.models.load_model(model_path)

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            categories = ['Bengin case', 'Malignant case', 'Normal case']

            img = image.load_img(path, target_size=(256, 256))
            img=image.img_to_array(img)
            img=np.expand_dims(img, axis=0)
            
            #pred = model.predict(img)

            uri = app.config['LUNG_CANCER_URI']
            api_key = app.config['LUNG_CANCER_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            pred = np.array(data, dtype=np.float32)

            max_index = np.argmax(pred)
            # Convert the greatest number to percentage
            percentage = round(100 * pred[0][max_index], 2)

            output = categories[np.argmax(pred)]
            return jsonify({
                'prediction': output,
                'success': True,
                'percentage': percentage
                }), 200

    @app.route("/ar/prediction-lung-cancer", methods=["POST"])
    def prediction_lung_cancer_ar_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['LUNG_CANCER_PATH'])
            #model =  tf.keras.models.load_model(model_path)

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            categories = ['ورم حميد', 'ورم خبيث', 'سلبي']

            img = image.load_img(path, target_size=(256, 256))
            img=image.img_to_array(img)
            img=np.expand_dims(img, axis=0)

            #pred = model.predict(x)

            uri = app.config['LUNG_CANCER_URI']
            api_key = app.config['LUNG_CANCER_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            pred = np.array(data, dtype=np.float32)

            max_index = np.argmax(pred)

            max_index = np.argmax(pred)
            # Convert the greatest number to percentage
            percentage = round(100 * pred[0][max_index], 2)

            output = categories[np.argmax(pred)]
            return jsonify({
                'prediction': output,
                'success': True,
                'percentage': percentage
                }), 200

    @app.route("/prediction-pneumonia", methods=["POST"])
    def prediction_pneumonia_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['PNEUMONIA_PATH'])
            #model =  tf.keras.models.load_model(model_path, custom_objects={'iou_bce_loss':iou_bce_loss, 'mean_iou': mean_iou, 'iou_loss': iou_loss})

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(256, 256), color_mode="grayscale")
            
            img = image.img_to_array(img)

            img=np.expand_dims(img, axis=0)
            
            #classes = model.predict(img, batch_size=1)
            uri = app.config['PNEUMONIA_URI']
            api_key = app.config['PNEUMONIA_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            classes = np.array(data, dtype=np.float32)

            classes = np.squeeze(classes, axis=0)

            pred = resize(classes, (1024, 1024), mode='reflect')
            comp = pred[:, :, 0] > 0.5
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            predictionString = ''
            prediction = 0
            percentage = 0
            prediction = "Negative"
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                # proxy for confidence score
                conf = np.mean(pred[y:y+height, x:x+width])
                # add to predictionString
                predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
                percentage = round(float(predictionString.split()[0]) * 100, 2)
                prediction = "Positive"
            return jsonify({
                'prediction': prediction,
                'success': True,
                'percentage': percentage
                }), 200
    
    @app.route("/ar/prediction-pneumonia", methods=["POST"])
    def prediction_pneumonia_ar_page():
        # check if the post request has the file part
        if request.method == 'POST':
            if 'files' not in request.files:
                flash('No file part')
            file = request.files['files']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #model_path = os.path.join(os.getcwd(), app.config['PNEUMONIA_PATH'])
            #model =  tf.keras.models.load_model(model_path, custom_objects={'iou_bce_loss':iou_bce_loss, 'mean_iou': mean_iou, 'iou_loss': iou_loss})

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(256, 256), color_mode="grayscale")
            
            img = image.img_to_array(img)

            img=np.expand_dims(img, axis=0)

            #classes = model.predict(img, batch_size=1)
            uri = app.config['PNEUMONIA_URI']
            api_key = app.config['PNEUMONIA_API_KEY']
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

            request_data = json.dumps({'input_image': img.tolist()})
            response = requests.post(uri, headers=headers, data=request_data)
            data = json.loads(response.text)["output_image"]
            classes = np.array(data, dtype=np.float32)

            classes = np.squeeze(classes, axis=0)

            pred = resize(classes, (1024, 1024), mode='reflect')
            comp = pred[:, :, 0] > 0.5
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            predictionString = ''
            prediction = 0
            percentage = 0
            prediction = "سلبي"
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                # proxy for confidence score
                conf = np.mean(pred[y:y+height, x:x+width])
                # add to predictionString
                predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
                percentage = round(float(predictionString.split()[0]) * 100, 2)
                prediction = "ايجابي"
            return jsonify({
                'prediction': prediction,
                'success': True,
                'percentage': percentage
                }), 200

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                filename)

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 400,
            'message': 'bad request'
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return render_template('/pages/errors/error.html', data={
            'success': False,
            'error': 404,
            'description': 'Sorry but the page you are looking for does not exist, have been removed, name changed or is temporarily unavailable.',
            'message': 'Page Not Be Found'
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'success': False,
            'error': 405,
            'message': 'method not allowed'
        }), 405

    @app.errorhandler(422)
    def unprocessable(error):
        return jsonify({
            "success": False,
            "error": 422,
            "message": "unprocessable"
        }), 422

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            'success': False,
            'error': 500,
            'message': 'internal server errors'
        }), 500
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4040, debug=True)
