import os
from flask import Flask, request, render_template, redirect, abort, jsonify, flash, url_for
from flask_cors import CORS
from flask_mail import Mail, Message
from sqlalchemy import or_
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import send_from_directory
from keras.preprocessing import image
import matplotlib.pyplot as plt


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__)
    app.config.from_pyfile('settings.py')
    #setup_db(app)
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
    
    @app.route("/about")
    def about_page():
        return render_template("pages/about.html")
    
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

    @app.route("/faq")
    def faq_page():
        return render_template("pages/faq.html")
    
    @app.route("/prevention")
    def prevention_page():
        return render_template("pages/prevention.html")

    @app.route("/search")
    def search_page():
        return render_template("pages/search.html")

    @app.route("/symptom")
    def symptom_page():
        return render_template("pages/symptom.html")
   
    @app.route("/symptom-checker-lung")
    def symptom_checker_lung_page():
        return render_template("pages/symptom-checker-lung.html")
    
    @app.route("/symptom-checker-covid")
    def symptom_checker_covid_page():
        return render_template("pages/symptom-checker-covid.html")
    
    @app.route("/symptom-checker-pneumonia")
    def symptom_checker__pneumonia_page():
        return render_template("pages/symptom-checker-pneumonia.html")

    @app.route("/virus-checker")
    def virus_checker_page():
        return render_template("pages/virus-checker.html")

    @app.route("/tracker")
    def tracker_page():
        return render_template("pages/tracker.html")

    @app.route("/prediction", methods=["POST", "GET"])
    def prediction_page():
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

            model =  tf.keras.models.load_model('.\\covid_classifier_model.h5')
            
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(200, 200))
            x=image.img_to_array(img)
            x /= 255
            x=np.expand_dims(x, axis=0)
            images = np.vstack([x])
            
            classes = model.predict(images, batch_size=10)
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
        return jsonify({
                'success': False
                }), 405

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