import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), 'vars.env')
load_dotenv(dotenv_path)

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER')
SECRET_KEY=os.environ.get('SECRET_KEY')
EMAIL_FILE_PATH=os.environ.get('EMAIL_FILE_PATH')

LUNG_CANCER_PATH = os.environ.get('LUNG_CANCER_PATH')
PNEUMONIA_PATH = os.environ.get('PNEUMONIA_PATH')
COVID19_PATH = os.environ.get('COVID19_PATH')

DEBUG = True
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = 'johnaziz269@gmail.com'
MAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')