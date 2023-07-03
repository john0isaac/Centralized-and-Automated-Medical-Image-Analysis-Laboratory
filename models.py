from sqlalchemy import Column, String, Integer, ForeignKey
from flask_sqlalchemy import SQLAlchemy
import os
from sqlalchemy.sql import func

# For using locally
#database_name = 'test'
#database_path = "postgres://{}@{}/{}".format('root:Passw0rd', 'localhost:5432', database_name)

# For production
database_path = os.environ['CLEARDB_DATABASE_URL']

db = SQLAlchemy()

'''
    setup_db(app)
    binds a flask application and a SQLAlchemy service
'''


def setup_db(app, database_path=database_path):
    app.config["SQLALCHEMY_DATABASE_URI"] = database_path
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.app = app
    db.init_app(app)
    db.create_all()


'''
City
'''


class City(db.Model):
    ___tablename__ = 'city'

    id = Column(Integer, primary_key=True)
    name = Column(String(30))
    hospitals = db.relationship('Hospitals', backref='city', lazy='dynamic')
    vaccine_centers = db.relationship('Vaccine_Centers', backref='city', lazy='dynamic')

    def __init__(self, name):
        self.name = name

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def format(self):
        return {
            'id': self.id,
            'name': self.name,
        }


'''
Hospitals
'''


class Hospitals(db.Model):
    ___tablename__ = 'hospitals'

    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone = Column(String(40))
    detailed_address = Column(String(100))
    resident_doctor = Column(String(50))
    website = Column(String(100))
    city_id = Column(Integer, ForeignKey('city.id'))

    def __init__(self, name, phone, detailed_address, resident_doctor, website, city_id):
        self.name = name
        self.phone = phone
        self.detailed_address = detailed_address
        self.resident_doctor = resident_doctor
        self.website = website
        self.city_id = city_id

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def format(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'detailed_address': self.detailed_address,
            'resident_doctor': self.resident_doctor,
            'website': self.website,
            'city_id': self.city_id
        }


'''
Vaccine_Centers
'''


class Vaccine_Centers(db.Model):
    ___tablename__ = 'vaccine_centers'

    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone = Column(String(40))
    detailed_address = Column(String(100))
    city_id = Column(Integer, ForeignKey('city.id'))

    def __init__(self, name, phone, detailed_address, city_id):
        self.name = name
        self.phone = phone
        self.detailed_address = detailed_address
        self.city_id = city_id

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def format(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'detailed_address': self.detailed_address,
            'city_id': self.city_id
        }


'''
Contact_Us
'''


class Contact_Us(db.Model):
    ___tablename__ = 'contact_us'

    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone = Column(String(40))
    email = Column(String(100))
    message = Column(String(100))

    def __init__(self, name, phone, email, message):
        self.name = name
        self.phone = phone
        self.email = email
        self.message = message

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def format(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'email': self.email,
            'message': self.message
        }

  
'''
News_Letter
'''


class News_Letter(db.Model):
    ___tablename__ = 'news_letter'

    id = Column(Integer, primary_key=True)
    email = Column(String(100))

    def __init__(self, email):
        self.email = email

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def format(self):
        return {
            'id': self.id,
            'email': self.email,
        }
