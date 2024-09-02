from config import db_init as db

class Model(db.Model):
    __tablename__ ='model'
    modelID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    modelVersion=db.Column(db.String(50), nullable=False)
    modelType = db.Column(db.String(50), nullable=False)
    releaseDate=db.Column(db.Date,nullable=False)
    modelName = db.Column(db.String(50), nullable=False)
    modelFileURL = db.Column(db.String(50), nullable=False)