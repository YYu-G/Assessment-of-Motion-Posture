from models.model import Model
from config import db_init as db

def add_model(modelVersion,modelType,releaseDate,modelName,modelFileURL):
    new_model=Model(modelVersion=modelVersion,modelType=modelType,releaseDate=releaseDate,modelName=modelName,modelFileURL=modelFileURL)
    db.session.add(new_model)
    db.session.commit()
    return new_model.modelID

def delete_model(model):
    db.session.delete(model)

def search_model_by_id(ID):
    return Model.query.filter_by(modelID=ID).first()

def search_model_by_type(type):
    return Model.query.filter_by(modelType=type).all()

def search_model_by_name(name):
    return Model.query.filter_by(modelName=name).first()

def search_model_by_date(minDate,maxDate):
    return Model.query.filter_by(Model.releaseDate>=minDate,Model.releaseDate<=maxDate).all()

def modify_model(modelID,modelVersion,modelType,releaseDate,modelName,modelFileURL):
    m=search_model_by_id(modelID)
    m.modelVersion=modelVersion
    m.modelType=modelType
    m.releaseDate=releaseDate
    m.modelName=modelName
    m.modelFileURL=modelFileURL
    db.session.commit()

