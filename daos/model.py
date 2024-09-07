from models.model import Model
from config import db_init as db
from sqlalchemy import and_, or_

class model_dao:
    def add_model(modelVersion, modelType, releaseDate, modelName, modelFileURL):
        new_model = Model(modelVersion=modelVersion, modelType=modelType, releaseDate=releaseDate, modelName=modelName,
                          modelFileURL=modelFileURL)
        db.session.add(new_model)
        db.session.commit()
        return new_model.modelID

    def delete_model(model):
        db.session.delete(model)

    def search_model_by_id(ID):
        return Model.query.filter_by(modelID=ID).first().modelName

    def search_model_by_type(type):
        return Model.query.filter_by(modelType=type).all()

    def search_model_by_name(name):
        return Model.query.filter_by(modelName=name).first()

    def search_model_by_date(minDate, maxDate):
        return Model.query.filter(Model.releaseDate >= minDate, Model.releaseDate <= maxDate).all()

    def all_model(self):
        return Model.query.all()

    def search_all_condition(name, old_version,new_version, type1,type2,type3):
        conditions = []
        if type1:
            conditions.append(Model.modelType=='shape')
        if type2:
            conditions.append(Model.modelType=='fitness')
        if type3:
            conditions.append(Model.modelType=='yoga')
        if conditions:
            return Model.query.filter(Model.modelName==name,Model.modelVersion>=old_version,Model.modelVersion<=new_version,
                                  or_(*conditions)).all()
        else:
            return Model.query.filter(Model.modelName == name, Model.modelVersion >= old_version,
                                      Model.modelVersion <= new_version).all()

    def modify_model(modelID, modelVersion, modelType, releaseDate, modelName, modelFileURL):
        m = Model.query.filter_by(modelID=modelID).first()
        m.modelVersion = modelVersion
        m.modelType = modelType
        m.releaseDate = releaseDate
        m.modelName = modelName
        m.modelFileURL = modelFileURL
        db.session.commit()

# def add_model(modelVersion,modelType,releaseDate,modelName,modelFileURL):
#     new_model=Model(modelVersion=modelVersion,modelType=modelType,releaseDate=releaseDate,modelName=modelName,modelFileURL=modelFileURL)
#     db.session.add(new_model)
#     db.session.commit()
#     return new_model.modelID
#
# def delete_model(model):
#     db.session.delete(model)
#
# def search_model_by_id(ID):
#     return Model.query.filter_by(modelID=ID).first()
#
# def search_model_by_type(type):
#     return Model.query.filter_by(modelType=type).all()
#
# def search_model_by_name(name):
#     return Model.query.filter_by(modelName=name).first()
#
# def search_model_by_date(minDate,maxDate):
#     return Model.query.filter_by(Model.releaseDate>=minDate,Model.releaseDate<=maxDate).all()
#
# def modify_model(modelID,modelVersion,modelType,releaseDate,modelName,modelFileURL):
#     m=search_model_by_id(modelID)
#     m.modelVersion=modelVersion
#     m.modelType=modelType
#     m.releaseDate=releaseDate
#     m.modelName=modelName
#     m.modelFileURL=modelFileURL
#     db.session.commit()

