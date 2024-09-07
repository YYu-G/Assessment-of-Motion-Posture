from models.standardPosture import StandardPosture
from config import db_init as db

class posture_dao:
    def add_posture(repID, mID):
        new_posture = StandardPosture(repID=repID, mID=mID)
        db.session.add(new_posture)
        db.session.commit()
        return new_posture.postureID

    def selete_posture(standardPosture):
        db.session.delete(standardPosture)

    def search_posture_by_posture_id(id):
        return StandardPosture.query.filter_by(postureID=id).first()

    def search_posture_by_movement_id(id):
        return StandardPosture.query.filter_by(mID=id).all()

    def modify_posture(postureID, repID, mID):
        posture = StandardPosture.query.filter_by(postureID=postureID).first()
        posture.repID = repID
        posture.mID = mID
        db.session.commit()

# def add_posture(repID,mID):
#     new_posture=StandardPosture(repID=repID,mID=mID)
#     db.session.add(new_posture)
#     db.session.commit()
#     return new_posture.postureID
#
# def selete_posture(standardPosture):
#     db.session.delete(standardPosture)
#
# def search_posture_by_posture_id(id):
#     return StandardPosture.query.filter_by(postureID=id).first()
#
# def search_posture_by_movement_id(id):
#     return StandardPosture.query.filter_by(mID=id).all()
#
# def modify_posture(postureID,repID,mID):
#     posture=search_posture_by_posture_id(postureID)
#     posture.repID=repID
#     posture.mID=mID
#     db.session.commit()