from models.movement import Movement
from config import db_init as db

def add_movement(name,type):
    new_movement=Movement(mName=name,sportType=type)
    db.session.add(new_movement)
    db.session.commit(new_movement)
    return new_movement.mID

def delete_movement(movement):
    db.session.delete(movement)

def search_movement_by_id(id):
    return Movement.query.filter_by(mID=id).first()

def search_movement_by_name(name):
    return Movement.query.filter_by(mName=name).first()

def search_movement_by_type(type):
    return Movement.query.filter_by(sportType=type).all()

def modify_movement(id,name,type):
    mov=search_movement_by_id(id)
    mov.mName=name
    mov.sportType=type
    db.session.commit()