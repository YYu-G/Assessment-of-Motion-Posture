from flask_jwt_extended import jwt_required
from flask_jwt_extended import get_jwt_identity
from flask import Blueprint, request, jsonify
from servicers.manager import (manager_show_user,manager_screen_user,manager_show_model,manager_add_model,manager_add_user,
                               manager_screen_users,manager_screen_model,manager_screen_models,manager_delete_model,
                               manager_delete_models,manager_delete_user,manager_modify_user)

manager = Blueprint('manager', __name__)

@manager.route('/user',methods=['GET'])
@jwt_required()
def show_user():
    id = get_jwt_identity()
    data=request.json
    page=data['page']
    size=data['pageSize']
    return manager_show_user(id,page,size)

@manager.route('/model',methods=['GET'])
@jwt_required()
def show_model():
    id = get_jwt_identity()
    data = request.json
    page = data['page']
    size = data['pageSize']
    return manager_show_model(id,page,size)

@manager.route('/screen_user',methods=['GET'])
def screen_user():
    data=request.json
    id=data.get('userID')
    return manager_screen_user(id)

@manager.route('/screen_model',methods=['POST'])
def screen_model():
    data=request.json
    id=data.get('id')
    return manager_screen_model(id)

@manager.route('/screen_users',methods=['POST'])
def screen_users():
    data=request.json
    name=data.get('name')
    gender=data.get('gender')
    min_age = data.get('min_age')
    max_age = data.get('max_age')
    min_h=data.get('min_h')
    max_h=data.get('max_h')
    min_w=data.get('min_w')
    max_w=data.get('max_w')
    return manager_screen_users(name,gender,min_age, max_age, min_h, max_h, min_w, max_w)

@manager.route('/screen_models',methods=['POST'])
def screen_models():
    data=request.json
    name=data.get('name')
    old_version=data.get('old_version')
    new_version=data.get('new_version')
    type1=data.get('type1')
    type2 = data.get('type2')
    type3 = data.get('type3')
    return manager_screen_models(name, old_version, new_version, type1, type2, type3)

@manager.route('/add_model',methods=['POST'])
def add_model():
    id = get_jwt_identity()
    data=request.form
    version=data['modelVersion']
    type=data['modelType']
    # date=data['releaseDate']
    name=data['modelName']
    file=request.files['file']
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({
            'code':1,
            'message':'没有文件部分'
        })
    file = request.files['file']
    return manager_add_model(id,version,type,name,file)

@manager.route('/delete_model',methods=['DELETE'])
def delete_model():
    id = get_jwt_identity()
    data=request.json
    mID=data.get('modelID')
    return manager_delete_model(id,mID)

@manager.route('/delete_models',methods=['POST'])
def delete_models():
    data=request.json
    list=data.get('list')
    return manager_delete_models(list)

@manager.route('/add_user',methods=['POST'])
def add_user():
    id = get_jwt_identity()
    data=request.json
    name = data.get('userName')
    phone = data.get('userPhoneNumber')
    password = data.get('password')
    age = data.get('userAge')
    return manager_add_user(id,name, phone, password, age)

@manager.route('/delete_user',methods=['DELETE'])
def delete_user():
    id = get_jwt_identity()
    data = request.json
    uID = data.get('userID')
    return manager_delete_user(id, uID)

@manager.route('/modify_user',methods=['POST'])
def modify_user():
    id=get_jwt_identity()
    data=request.json
    uID=data.get('id')
    name=data.get('userName')
    phone=data.get('userPhoneNumber')
    password=data.get('password')
    return manager_modify_user(id,uID,phone,name,password)