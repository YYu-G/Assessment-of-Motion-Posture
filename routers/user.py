from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from flask_jwt_extended import get_jwt_identity
from servicers.user import user_get_user_message,user_modify_message,user_modify_password

user = Blueprint('user', __name__)

@user.route('/verify',methods=['GET'])
@jwt_required()
def get_user_message():
    # data = request.json
    id = get_jwt_identity()
    return user_get_user_message(id)

@user.route('/modify',methods=['POST'])
def modify_message():
    data=request.json
    id=data['userID']
    name=data['username']
    age=data['userAge']
    phone=data['userPhoneNumber']
    height=data['userHeight']
    weight=data['userWeight']
    gender=data['userGender']
    return user_modify_message(id,name,phone,gender,height,weight,age)

@user.route('/password',methods=['POST'])
def modify_password():
    data=request.json
    id=data['userID']
    old_pass=data['old_pass']
    new_pass=data['new_pass']
    return user_modify_password(id,old_pass,new_pass)