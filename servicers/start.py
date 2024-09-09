from flask import jsonify
import datetime
import jwt
from config import SECRET_KEY,algorithm,piece
from daos.user import user_dao
from flask_jwt_extended import JWTManager,create_access_token,jwt_required,get_jwt_identity

def start_login(userPhoneNumber,password):#通过手机号登录
    u = user_dao.search_user_by_phoneNumber(userPhoneNumber)
    if u is None:
        # 用户不存在
        return jsonify({
            'code': -1,
            'message': '用户不存在'
        })
    if user_dao.verify_password(userPhoneNumber,password)==False:
        # 用户存在 密码错误
        return jsonify({
            'code': -2,
            'message': '密码错误'
        })
    else:
        token=create_access_token(identity=u.userID)
        return jsonify({
        'code': 0,
        'message': '登录成功',
        'token':token
    })

def start_register(name,phone,password,age):
    u=user_dao.search_user_by_phoneNumber(phone)
    if u :
        return jsonify({
            'code':-1,
            'message':'用户已存在'
        })
    data=user_dao.add_user(name,phone,password,age)
    return jsonify({
            'code':0,
            'message':'注册成功',
            'data': data
        })