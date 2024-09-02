from daos import user
from flask import jsonify
import datetime
import jwt
from config import SECRET_KEY,algorithm,piece

def start_login(userPhoneNumber,password):#通过手机号登录
    u = user.search_user_by_phoneNumber(userPhoneNumber)
    if u is None:
        # 用户不存在
        return jsonify({
            'code': -1,
            'message': '用户不存在'
        })

    if user.verify_password(userPhoneNumber,password)==False:
        # 用户存在 密码错误
        return jsonify({
            'code': -2,
            'message': '密码错误'
        })
    else:
        # 定义JWT负载
        payload = {
            'user_id':u.userID,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=piece)  # 令牌过期时间为1小时后
        }
        # 生成JWT令牌
        token = jwt.encode(payload, SECRET_KEY, algorithm=algorithm)

        return jsonify({
        'code': 0,
        'message': '登录成功',
        'token':token
    })

def start_register(json):
    u=user.search_user_by_phoneNumber(json.get('userPhoneNumber'))
    if u :
        return jsonify({
            'code':-1,
            'message':'用户已存在'
        })
    data=user.add_user(json)
    return jsonify({
            'code':0,
            'message':'注册成功',
            'data': data
        })