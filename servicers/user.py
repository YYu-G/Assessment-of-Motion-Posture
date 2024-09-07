from flask import jsonify
import jwt
from flask_jwt_extended import get_jwt_identity

from config import SECRET_KEY,algorithm
from daos.user import user_dao


def user_get_user_message(id):
    # try:
        # # 验证令牌并解码
        # decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=[algorithm])
        # # 提取用户ID
        # user_id = decoded_payload['user_id']
        # # 根据用户ID获取用户信息
        # u=user_dao.search_user_by_id(user_id)
        # # u_mess=u.to_dict()
    u=user_dao.search_user_by_id(id)
    if u:
        return jsonify({
            'code':0,
            'message':'请求成功',
            'data': {
                'userID': u.userID,
                'username': u.userName,
                'userAge': u.userAge,
                'userPhoneNumber': u.userPhoneNumber,
                'userHeight': u.userHeight,
                'userWeight': u.userWeight,
                'userGender': u.userGender,
                'userAuthority': u.userAuthority
            }
        })
    else:
        return jsonify({
            'code':1,
            'message':'验证失败'
        })

    # except jwt.ExpiredSignatureError:
    #     return jsonify({
    #         'code':1,
    #         'message': '令牌已失效'
    #     })
    # except jwt.InvalidTokenError:
    #     return jsonify({
    #         'code':2,
    #         'message': '令牌无效'
    #     })

def user_modify_message(id,name,phone,gender,height,weight,age):
    u=user_dao.search_user_by_id(id)
    u_dict=u.to_dict()
    # if u_dict['userPhoneNumber']!=phone and user_dao.search_user_by_phoneNumber(phone) :
    #     return jsonify({
    #         'code':1,
    #         'message':'该号码已注册'
    #     })
    user_dao.modify_user(u,name,phone,gender,height,weight,age)
    return jsonify({
            'code':0,
            'message':'修改成功'
        })

def user_modify_password(id,old_pass,new_pass):
    u=user_dao.search_user_by_id(id)
    u_dict=u.to_dict()
    if user_dao.verify_password(u_dict['userPhoneNumber'],old_pass):
        user_dao.modify_password(u,new_pass)
        return jsonify({
            'code':0,
            'message':'修改成功'
        })
    else:
        return jsonify({
            'code': 1,
            'message': '旧密码错误'
        })