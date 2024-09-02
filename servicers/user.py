from daos import user
from flask import jsonify
import jwt
from config import SECRET_KEY,algorithm
from daos.user import search_user_by_id


def user_get_user_message(token):
    try:
        # 验证令牌并解码
        decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=[algorithm])
        # 提取用户ID
        user_id = decoded_payload['user_id']
        # 根据用户ID获取用户信息
        u=search_user_by_id(user_id)
        u_mess=u.to_dict
        return jsonify({
            'code':0,
            'message':'请求成功',
            'data':u_mess
        })

    except jwt.ExpiredSignatureError:
        return jsonify({
            'code':1,
            'message': '令牌已失效'
        })
    except jwt.InvalidTokenError:
        return jsonify({
            'code':2,
            'message': '令牌无效'
        })