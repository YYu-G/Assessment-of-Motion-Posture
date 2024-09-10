from flask import jsonify
import jwt
from config import SECRET_KEY,algorithm,model_file_path
from daos.user import user_dao
from daos.model import model_dao
from datetime import datetime
import os

def manager_show_user(id,page,size):
    # try:
    #     # 验证令牌并解码
    #     jwt.decode(token, SECRET_KEY, algorithms=[algorithm])
    u=user_dao.search_user_by_id(id)
    if u and user_dao.verify_manager(u):
        user_list=user_dao.all_user(u)
        user_page=user_list[(page-1)*size:page*size]
        users=[{
            'userID':u.userID,
            'username':u.userName,
            'userPhoneNumber':u.userPhoneNumber,
            'password':u.password
            # 'userGender':u.userGender,
            # 'userAge':u.userAge,
            # 'userHeight':u.userHeight,
            # 'userWeight':u.userWeight
        }for u in user_page
        ]
        return jsonify({
            'code':0,
            'message':'请求成功',
            'total':len(users),
            'data':users
        })
    else:
        return jsonify({
            'code': 1,
            'message': '验证失败'
        })

def manager_add_user(id,name,phone,password,age):
    m=user_dao.search_user_by_id(id)
    if m and user_dao.verify_manager(m):
        u = user_dao.search_user_by_phoneNumber(phone)
        if u:
            return jsonify({
                'code': -1,
                'message': '用户已存在'
            })
        else:
            data = user_dao.add_user(name, phone, password, age)
            return jsonify({
                'code': 0,
                'message': '添加成功',
                'data': data
            })
    else:
        return jsonify({
            'code': 1,
            'message': '验证失败'
        })

def manager_delete_user(id,uID):
    m = user_dao.search_user_by_id(id)
    if m and user_dao.verify_manager(m):
        u=user_dao.search_user_by_id(uID)
        user_dao.delete_user(u)
        return jsonify({
            'code': 0,
            'message': '删除完成'
        })
    else:
        return jsonify({
            'code': 1,
            'message': '无权限'
        })

def manager_modify_user(id,uID,phone,name,password):
    m = user_dao.search_user_by_id(id)
    if m and user_dao.verify_manager(m):
        u=user_dao.search_user_by_id(uID)
        if u:
            user_dao.modify_user_by_manager(u,name,phone)
            user_dao.modify_password(u,password)
            return jsonify({
                'code': 0,
                'message': '修改成功'
            })
        else:
            return jsonify({
                'code':1,
                'message':'用户不存在'
            })
    else:
        return jsonify({
            'code': 2,
            'message': '无权限'
        })

def manager_screen_user(id):#指定查找
    # m=user_dao.search_user_by_id(id)
    # if m and user_dao.verify_manager(m):
    #     if len(id)==0:
    #         return jsonify({
    #             'code': 1,
    #             'message': '空的ID或手机号'
    #         })
    u=user_dao.search_user_by_id(id)
    if u:
        data = u.to_dict()
        return jsonify({
            'code': 0,
            'message': '查找成功',
            'data': data
        })
    else:
        return jsonify({
            'code': 2,
            'message': '用户不存在'
        })

    # else:
    #     return jsonify({
    #         'code': 3,
    #         'message': '无权限'
    #     })



def manager_screen_users(name, gender, min_age, max_age, min_h, max_h, min_w, max_w):#范围查找
    if len(gender)==0:#gender标签为空则性别范围包含男女
        gender=-1
    user_list=user_dao.search_all_condition(name,gender,min_age,max_age,min_h,max_h,min_w,max_w)
    users=[{
        'userID': u.userID,
        'username': u.userName,
        'userPhoneNumber': u.userPhoneNumber,
        'userGender': u.userGender,
        'userAge': u.userAge,
        'userHeight': u.userHeight,
        'userWeight': u.userWeight
    }for u in user_list
    ]
    return jsonify({
            'code':0,
            'message':'请求成功',
            'data':users
        })

def delete_users(list):
    err=[]
    for l in list:
        id=l.get('id')
        u=user_dao.search_user_by_id(id)
        if u:
            user_dao.delete_user(u)
        else:
            err.append(id)
    if err:
        return jsonify({
            'code':1,
            'message':'部分id不存在',
            'data':err
        })
    else:
        return jsonify({
            'code':0,
            'message':'删除完成'
        })


def manager_show_model(id,page,size):
    # try:
    #     # 验证令牌并解码
    #     jwt.decode(token, SECRET_KEY, algorithms=[algorithm])
    u=user_dao.search_user_by_id(id)
    if u and user_dao.verify_manager(u):
        model_list=model_dao.all_model()
        model_page=model_list[(page-1)*size:page*size]
        models = [{
            'modelID': m.modelID,
            'modelVersion': m.modelVersion,
            'modelType': m.modelType,
            'releaseDate': m.releaseDate,
            'modelName': m.modelName
        } for m in model_page
        ]
        return jsonify({
            'code': 0,
            'message': '请求成功',
            'total':len(models),
            'data': models
        })
    else:
        return jsonify({
            'code': 1,
            'message': '验证失败'
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

def manager_screen_model(id):#指定查找
    if len(id)!=0:
        m=model_dao.search_model_by_id(id)
    else:
        return jsonify({
            'code': 1,
            'message': '空的ID'
        })
    if m:
        data = m.to_dict()
        return jsonify({
            'code': 0,
            'message': '查找成功',
            'data': data
        })
    else:
        return jsonify({
            'code': 2,
            'message': '该模型不存在'
        })

def manager_screen_models(name, old_version,new_version, type1,type2,type3):#范围查找
    model_list=model_dao.search_all_condition(name, old_version,new_version, type1,type2,type3)
    models = [{
        'modelID': m.modelID,
        'modelVersion': m.modelVersion,
        'modelType': m.modelType,
        'releaseDate': m.releaseDate,
        'modelName': m.modelName
    } for m in model_list
    ]
    return jsonify({
        'code': 0,
        'message': '请求成功',
        'data': models
    })

def manager_delete_model(id,mID):
    u = user_dao.search_user_by_id(id)
    if u and user_dao.verify_manager(u):
        m=model_dao.search_model_by_id(mID)
        file_path=os.path.join(model_file_path,m.modelFileURL)
        if os.path.exists(file_path):
            os.remove(file_path)
            model_dao.delete_model(m)
            return jsonify({
                'code': 0,
                'message': '删除完成'
            })
        else:
            return jsonify({
                'code': 2,
                'message': '文件不存在'
            })
    else:
        return jsonify({
            'code': 1,
            'message': '无权限'
        })

def manager_delete_models(list):
    err = []
    for l in list:
        id = l.get('id')
        m = model_dao.search_model_by_id(id)
        if m:
            user_dao.delete_user(m)
        else:
            err.append(id)
    if err:
        return jsonify({
            'code': 1,
            'message': '部分id不存在',
            'data': err
        })
    else:
        return jsonify({
            'code': 0,
            'message': '删除完成'
        })

def manager_add_model(id,modelVersion, modelType, modelName,file):
    u=user_dao.search_user_by_id(id)
    if u and user_dao.verify_manager(u):
        timestamp = datetime.utcnow().isoformat()
        file_name=f"{modelName}_{timestamp}_{file.filename}"
        file.save(os.path.join(model_file_path, file_name))
        releaseDate=datetime.now().date()
        id=model_dao.add_model(modelVersion,modelType,releaseDate,modelName,file_name)
        return jsonify({
            'code':0,
            'message':'上传成功',
            'data':id
        })
    else:
        return jsonify({
            'code': 1,
            'message': '验证失败'
        })




