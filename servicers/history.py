import os

from flask import jsonify,send_file
from daos.reportData import report_dao
import jwt
from config import SECRET_KEY,algorithm,rep_file_path

def history_show(id,page,size):
        # 根据用户ID获取信息
        report_list=report_dao.search_report_by_owner_id(id)
        # report_list=report_dao.search_report_by_owner_id(json.get('userID'))
        report_page = report_list[(page - 1) * size:page * size ]
        number=len(report_page)
        reports= [
        {
            'repID': report.repID,
            'ownerID': report.ownerID,
            'sptType': report.sptType,
            'des': report.des,
            'dataFileURL':report.dataFileURL,
            'repDate':report.repDate
        } for report in report_page
    ]
        return jsonify({
            'code':0,
            'message':'请求成功',
            'number':number,
            'data':reports
        })



def history_screen(id,typ1,type2,type3,min_date,max_date):
    # token=json.get('token')
    # try:
    #     # 验证令牌并解码
    #     decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=[algorithm])
    #     # 提取用户ID
    #     user_id = decoded_payload['user_id']
    #     # 根据用户ID获取信息
    report_list=report_dao.search_report_all_condition(id,typ1,type2,type3,min_date,max_date)
    data=[{
        'repID': report.repID,
        'ownerID': report.ownerID,
        'sptType': report.sptType,
        'repDate': report.repDate
    }for report in report_list
    ]
    return jsonify({
        'code':0,
        'message':'请求成功',
        'data':data
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

def history_report(id):
    rep=report_dao.search_report_by_id(id)
    if rep:
        rep_dict=rep.to_dict()
        f_path=os.path.join(rep_file_path, rep.dataFileURL)
        try:
            with open(f_path, 'r') as file:
                file_content = file.read()
        except FileNotFoundError:
            return jsonify({
                'code':1,
                'message': '报告不存在'
            }), 404
        except Exception as e:
            return jsonify({
                'code':2,
                'message': str(e)
            }), 500

        return jsonify({
            'code':0,
            'message':'查找成功',
            'file_content': file_content,
            'data':rep_dict
        })
    else:
        return jsonify({
            'code': 3,
            'message': '报告不存在'
        }), 404
