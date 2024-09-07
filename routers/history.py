from flask import Blueprint, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from servicers.history import history_show,history_screen,history_report


history = Blueprint('history', __name__)

@history.route('/',methods=['POST'])
# @jwt_required()
def show():
    #id = get_jwt_identity()
    data=request.json
    page=data.get('page')
    size=data.get('pageSize')
    id=data.get('userID')
    return history_show(id,page,size)

@history.route('/screen',methods=['POST'])
def screen():
    data=request.json
    id=data.get('ownerID')
    typ1=data.get('type1')
    type2=data.get('type2')
    type3=data.get('type2')
    min_date=data.get('min_date')
    max_date=data.get('max_date')
    return history_screen(id,typ1,type2,type3,min_date,max_date)

@history.route('/report',methods=['POST'])
def report():
    data=request.json
    return history_report(data.get('repID'))