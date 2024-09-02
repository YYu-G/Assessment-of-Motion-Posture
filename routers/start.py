from flask import Blueprint, request
from servicers.start import start_login,start_register

start = Blueprint('start', __name__)

@start.route('/login',methods=['POST'])
def login():
    data=request.json
    phoneNumber=data.get('userPhoneNumber')
    password=data.get('password')
    mess=start_login(phoneNumber,password)
    return mess

@start.route('/register',methods=['POST'])
def register():
    data=request.json
    mess=start_register(data)
    return mess
