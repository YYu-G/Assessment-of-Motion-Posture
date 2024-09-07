from flask import Blueprint, request
from servicers.start import start_login,start_register

start = Blueprint('start', __name__)

@start.route('/login',methods=['POST'])
def login():
    data=request.json
    phoneNumber=data.get('userPhoneNumber')
    password=data.get('password')
    return start_login(phoneNumber,password)


@start.route('/register',methods=['POST'])
def register():
    data=request.json
    name=data.get('username')
    phone=data.get('userPhoneNumber')
    password=data.get('password')
    age=data.get('userAge')
    mess=start_register(name,phone,password,age)
    return mess
