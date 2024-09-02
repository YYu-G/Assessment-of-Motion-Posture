from flask import Blueprint, request
from servicers.user import user_get_user_message

user = Blueprint('user', __name__)

@user.route('/verify',methods=['POST'])
def get_user_message():
    data = request.json
    token=data.get('token')
    return user_get_user_message(token)

