from models.user import User
from config import db_init as db

def add_user(json):
    new_user=User(userName=json.get('username'),userPhoneNumber=json.get('userPhoneNumber'),password=json.get('password'))
    db.session.add(new_user)
    db.session.commit()
    return new_user.to_dict()

def add_manager(json):
    new_user=User(userName=json.get('username'),userPhoneNumber=json.get('userPhoneNumber'),password=json.get('password'),userAuthority=1)
    db.session.add(new_user)
    db.session.commit()
    return new_user.to_dict()

def search_user_by_id(ID):
    return User.query.filter_by(userID=ID).first()

def search_user_by_gender(gender):
    return User.query.filter_by(userGender=gender).all()

def search_user_by_age(minAge=0,maxAge=1000):
    return User.query.filter_by(User.userAge>=minAge,User.userAge<=maxAge).all()

def search_manager():
    return User.query.filter_by(userAuthority=True).all()

def search_user_by_phoneNumber(phoneNumber):
    return User.query.filter_by(userPhoneNumber=phoneNumber).first()

def search_user_by_height(minHeight,maxHeight):
    return User.query.filter_by(User.userHeight>=minHeight,User.userHeight<=maxHeight)

def search_user_by_weight(minWeight,maxWeight):
    return User.query.filter_by(User.userWeight>=minWeight,User.userWeight<=maxWeight)

def search_user_by_userName(userName):
    return User.query.filter_by(userName=userName)

def delete_user(user):
    db.session.delete(user)

def modify_user(form):
    u=search_user_by_id(form['userID'])
    u.userName = form['userName']
    u.userPhoneNumber = form['userPhoneNumber']
    u.password = form['password']
    if len(form['userAge'])!=0:
        u.userAge = form['userAge']
    if len(form['userGender']) != 0:
        u.userGender = form['userGender']
    if len(form['userHeight']) != 0:
        u.userHeight = form['userHeight']
    if len(form['userWeight']) != 0:
        u.userWeight = form['userWeight']
    db.session.commit()

def verify_password(userPhoneNumber,password):
    data=search_user_by_phoneNumber(userPhoneNumber).to_dict()
    print('ok')
    print(data ['password'])
    if data ['password']!=password:
        return False
    else:
        return True
