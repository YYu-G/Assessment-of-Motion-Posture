from models.user import User
from config import db_init as db

class user_dao:
    def add_user(name,phone,password,age):
        new_user = User(userName=name, userPhoneNumber=phone,password=password,userAge=age)
        db.session.add(new_user)
        db.session.commit()
        return new_user.to_dict()

    def add_manager(json):
        new_user = User(userName=json.get('username'), userPhoneNumber=json.get('userPhoneNumber'),
                        password=json.get('password'), userAuthority=1)
        db.session.add(new_user)
        db.session.commit()
        return new_user.to_dict()

    def search_user_by_id(ID):
        return User.query.filter_by(userID=ID).first()

    def search_user_by_gender(gender):
        return User.query.filter_by(userGender=gender).all()

    def search_user_by_age(minAge=0, maxAge=1000):
        return User.query.filter(User.userAge >= minAge, User.userAge <= maxAge).all()

    def search_manager(self):
        return User.query.filter_by(userAuthority=1).all()

    def search_user_by_phoneNumber(phoneNumber):
        return User.query.filter_by(userPhoneNumber=phoneNumber).first()

    def search_user_by_height(minHeight, maxHeight):
        return User.query.filter(User.userHeight >= minHeight, User.userHeight <= maxHeight)

    def search_user_by_weight(minWeight, maxWeight):
        return User.query.filter(User.userWeight >= minWeight, User.userWeight <= maxWeight)

    def search_user_by_userName(userName):
        return User.query.filter_by(userName=userName)

    def search_all_condition(name,gender,min_age,max_age,min_h,max_h,min_w,max_w):
        if gender==-1:
            return User.query.filter(User.userName==name, User.userAge>=min_age,User.userAge<=max_age,
                              User.userHeight>=min_h,User.userHeight<=max_h,User.userWeight>=min_w,User.userWeight<=max_w)
        else:
            return User.query.filter(User.userName==name, User.userGender==gender,User.userAge>=min_age,User.userAge<=max_age,
                              User.userHeight>=min_h,User.userHeight<=max_h,User.userWeight>=min_w,User.userWeight<=max_w)

    def all_user(self):
        return User.query.filter_by(userAuthority=0).all()

    def delete_user(user):
        db.session.delete(user)
        db.session.commit()

    def modify_user(u,name,phone,gender,height,weight,age):
        u.userName = name
        u.userPhoneNumber = phone
        u.userAge = age
        u.userGender = gender
        u.userHeight = height
        u.userWeight = weight
        db.session.commit()

    def modify_user_by_manager(u,name,phone):
        u.userName = name
        u.userPhoneNumber = phone
        db.session.commit()

    def modify_password(u,password):
        u.password=password
        db.session.commit()

    def verify_manager(u):
        if u.userAuthority==1:
            return True
        else:
            return False

    def verify_password(userPhoneNumber, password):
        data = User.query.filter_by(userPhoneNumber=userPhoneNumber).first().to_dict()
        print('ok')
        print(data['password'])
        print(data['userPhoneNumber'])
        print(password)
        print(userPhoneNumber)
        if data['password'] != password:
            return False
        else:
            return True

# def add_user(json):
#     new_user=User(userName=json.get('username'),userPhoneNumber=json.get('userPhoneNumber'),password=json.get('password'))
#     db.session.add(new_user)
#     db.session.commit()
#     return new_user.to_dict()
#
# def add_manager(json):
#     new_user=User(userName=json.get('username'),userPhoneNumber=json.get('userPhoneNumber'),password=json.get('password'),userAuthority=1)
#     db.session.add(new_user)
#     db.session.commit()
#     return new_user.to_dict()
#
# def search_user_by_id(ID):
#     return User.query.filter_by(userID=ID).first()
#
# def search_user_by_gender(gender):
#     return User.query.filter_by(userGender=gender).all()
#
# def search_user_by_age(minAge=0,maxAge=1000):
#     return User.query.filter_by(User.userAge>=minAge,User.userAge<=maxAge).all()
#
# def search_manager():
#     return User.query.filter_by(userAuthority=True).all()
#
# def search_user_by_phoneNumber(phoneNumber):
#     return User.query.filter_by(userPhoneNumber=phoneNumber).first()
#
# def search_user_by_height(minHeight,maxHeight):
#     return User.query.filter_by(User.userHeight>=minHeight,User.userHeight<=maxHeight)
#
# def search_user_by_weight(minWeight,maxWeight):
#     return User.query.filter_by(User.userWeight>=minWeight,User.userWeight<=maxWeight)
#
# def search_user_by_userName(userName):
#     return User.query.filter_by(userName=userName)
#
# def delete_user(user):
#     db.session.delete(user)
#
# def modify_user(form):
#     u=search_user_by_id(form['userID'])
#     u.userName = form['userName']
#     u.userPhoneNumber = form['userPhoneNumber']
#     u.password = form['password']
#     if len(form['userAge'])!=0:
#         u.userAge = form['userAge']
#     if len(form['userGender']) != 0:
#         u.userGender = form['userGender']
#     if len(form['userHeight']) != 0:
#         u.userHeight = form['userHeight']
#     if len(form['userWeight']) != 0:
#         u.userWeight = form['userWeight']
#     db.session.commit()
#
# def verify_password(userPhoneNumber,password):
#     data=search_user_by_phoneNumber(userPhoneNumber).to_dict()
#     print('ok')
#     print(data ['password'])
#     if data ['password']!=password:
#         return False
#     else:
#         return True
