from config import db_init as db

class User(db.Model):
    __tablename__ = 'user'  # 指定表名
    userID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userName = db.Column(db.String(50), nullable=False)
    userAge = db.Column(db.Integer, nullable=True, default=None)
    userPhoneNumber = db.Column(db.String(50), nullable=True, default=None)
    password = db.Column(db.String(50), nullable=False)
    userHeight = db.Column(db.DOUBLE, nullable=True, default=None)
    userWeight = db.Column(db.DOUBLE, nullable=True, default=None)
    userGender = db.Column(db.Integer, nullable=True, default=None)
    userAuthority = db.Column(db.Integer, nullable=False, default=0)

    def to_dict(self):
        return {
            'userID': self.userID,
            'userName': self.userName,
            'password': self.password,
            'userAge':self.userAge,
            'userPhoneNumber':self.userPhoneNumber,
            'userHeight':self.userHeight,
            'userWeight':self.userWeight,
            'userGender':self.userGender,
            'userAuthority':self.userAuthority
        }