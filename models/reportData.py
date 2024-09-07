from config import db_init as db

class ReportData(db.Model):
    __tablename__ ='reportData'
    repID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    ownerID=db.Column(db.Integer, nullable=False)
    sptType = db.Column(db.String(50), nullable=False)
    des = db.Column(db.String(50), nullable=False)
    dataFileURL=db.Column(db.String(50), nullable=False)
    repDate = db.Column(db.Date, nullable=False)

    def to_dict(self):
        return {
            'repID':self.repID,
            'ownerID':self.ownerID,
            'sptType':self.sptType,
            'des':self.des,
            'dataFileURL':self.dataFileURL,
            'repDate':self.repDate
        }