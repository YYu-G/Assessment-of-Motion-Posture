from models.reportData import ReportData
from config import db_init as db

def add_report(ownerID,type ,des,dataFileURL,repDate):
    new_report=ReportData(ownerID=ownerID,sptType=type,des=des,dataFileURL=dataFileURL,repDate=repDate)
    db.session.add(new_report)
    db.session.commit()
    return new_report.repID

def delete_report(report):
    db.session.delete(report)

def search_report_by_id(id):
    return ReportData.query.filter_by(repID=id).first()

def search_report_by_owner_id(id):
    return ReportData.query.filter_by(ownerID=id).all()

def search_report_by_type(type):
    return ReportData.query.filter_by(sptType=type).all()

def search_report_by_date(min_date,max_date):
    return ReportData.query.filter_by(ReportData.repDate>=min_date,ReportData.repDate<=max_date).all()

def modify_report(repID,ownerID,type ,des,dataFileURL,repDate):
    rep=search_report_by_id(repID)
    rep.ownerID=ownerID
    rep.sptType=type
    rep.des=des
    rep.dataFileURL=dataFileURL
    rep.repDate=repDate