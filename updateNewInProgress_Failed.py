# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:07:49 2022

@author: DELL
"""
import requests
import json
import traceback
import preProcUtilities as putil
# from sys import argv
import TAPPconfig as cfg
import sys
import os
from datetime import datetime as dt
import time as tm
import math
import uuid


statusRevComp = cfg.getStatusReviewCompleted()
statusReview = cfg.getStatusReview()

#Get values for request/response
docUpdApi = cfg.getDocUpdApi()
tappVer = cfg.getTappVersion()
docType = cfg.getDocumentType()
sysUser = cfg.getSystemUser()

paramStatusFailed = cfg.getParamStatusFailed()
statusFailed = cfg.getStatusFailed()
stgExtract = cfg.getStageExtract()
errCode = cfg.getErrcodeError()
errmsgExtractionUpdateFail = cfg.getErrmsgExtractResNotUpd()

UI_SERVER = cfg.getUIServer()
GET_DOCUMENT_RESULT = cfg.getDocoumentResult()
FIND_DOCUMENT = cfg.getDocumentFind()



def updDocInfo(reqStatus,
               stage,
               statusMsg,
               err,
               errMsg,
               prmStatus,
               docRequest,
               docParams,
               docApiInfo):
    docRequest["status"] = reqStatus
    docRequest["stage"] = stage
    docRequest["statusMsg"] = statusMsg
    docRequest["lastProcessedOn"] = math.trunc(tm.time())

    docParams["err"] = err
    docParams["errmsg"] = errMsg
    docParams["status"] = prmStatus

    docApiInfo["request"] = docRequest
    docApiInfo["params"] = docParams
    return docApiInfo


def apiInit(documentId):
    
    docApiInfo = {}
    docApiInfo["id"] = docUpdApi
    docApiInfo["ver"] = tappVer
    docApiInfo["ts"] = math.trunc(tm.time())
    docParams = {}
    docRequest = {}
    docApiInfo["params"] = docParams
    docApiInfo["request"] = docRequest
    docRequest["documentId"] = documentId
    docParams["msgid"] = str(uuid.uuid1())
    docRequest["documentType"] = docType
    docRequest["lastUpdatedBy"] = sysUser

    docApiInfo = updDocInfo(None,
                            None,
                            None,
                            None,
                            None,
                            paramStatusFailed,
                            docRequest,
                            docParams,
                            docApiInfo)

    return docApiInfo

def updateFailure(stage,
                  statusMsg,
                  error,
                  errorMsg,
                  documentId,
                  callbackUrl,
                  auth_token = None):
    #Check doc status
    status = putil.getDocumentStatus(documentId,callbackUrl)
    if not((status == statusReview) or (status == statusRevComp)) and (status == "NEW" or status == "EXTRACTION_INPROGRESS" or status == "EXTRACTION IN PROGRESS"):

        #Init API Jsons
        docApiInfo = apiInit(documentId)
        docRequest = docApiInfo["request"]
        docParams = docApiInfo["params"]
        docApiInfo = updDocInfo(statusFailed, #reqStatus
                                stage, #stage
                                statusMsg, #statusMsg
                                error, #err
                                errorMsg, #errMsg
                                paramStatusFailed, #paramStatus
                                docRequest,
                                docParams,
                                docApiInfo)
        docRequest = docApiInfo["request"]
        docRequest["pp_cloud_update"] = 0
        
        #Mark that extraction is completed
        docRequest["extraction_completed"] = 1
    
        docApiInfo["request"] = docRequest
    
        updated = putil.updateDocumentApi(documentId,
                                          docApiInfo,
                                          callbackUrl)
    
        return updated
    else:
        return True

def updDocFailed():
    """
    input json
    {
        "id": "api.document.find",
        "ver": "1.0",
        "ts": 1571813791276,
        "params": {
            "msgid": ""
        },
        "request": {
            "token": "",
            "filter": {
                "status": "NEW",
                "lastUpdatedOn": {"<":"1658319693017.227"}
            },
            "offset": 0,
            "limit": 10,
            "page": 1
        }
    }
    """
    try:
        rootFolderPath = cfg.getRootFolderPath()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_file_path = os.path.join(rootFolderPath,
                                     str(dt.now()).replace(":","_").replace(" ","_")
                                     + "_updateDocFailed" + ".log")
        log_file = open(log_file_path,"w")
        sys.stdout = log_file
        sys.stderr = log_file
        print("rootFolderPath :",rootFolderPath)
        callbackUrl = cfg.getUIServer()
        documentFind = "/document/find"
        url = callbackUrl + documentFind
        headers = {"Content-Type":"application/json"}
        #Update NEW status to Failed if they are delayed
        t = (tm.time() - 120) * 1000
        data = json.dumps(
            {
            "id": "api.document.find",
            "ver": "1.0",
            "ts": tm.time(),
            "params": {
                "msgid": ""
            },
            "request": {
                "token": "",
                "filter": {
                    "status": "NEW",
                    "submittedOn": {"<":t}
                },
                "offset": 0,
                "limit": 20,
                "page": 1
            }
        })
        print("API details before running",
              url,
              data,
              headers)
        r = requests.post(url = url,
                          data = data,
                          headers = headers)
        print("API get details",
              url,
              data,
              headers,
              r.status_code)
        if r.status_code == 200:
            try:
                r_json = r.json()
                print("output",r_json)
                if r_json["responseCode"].lower() == "ok":
                    docs = r_json["result"]["documents"]
                    for doc in docs:
                        documentId = doc["documentId"]
                        try:
                            updated = updateFailure(stgExtract,
                                                    statusFailed,
                                                    errCode,
                                                    errmsgExtractionUpdateFail,
                                                    documentId,
                                                    callbackUrl)
                        except:
                            print("Failed while updating status",
                                  traceback.print_exc())
                            pass
                        
            except:
                print("Failed during update new to failed",
                      traceback.print_exc())
                pass

        t = (tm.time() - 720) * 1000
        data = json.dumps(
            {
            "id": "api.document.find",
            "ver": "1.0",
            "ts": tm.time(),
            "params": {
                "msgid": ""
            },
            "request": {
                "token": "",
                "filter": {
                    "status": "EXTRACTION_INPROGRESS",
                    "lastUpdatedOn": {"<":t}
                },
                "offset": 0,
                "limit": 20,
                "page": 1
            }
        })
        r = requests.post(url = url,
                          data = data,
                          headers = headers)
        if r.status_code == 200:
            try:
                r_json = r.json()
                if r_json["responseCode"].lower() == "ok":
                    docs = r_json["result"]["documents"]
                    for doc in docs:
                        documentId = doc["documentId"]
                        try:
                            updated = updateFailure(stgExtract,
                                                    statusFailed,
                                                    errCode,
                                                    errmsgExtractionUpdateFail,
                                                    documentId,
                                                    callbackUrl)
                        except:
                            print("Failed while updating status",
                                  traceback.print_exc())
                            pass
            except:
                print("Failed during update new to failed",
                      traceback.print_exc())
                pass
    except:
        print("updDocFailed",
              traceback.print_exc())
        # return json.dumps({"status_code":500,
        #                    "message":"Document status Update Failed"})
    finally:
        try:
            if log_file is not None:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_file.close()
                # uploaded,orgFileLocation = putil.uploadFilesToBlobStore([log_file_path])
                # if uploaded:
                #     os.remove(log_file_path)
        except:
            pass

if __name__ == "__main__":
    updDocFailed()
