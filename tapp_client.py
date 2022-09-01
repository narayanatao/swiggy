# -*- coding: utf-8 -*-
# from socket import timeout
import traceback
import os
import time
import pandas as pd
import math
import preProcUtilities as putil
import TAPPconfig as cfg
import uuid
import json
import post_processor as pp
import numpy as np
import dateparser
import datetime
import sys
from celery import Celery
from poller import pollStatus
from client_rules import is_equal_subTotal_TotalAmount,check_date_field_stp_score,check_multi_invoices_stp

import warnings
warnings.filterwarnings("ignore")

broker = cfg.getTaskBroker()
task_name = cfg.getTaskName()
celeryapp = Celery(task_name,broker=broker)

app_name = cfg.getAppName()

#Get values for request/response
docType = cfg.getDocumentType()
sysUser = cfg.getSystemUser()
tappVer = cfg.getTappVersion()
rootFolderPath = cfg.getRootFolderPath()

docUpdApi = cfg.getDocUpdApi()
docResAddApi = cfg.getDocResAddApi()

paramStatusSuccess = cfg.getParamStatusSuccess()
paramStatusFailed = cfg.getParamStatusFailed()

statusRevComp = cfg.getStatusReviewCompleted()
statusProcessed = cfg.getStatusProcessed()
statusReview = cfg.getStatusReview()
statusFailed = cfg.getStatusFailed()

errmsgExtractionUpdateFail = cfg.getErrmsgExtractResNotUpd()
errCode = cfg.getErrcodeError()

stgExtract = cfg.getStageExtract()

statusmsgExtractSuccess = cfg.getStatusmsgExtractSuccess()

def timing(f):
    """
    Function decorator to get execution time of a method
    :param f:
    :return:
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s}: {:.3f} sec'.format(f.__name__, (time2 - time1)))
        return ret

    return wrap

def apiInit(documentId):
    docApiInfo = {}
    docApiInfo["id"] = docUpdApi
    docApiInfo["ver"] = tappVer
    docApiInfo["ts"] = math.trunc(time.time())
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
    docRequest["lastProcessedOn"] = math.trunc(time.time())

    docParams["err"] = err
    docParams["errmsg"] = errMsg
    docParams["status"] = prmStatus

    docApiInfo["request"] = docRequest
    docApiInfo["params"] = docParams
    return docApiInfo

def prepare_request_ML(docRequest,
                       docInfo):
    #Form the TAPP API request object for document result update
    resultApiInfo = {}
    resultApiInfo["id"] = docResAddApi
    resultApiInfo["ver"] = tappVer
    resultApiInfo["ts"] = math.trunc(time.time())

    resRequest = {}
    resultApiInfo["request"] = resRequest
    resRequest["documentId"] = docRequest["documentId"]
    resRequest["processingEngine"] = "ML"
    resRequest["documentInfo"] = docInfo["result"]["documentInfo"]
    resRequest["documentLineItems"] = docInfo["result"]["documentLineItems"]

    for key in docInfo.keys():
        if key != "result":
            docRequest[key] = docInfo[key]

    return docRequest,resultApiInfo

def updateStatusToCloud(auth_token,
                        documentId,
                        success):
    uploaded = False
    try:
        localPaths = []
        #update Cloud with status against Auth-token
        cloud_dict = {}
        cloud_dict["auth_token"] = auth_token
        cloud_dict["sub_id"] = cfg.getSubscriberId()
        cloud_dict["document_id"] = documentId
        cloud_dict["body"] = {}
        cloud_dict["stage"] = "client_processing"
        cloud_dict["create_time"] = str(datetime.datetime.now())
        cloud_dict["success"] = success
        cloud_dict["is_start"] = 0
        cloud_dict["is_end"] = 1
        cloud_json = json.dumps(cloud_dict)

        localPath = os.path.join(rootFolderPath,
                                 documentId + "_auth:" + auth_token + "__islaststage.json")
        localPaths.append(localPath)

        with open(localPath,"w") as f:
            f.write(cloud_json)

        uploaded = putil.uploadFilesToBlobStore(localPaths)
        if uploaded:
            for localPath in localPaths:
                try:
                    os.remove(localPath)
                except:
                    pass
        return uploaded
    except:
        print("updateStatusToCloud",
              traceback.print_exc())
        return False

def updateSuccess(status,
                  stage,
                  docResult,
                  docApiInfoInp,
                  documentId,
                  auth_token,
                  callbackUrl):

    #Init API Jsons
    # docRequest = docApiInfo["request"]
    # docParams = docApiInfoInp["params"]
    # print(docParams)
    docApiInfo = apiInit(documentId)
    docApiInfo["request"]["status"] = status
    docApiInfo["request"]["stage"] = stage
    docApiInfo["request"]["statusMsg"] = statusmsgExtractSuccess
    docApiInfo["request"]["lastProcessedOn"] = math.trunc(time.time())
    docApiInfo["request"]["extractionCompletedOn"] = math.trunc(time.time())
    #Not storing invoice data in document metadata due to security concerns from Swiggy
    # docApiInfo["request"]["invoiceNumber"] = docApiInfoInp["request"].get("invoiceNumber","")
    # docApiInfo["request"]["invoiceDate"] = docApiInfoInp["request"].get("invoiceDate","")
    # docApiInfo["request"]["totalAmount"] = docApiInfoInp["request"].get("totalAmount","")
    #Not storing invoice data in document metadata due to security concerns from Swiggy
    docApiInfo["request"]["currency"] = docApiInfoInp["request"].get("currency","")
    docApiInfo["request"]["overall_score"] = docApiInfoInp["request"]["overall_score"]
    docApiInfo["request"]["stp"] = docApiInfoInp["request"]["stp"]
    docApiInfo["request"]["vendorId"] = docApiInfoInp["request"].get("vendorId","")
    docApiInfo["request"]["pages"] = docApiInfoInp["request"]["pages"]
    #Aug 05 2022 - Add number of pages OCRed
    docApiInfo["request"]["pages_ocred"] = docApiInfoInp.get("pages_ocred",0)
    #Aug 05 2022 - Add number of pages OCRed
    downloadUrl = cfg.getPreprocServer() + "/" + cfg.getDownloadResultsApi() + "/" + documentId
    docApiInfo["request"]["resultDownloadLink"] = downloadUrl
    #Mark that extraction is completed
    docApiInfo["request"]["extraction_completed"] = 1
    #download png file to local folder and update path
    docApiInfo["request"]["rawPrediction"] = docApiInfoInp['result'].get("rawPrediction","")

    docApiInfo["params"]["err"] = None
    docApiInfo["params"]["errmsg"] = None
    docApiInfo["params"]["status"] = paramStatusSuccess
    # docResult["params"] = {}
    # docResult["params"]["msgid"] = docApiInfo["params"]["msgid"]
    
    #Jul 28 2022 - When we retry a failed document, check if results were already created.
    #It happens sometimes that only status update has failed, so it is better to update only the status
    docResultPresent = putil.getDocumentResultApi(documentId,
                                                  callbackUrl)
    docResultUpdated = True
    if not docResultPresent:
        docResultUpdated = putil.createDocumentResultsApi(documentId,
                                                          docResult,
                                                          callbackUrl)
    print("Document Update Request for successful transaction: ")
    if docResultUpdated:
        uploaded = updateStatusToCloud(auth_token,
                                       documentId,1)
        if not uploaded:
            docApiInfo["request"]["pp_cloud_update"] = 0
        else:
            docApiInfo["request"]["pp_cloud_update"] = 1
    
        # docApiInfo["request"] = docRequest
        print("Update Document Api", docApiInfo)
        if "result" in docApiInfo.keys():
            del docApiInfo["result"]

        #Jul 05 2022 - set stp true only if all business rules satisfy
        from client_rules import bizRuleValidateForUi as biz_rl
        from client_rules import isMasterDataPresent
        from client_rules import isInvoiceNumberAnAmount
        from client_rules import isTotOrSubCalc
        from client_rules import getBuyerStateCode as getBSC
        callbackUrl = cfg.getUIServer()
        result = biz_rl(documentId, callbackUrl)
        print("Get Buyer State Code", documentId)
        buyerStateCode = getBSC(documentId,
                                callbackUrl)
        if buyerStateCode:
            docApiInfo["request"]["buyer_state_code"] = buyerStateCode
            print("Buyer State Code",
                  documentId,
                  buyerStateCode)
        else:
            docApiInfo["request"]["buyer_state_code"] = ''
        print("Business Rules Result",result)
        if result is not None:
            if len(result) > 0:
                stp = False
            else:
                stp = True
                #After biz rule validation passed through,
                #please also check if invoiceNumber is not an Amount field
                
                #After biz rule validation passed through,
                #please also check if invoiceNumber is not an Amount field

                isMasterDataPresent = isMasterDataPresent(documentId,
                                                          cfg.getUIServer())
                stp = isMasterDataPresent
                print("isMasterDataPresent stp flag :",isMasterDataPresent)
                # if stp:
                #     stp = not isInvoiceNumberAnAmount(documentId,
                #                                       cfg.getUIServer())
                #     print("stp flag  After isInvoiceNumberAnAmount :",stp)
                if stp:
                    stp = not isTotOrSubCalc(documentId,
                                             cfg.getUIServer())
                    print("stp flag  After isTotOrSubCalc :",stp)
                #     #Check if total or subtotal is calculated and do not allow for stp
                # Checking total == subtotal if true making stp false
                # sub_tot_match = is_equal_subTotal_TotalAmount(documentId,
                #                                               cfg.getUIServer())
                # print("sub_tot_match :",sub_tot_match)
                # if sub_tot_match is not None:
                #     stp = sub_tot_match
                if stp:
                    invdt_stp = check_date_field_stp_score(documentId,
                                                           cfg.getUIServer())
                    print("date field stp Check :",invdt_stp)
                    stp = invdt_stp
                if stp:
                    multi_invoices = check_multi_invoices_stp(documentId,
                                                           cfg.getUIServer())
                    print("Mult invoice stp check :",multi_invoices)
                    stp = multi_invoices

        else:
            print("Business Rules Result is empty",False)
            stp = False

        print("STP after validating rules : ",stp)
        docApiInfo["request"]["stp"] = stp

        if stp:
            docApiInfo["request"]["status"] = statusRevComp
        else:
            docApiInfo["request"]["status"] = statusReview

        #Jul 05 2022 - set stp true only if all business rules satisfy
        print("Doc Status before update: ",
              docApiInfo.get("request").get("status"))

        #Jul 18 2022 Call new function for updateDocument that returns response code as well
        # updated = putil.updateDocumentApi(documentId,
        #                                   docApiInfo,
        #                                   callbackUrl)
        print("Is RawPrediction there",
              docApiInfo.get("request").get("rawPrediction"))
        if docApiInfo.get("request").get("rawPrediction"):
            del docApiInfo["request"]["rawPrediction"]
        print("Is RawPrediction there 2",
              docApiInfo.get("request").get("rawPrediction"))

        updated,resp_code = putil.updateDocumentApi_New(documentId,
                                                        docApiInfo,
                                                        callbackUrl)

        # # if not updated and (resp_code == 413):
        # if docApiInfo.get("request").get("rawPrediction"):
        #     del docApiInfo["request"]["rawPrediction"]
        #     updated,resp_code = putil.updateDocumentApi_New(documentId,
        #                                                     docApiInfo,
        #                                                     callbackUrl)
        if not updated:
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)

        #Jul 18 2022 Call new function for updateDocument that returns response code as well

    else:
        updated = updateFailure(stgExtract,
                                statusFailed,
                                errCode,
                                errmsgExtractionUpdateFail,
                                documentId,
                                callbackUrl,
                                auth_token)
    return updated

def updateFailure(stage,
                  statusMsg,
                  error,
                  errorMsg,
                  documentId,
                  callbackUrl,
                  auth_token):

    #Check doc status
    status = putil.getDocumentStatus(documentId,callbackUrl)
    if not((status == statusReview) or (status == statusRevComp)) and (status == "NEW" or status == "EXTRACTION_INPROGRESS"):
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
        print("Document Update Request for failed transaction: ",docApiInfo)
        #print("Document Update Request for failed transaction: ")
        uploaded = updateStatusToCloud(auth_token,
                                       documentId,0)
        docRequest = docApiInfo["request"]
        if not uploaded:
            docRequest["pp_cloud_update"] = 0
        else:
            docRequest["pp_cloud_update"] = 1
        
        #Mark that extraction is completed
        docRequest["extraction_completed"] = 1
    
        docApiInfo["request"] = docRequest
    
        updated = putil.updateDocumentApi(documentId,
                                          docApiInfo,
                                          callbackUrl)
    
        return updated
    return False

def processExtRes(output,docInfo):
    try:
        inpFlds = ['index', 'token_id', 'page_num', 'line_num', 'line_text', 'line_left',
        'line_top', 'line_height', 'line_width', 'line_right', 'line_down', 'word_num',
        'text', 'conf', 'left', 'top', 'height', 'width', 'right', 'bottom', 'image_height', 
        'image_widht', 'wordshape', 'W1Ab', 'W2Ab', 'W3Ab', 'W4Ab', 'W5Ab', 'd1Ab', 'd2Ab', 'd3Ab', 'd4Ab', 
        'd5Ab', 'W1Lf', 'W2Lf', 'W3Lf', 'W4Lf', 'W5Lf', 'd1Lf', 'd2Lf', 'd3Lf', 'd4Lf', 'd5Lf',
        'predict_label', 'prediction_probability']
        if output is not None:
            docObj = output[0]
            rawPrediction = output[1][inpFlds]
            overall_score = output[2]
            stp = output[3]
            vendor_id = output[4]
            docInfo["result"] = docObj

            #update docInfo with invoice number, invoice date, currency and total Amount
            documentInfo = docObj["documentInfo"]
            reqFlds = ["invoiceNumber","invoiceDate","currency","totalAmount"]
            for fld in documentInfo:
                if fld["fieldId"] in reqFlds:
                    if fld["fieldId"] == "invoiceDate":
                        dtparsed = dateparser.parse(fld["fieldValue"])
                        if dtparsed is not None:
                            timeval = dtparsed.hour + dtparsed.minute + dtparsed.second
                            if timeval > 0:
                                docInfo[fld["fieldId"]] = str(datetime.datetime.timestamp(dtparsed))
                            else:
                                docInfo[fld["fieldId"]] = str(datetime.datetime.timestamp(dtparsed) * 1000)
                        else:
                            docInfo[fld["fieldId"]] = fld["fieldValue"]
                    else:
                        docInfo[fld["fieldId"]] = fld["fieldValue"]

            docInfo["overall_score"] = overall_score
            docInfo["stp"] = stp
            docInfo["vendorId"] = vendor_id
            #Jul 20 2022 - stop updating rawPrediction
            # docInfo['result']["rawPrediction"] = rawPrediction.to_json(orient = "records")
            docInfo['result']["rawPrediction"] = json.dumps({})
            #Jul 20 2022 - stop updating rawPrediction
            #Aug 05 2022 - find number of pages OCRed
            no_pages_ocred = 0
            try:
                no_pages_ocred = len(list(rawPrediction["page_num"].unique()))
            except:
                pass
            docInfo['pages_ocred'] = no_pages_ocred
            #Aug 05 2022 - find number of pages OCRed
            return docInfo
        else:
            return None
    except:
        print("Error in extraction or in processing post processor result",
              traceback.print_exc())
        return None

@timing
def processExtraction(pred_file_path,
                      documentId,
                      callbackUrl):
    #1. It calls the image preprocess method for image enhancement and extract invoice number,
    #Date, Currency and Amount fields from Invoice
    #2. It updates the status of a preprocessed image to "Ready for Extraction", if the
    # processes are successfully completed. In case of failure, it is updated as "Failed"

    def convert(o):
        if isinstance(o, np.int64): return int(o)

    try:

        DF_PRED = pd.read_csv(pred_file_path,
                              index_col = 0)
        
        #Save the pred file to Blob storage
        #Post Processor On Premise
        #Call post processor
        #get vendorname if available and pass it to post-processor
        print("CallbackUrl: ",callbackUrl)
        docMetaData = putil.getDocumentApi(documentId, callbackUrl)
        print("docMetaData is \n{}".format(docMetaData))

        if not docMetaData:
            dict_final,stp_score,overall_score, vendor_id = pp.post_process(DF_PRED)
        else:
            dict_final,stp_score,overall_score, vendor_id = pp.post_process(DF_PRED,
                                                                            docMetaData=docMetaData)
        print("Post Processor Executed")

        print("Post Processor Build Json Called ", dict_final)
        json_dict = pp.build_final_json(dict_final)
        print("Post Processor Build Json Executed",
              json_dict)

        stp = False
        if stp_score > 0:
            stp = True

        output = [json_dict, DF_PRED, overall_score * 100, stp, vendor_id]

        print("Extraction and Post-Processor completed with overall score and stp {} and {}".format(overall_score,stp))
        return output

    except:
        print("processExtraction",
              traceback.print_exc())
        return None

def timeExpired(delta):

    from datetime import datetime as dt
    from datetime import timedelta as td
    expired = False
    exp_time = dt.now()
    try:
        print("Delta type:", type(delta))
        if isinstance(delta, str):
            delta_time = dt.strptime(delta, '%H:%M:%S')
            time_delta = td(hours=delta_time.hour,
                            minutes=delta_time.minute,
                            seconds=delta_time.second)
            exp_time = dt.now() + time_delta
        elif isinstance(delta,dt):
            exp_time = delta
    
        t = dt.fromtimestamp(time.time())
        expired = t > exp_time

        print(t,exp_time,type(t),type(exp_time))

        return expired,exp_time
    except:
        print(traceback.print_exc())
        return True, exp_time
    
# GetExtraction result 

def getExtractionResults(auth_token,
                         delta,
                         documentId,
                         callbackUrl,
                         sub_id):

    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_file_path = os.path.join(rootFolderPath,
                                     documentId + "_auth:" + auth_token + ".log")
        log_file = open(log_file_path,"a")
        sys.stdout = log_file
        sys.stderr = log_file
        #Check if extraction request has expired

        expired,exp_time = timeExpired(delta)
        print("Expired time : ",expired, "Expected time :",exp_time)
        if expired:
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)
            return None
        docApiInfo = apiInit(documentId)    
        docRequest = docApiInfo["request"]
        import datetime
        print("Delta input",delta)
        delta = datetime.datetime.strptime(delta, "%H:%M:%S")
        print("Delta strptime",delta)
        delta = delta - datetime.datetime(1900, 1, 1)
        print("Delta diff",delta)
        delta = delta.total_seconds()
        print("Delta time in Seconds", delta)
        delta = delta + 600
        print("Delta plus + 10 minutes : ",delta)

        Poll_result = pollStatus(auth_token,
                                 delta,
                                 documentId,
                                 sub_id)
        print("Poll_result ", Poll_result)
        if Poll_result == None:
            #Update document metadata to failure status
            updated = updateFailure(stgExtract,
                                    statusFailed,
                                    errCode,
                                    errmsgExtractionUpdateFail,
                                    documentId,
                                    callbackUrl,
                                    auth_token)
            print("Error updated ")
            return None
        else:
            print("Result generated", Poll_result)
            ext_status = Poll_result["status"]
            print("Extraction Status:", ext_status)
            if ext_status == "Extracted":
                #Extraction is complete. So, get the result from "body"
                result_json = Poll_result["result"]
                #This result must be having the pred file path
                result = result_json
                #Download the pred file
                blob_pred_path = result["pred_file"]
                fileName = os.path.basename(blob_pred_path)
                local_pred_path = os.path.join(rootFolderPath,
                                            os.path.splitext(fileName)[0] +
                                            "_pred" +
                                            os.path.splitext(fileName)[1])
                pages = result["pages"]

                storageType = cfg.getStorageType()
                if storageType.upper() == "BLOB":
                    for page_index,page in enumerate(pages):
                        pngUrl = page["url"]
                        pages[page_index]["pngURI"] = pngUrl
                        # pages[page_index]["url"] = "preprocessor/" + filename
                else:
                    pngLocations = []
                    pngLocalPaths = []
                    print("Current Working Directory 1:",
                        os.getcwd())
                    uiRoot = cfg.getUIRootFolder()
                    print("Current Working Directory 2:",
                        os.getcwd())
                    png_folder = os.path.join(uiRoot,
                                            "preprocessor")
                    print("Current Working Directory:", os.getcwd())
                    print("png folder:",
                        png_folder)
                    os.makedirs(png_folder,
                                exist_ok=True)
                    for page_index,page in enumerate(pages):
                        pngUrl = page["url"]
                        filename = os.path.basename(pngUrl)
                        pngLocations.append(pngUrl)
                        localFilePath = os.path.join(png_folder,
                                                    filename)
                        pngLocalPaths.append(localFilePath)
                        pages[page_index]["pngURI"] = "preprocessor/" + filename
                        pages[page_index]["url"] = "preprocessor/" + filename
    
                    blob_downloads = [blob_pred_path] + pngLocations
                    local_downloads = [local_pred_path] + pngLocalPaths
                    print("download folders: ",
                        blob_downloads,
                        local_downloads)
                    downloads = zip(blob_downloads,local_downloads)
    
                    downloaded = putil.downloadFilesFromBlob(downloads)
                    print("Pred File Downloaded", downloaded)
                    if not downloaded:
                        #Update document metadata to failure status
                        updated = updateFailure(stgExtract,
                                                statusFailed,
                                                errCode,
                                                errmsgExtractionUpdateFail,
                                                documentId,
                                                callbackUrl,
                                                auth_token)
                        return None
                
                #Update pred file with new rules to extract critical header fields
                #predHdrUsingFuzzy(df)

                #If it's a successful call, go to post-processor
                docApiInfo["request"]["pages"] = pages
                output = processExtraction(local_pred_path,
                                           documentId,
                                           callbackUrl)
                docInfo = processExtRes(output,
                                        docApiInfo)
                print("o/p of process Extraction Response",
                    docInfo)

                if docInfo is None:
                    updated = updateFailure(stgExtract,
                                            statusFailed,
                                            errCode,
                                            errmsgExtractionUpdateFail,
                                            documentId,
                                            callbackUrl,
                                            auth_token)
                    return None

                print("output json:",output,
                    "docInfo json:",docInfo,
                    "docRequest:",docRequest)
                docRequest, resultApiInfo = prepare_request_ML(docRequest,
                                                               docInfo)
                print("prepare request ML",docRequest,resultApiInfo)
                if (docRequest["stp"] == True) or (docRequest["stp"] == "True"):
                    status = statusRevComp
                else:
                    status = statusReview
                #Aug 05 2022 - Add number of pages OCRed in mongo db
                # docInfo["request"]["pages_ocred"] = len(pages)
                #Aug 05 2022 - Add number of pages OCRed in mongo db
                updated = updateSuccess(status,
                                        stgExtract,
                                        resultApiInfo,
                                        docInfo,
                                        documentId,
                                        auth_token,
                                        callbackUrl)
                print("Success Api Update", updated)
                if not updated:
                    fail_updated = updateFailure(stgExtract,
                                                 statusFailed,
                                                 errCode,
                                                 errmsgExtractionUpdateFail,
                                                 documentId,
                                                 callbackUrl,
                                                 auth_token)
                    return False
                return True
            else:
                updated = updateFailure(stgExtract,
                                        statusFailed,
                                        errCode,
                                        errmsgExtractionUpdateFail,
                                        documentId,
                                        callbackUrl,
                                        auth_token)


    except:
        print("getExtractionResults",
              traceback.print_exc())
        updated = updateFailure(stgExtract,
                                statusFailed,
                                errCode,
                                errmsgExtractionUpdateFail,
                                documentId,
                                callbackUrl,
                                auth_token)
        print("System Exception", updated)
        return None
    finally:
        try:
            if log_file is not None:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_file.close()
                uploaded,orgFileLocation = putil.uploadFilesToBlobStore([log_file_path])
                if uploaded:
                    os.remove(log_file_path)
        except:
            pass
        #Remove other files from local drive
        try:
            os.remove(local_pred_path)
        except:
            pass


