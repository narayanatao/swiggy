# -*- coding: utf-8 -*-
from tkinter.tix import Tree
import traceback
import os
from PIL import Image
import pdf2image
from pdf2image.exceptions import ( PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError)
import cv2
import numpy as np
import TAPPconfig as cfg
import shutil
import json
import requests
import pandas as pd
import pickle
import base64
from pyzbar.pyzbar import decode
import copy
import tifftools


#get script directory
script_dir = os.path.dirname(__file__)
print("Script Directory:", script_dir)

#Get all the blob stored details from config file

from ghostscript import Ghostscript as GS
#Get GhostScript executable. The executable path should be added to system parameters.
#In windows it would be "PATH" environmental variable
ghostExecutable = cfg.getGhostPath()
ghostPause = cfg.getGhostPause()
ghostDevice = cfg.getGhostTiffDvc()
ghostDownScale = cfg.getGhostTiffDownScale()
ghostDownScaleFactor = cfg.getGhostTiffDownScaleFactor()
ghostQuit = cfg.getGhostQuit()
ghostCommandNoPause = cfg.getGhostCommandNoPause()
ghostCommandDsf = cfg.getGhostCommandDsf()
ghostCommandQuit = cfg.getGhostCommandQuit()
ghostCommandOut = cfg.getGhostCommandOut()
ghostCommandForce = cfg.getGhostCommandForce()
rootFolderPath = cfg.getRootFolderPath()
stopWordsPath = cfg.getStopWordFilePath()

#Client Blob Folder
blobClientFolder = cfg.getBlobClientFolder()

#Get mimetype and extensions
mimeTiff = cfg.getMimeTiff()
mimePdf = cfg.getMimePdf()
mimePng = cfg.getMimePng()
mimeJson = cfg.getMimeJson()
mimeXml = cfg.getMimeXml()
extnTiff = cfg.getExtnTiff()
extnPdf = cfg.getExtnPdf()
extnTxt = cfg.getExtnTxt()
extnJson = cfg.getExtnJson()

#Get the server URL to update doc status, doc results, etc.,
uiServerUrl = cfg.getUIServer()
docUpdUrl = cfg.getDocUpdURL()
docResCrtUrl = cfg.getDocResCrtURL()
docGetUrl = cfg.getDocGetURL()
docResGetUrl = cfg.getDocResGetURL()
vendorGetUrl = cfg.getVendorGetURL()
docGetResultRPA_URL = cfg.getDocGetResultRPA_URL()

#Get Extraction API details
extApiAddr = cfg.getExtractionApiAddr()
extApiIP = cfg.getExtractionApiIP()
extApiPort = cfg.getExtractionApiPort()
extApiProtocol = "http"

#Get specs of xls to be downloaded
xlsDownloadColumnsPath = cfg.getXLSDownloadColumnsPath()
# mser = cv2.MSER_create()

# In[decorator]

import time
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

# In[amount patterns]
ptn1 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn2 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn3 = "[0-9]{1,3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn4 = "[0-9]{1,3}[.]{1}[0-9]{1,4}"

ptn5 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn6 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn7 = "[0-9]{1,3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn8 = "[0-9]{1,3}[,]{1}[0-9]{1,4}"
ptn9 = "\d{1,12}"

ptns = [ptn1,ptn2,ptn3,ptn4,ptn5,ptn6,ptn7,ptn8,ptn9]
from price_parser import parse_price
import re

def isAmount(s):
    try:
        for ptn in ptns:
            l = re.findall(ptn,s)
            l1 = [g for g in l if len(g) > 0]
            if len(l1) >= 1:
                return True
    except:
        return False
    return False

def extract_amount(text):
    """
    Checks whether passed string is valid amount or not
    Returns: 1 if amount, 0 otherwise
    """
    try:
        if isAmount(text):
            p = parse_price(text)
            if p.amount is not None:
                if isinstance(p.amount_float, float):
                    return p.amount_float
                else:    
                    return 0.0
    except:
        return 0.0
    return 0.0


# In[encryption]

def encryptMessage(inputJson):

    def __getPrivateEncryptKey__():
        return cfg.getPrivateEncryptKey()

    key = bytes(__getPrivateEncryptKey__(),
                'utf-8')

    inputJson = bytes(inputJson,
                      'utf-8')

    from cryptography.fernet import Fernet
    f = Fernet(key)

    encrypted_token = f.encrypt(inputJson)
    return encrypted_token

def decryptMessage(inputToken):

    def __getPrivateEncryptKey__():
        return cfg.getPrivateEncryptKey()

    key = bytes(__getPrivateEncryptKey__(),
                'utf-8')
    inputToken = bytes(inputToken,
                       'utf-8')
    from cryptography.fernet import Fernet
    f = Fernet(key)

    message = f.decrypt(inputToken)
    return message.decode('utf-8')


def genSubscriberHash():

    def __getSubscriberId__():
        return cfg.getSubscriberId()

    def __getMacAddress__():
        from uuid import getnode as get_mac
        return get_mac()

    from datetime import datetime
    sub_id = __getSubscriberId__()
    mac_id = __getMacAddress__()
    #Jun 27 2022 - hardcode macid
    mac_id = 45015787925255
    #Jun 27 2022 - hardcode macid
    input_ = str(sub_id) + "__" + str(mac_id) + "__" + str(datetime.now())
    print("Subscriber hash input",
          str(sub_id),
          str(mac_id))
    m = encryptMessage(input_)
    return m.decode('utf-8')


#Submit extraction request and get auth token
def extractionSubmit(docInfo):

    try:
        url = cfg.getExtractionSubmitAPI()

        #get subscriber hash
        subscriber_hash = genSubscriberHash()
        docInfo["input_hash"] = subscriber_hash
        #encrypt Input Json
        data = json.dumps(docInfo)
        print("Extraction submission data:", data)
        enc_data = encryptMessage(data)
        enc_data = enc_data.decode('utf-8')
        message = json.dumps({"message":enc_data})

        headers = {}
        headers["Content-Type"] = mimeJson

        print("Calling Extraction API:", url, message)
        r = requests.post(url,
                          data = message,
                          headers = headers)
        print("Calling Extraction API:", r, r.status_code)
        if r.status_code != 200:
            return None
        response = r.json()
        print("response",response)
        if response["status_code"] != 200:
            print("Failed in extraction submit")
            return None
        return response

    except:
        print("extractionSubmit",
              traceback.print_exc())
        return None

@timing
def saveTiffFiles(images, desc_path):

    try:
        outfile = desc_path
        imlist = []

        for page in images:
            t = time.time()
            imlist.append(page.convert("RGB"))
            print("RGB convert",time.time() - t)

        if len(imlist) > 1:
            t = time.time()
            imlist[0].save(outfile,
                           compression="tiff_deflate",
                           save_all=True,
                           append_images=imlist[1:])
            print("compression of tiff",
                  time.time() - t)
        else:
            t = time.time()
            imlist[0].save(outfile,
                           compression="tiff_deflate",
                           save_all=True)
            print("compression of tiff",
                  time.time() - t)

        return outfile
    except:
        print("saveTiffFiles",
              traceback.print_exc())
        return None

@timing
def pdf_to_img(pdf_file):
    try:
        # reading pdf as simple 
        #return pdf2image.convert_from_path(pdf_file)
        # reading as bytes
        with open(pdf_file,'rb') as f:
            content = f.read()
        return pdf2image.convert_from_bytes(content,
                                            dpi = 300,
                                            poppler_path = "/usr/bin")

    except:
        print("pdf_to_img; file:",pdf_file,"\n",
              traceback.print_exc())
        return None

@timing
def convertPDFToTiff(src_path, desc_path):

    def convertPDFToTiff_GS(src,dst):
        try:
            args = []
            bDst = dst.encode('utf-8')
            bSrc = src.encode('utf-8')
            args.append(b'' + ghostExecutable.encode('utf-8'))
            args.append(b'' + ghostDevice.encode('utf-8'))
            if ghostPause == 1:
                noPause = b"" + ghostCommandNoPause.encode('utf-8')
                args.append(noPause)
            if ghostDownScale == 1:
                args.append(b"" + ghostCommandDsf.encode('utf-8') + str(ghostDownScaleFactor).encode('utf-8'))

            args.append(b'' + "-r800X800".encode('utf-8'))
            args.append(b'' + ghostCommandOut.encode('utf-8'))
            args.append(b'' + bDst)
            args.append(b'' + bSrc)
            if ghostQuit == 1:
                args.append(b'' + ghostCommandForce.encode('utf-8'))
                args.append(b'' + ghostCommandQuit.encode('utf-8'))
            g = GS(*args)
            g.exit()
        except:
            print("convertPDFToTiff_GS",
                  traceback.print_exc())
            return False, None
        return True, dst

    try:
        if src_path.split(".")[-1].upper() == 'PDF':
            images = pdf_to_img(src_path)
            pageNo = 0
            for image in images:
                image.save("convertedtiff"+"_"+str(pageNo),'tiff')
                pageNo = pageNo + 1
            if images is not None:
                imgPxl = np.array(images[0])
                imgPxl = cv2.cvtColor(imgPxl, cv2.COLOR_BGR2GRAY)
                gs_ok = False
                if (np.unique(imgPxl).shape[0] == 1):
                    check, filepath = convertPDFToTiff_GS(src_path,desc_path)
                    if check:
                        imgGS = cv2.imread(filepath,0)
                        if (np.unique(imgGS).shape[0] == 1):
                            gs_ok = False
                        else:
                            gs_ok = True
                            images_GS = Image.open(filepath)
                            print("Ghost Script Conversion Successful")
                            out_file = saveTiffFiles(images_GS, desc_path)
                            if out_file is None:
                                return False
                    else:
                        return False
                if not gs_ok:
                    print("PDF to TIFF Conversion Successful")
                    tiff_filepath = saveTiffFiles(images, desc_path)
                    if tiff_filepath is None:
                        return False
            else:
                check, filepath = convertPDFToTiff_GS(src_path, desc_path)
                if check:
                    images_GS = Image.open(filepath)
                    print("Ghost Script Conversion Successful")
                    out_file = saveTiffFiles(images_GS, desc_path)
                    if out_file is None:
                        return False
                else:
                    return False

            return True
    except:
        # print("Error in Converting PDF to TIFF: ",e)
        print("convertPDFToTiff",traceback.print_exc())
        return False

def convertPDFToTiff_GS(src,dst):
    try:
        args = []
        bDst = dst.encode('utf-8')
        bSrc = src.encode('utf-8')
        args.append(b'' + ghostExecutable.encode('utf-8'))
        args.append(b'' + ghostDevice.encode('utf-8'))
        if ghostPause == 1:
            noPause = b"" + ghostCommandNoPause.encode('utf-8')
            args.append(noPause)
        if ghostDownScale == 1:
            args.append(b"" + ghostCommandDsf.encode('utf-8') + str(ghostDownScaleFactor).encode('utf-8'))

        args.append(b'' + "-r800X800".encode('utf-8'))
        args.append(b'' + ghostCommandOut.encode('utf-8'))
        args.append(b'' + bDst)
        args.append(b'' + bSrc)
        if ghostQuit == 1:
            args.append(b'' + ghostCommandForce.encode('utf-8'))
            args.append(b'' + ghostCommandQuit.encode('utf-8'))
        g = GS(*args)
        g.exit()
    except:
        print("convertPDFToTiff_GS",
              traceback.print_exc())
        return False, None
    return True, dst

### converting original pdf onto tiffes
def convertOrignal_PDFToTiff_GS(src_path, desc_path):
    try:
        print("source path :",src_path, "\t Destination folder :",desc_path)
        extn = os.path.splitext(os.path.basename(src_path))[1]
        fileName = os.path.splitext(os.path.basename(src_path))[0]
        dest_path = os.path.join(desc_path,str(fileName) + '.tiff')
        print("New Destination path :",dest_path)

        if src_path.split(".")[-1].upper() == 'PDF':
            print("Source Path conv pdf to tiff",
                  src_path,
                  os.path.isfile(src_path))
            # images = pdf_to_img(src_path)
            converted, path = convertPDFToTiff_GS(src_path, dest_path)
            if converted:
                print("recd dest path:",dest_path)
                return True, dest_path
            else:
                return False, src_path
            # print("images list size",len(images))
            # if len(images)>0:
            #     desc_path = os.path.join(desc_path,str(fileName) + 'tiff')
            #     images[0].save(desc_path, save_all=True, append_images=images[1:])
            #     print("tiff convert path:",desc_path)    
            #     return True, desc_path       
            # else:
            #     print("images list empty:",images)
            #     return False, src_path
        elif 'tif' in extn:
            print("It's already tiff file")
            return False, src_path
        else:
            print("ext other than pdf or tiff")
            return False, None
    except:
        # print("Error in Converting PDF to TIFF: ",e)
        print("convertPDFToTiff",
              traceback.print_exc())
        return False, src_path

### converting original pdf onto tiffes
def convertOrignal_PDFToTiff(src_path, desc_path):
    try:
        # print("surce path :",src_path, "\tDestination path :",desc_path)
        extn = os.path.splitext(os.path.basename(src_path))[1]
        fileName = os.path.splitext(os.path.basename(src_path))[0]

        if src_path.split(".")[-1].upper() == 'PDF':
            print("Source Path conv pdf to tiff",
                  src_path,
                  os.path.isfile(src_path))
            images = pdf_to_img(src_path)
            print("recd dest path:",desc_path)
            print("images list size",len(images))
            if len(images) > 0:
                desc_path = os.path.join(desc_path,
                                         str(fileName) + '.tiff')
                images[0].save(desc_path,
                               save_all=True,
                               append_images=images[1:])
                print("tiff convert path:",desc_path)    
                return True, desc_path       
            else:
                print("images list empty:",images)
                return False, src_path
        elif 'tif' in extn:
            print("It's already tiff file")
            return False, src_path
        else:
            print("ext other than pdf or tiff")
            return False, None
    except:
        # print("Error in Converting PDF to TIFF: ",e)
        print("convertPDFToTiff",traceback.print_exc())
        return False, src_path

# def convertPDFToTiff_GS(src,dst):
#     try:
#         args = []
#         bDst = dst.encode('utf-8')
#         bSrc = src.encode('utf-8')
#         args.append(b'' + ghostExecutable.encode('utf-8'))
#         args.append(b'' + ghostDevice.encode('utf-8'))
#         if ghostPause == 1:
#             noPause = b"" + ghostCommandNoPause.encode('utf-8')
#             args.append(noPause)
#         if ghostDownScale == 1:
#             args.append(b"" + ghostCommandDsf.encode('utf-8') + str(ghostDownScaleFactor).encode('utf-8'))

#         args.append(b'' + "-r800X800".encode('utf-8'))
#         args.append(b'' + ghostCommandOut.encode('utf-8'))
#         args.append(b'' + bDst)
#         args.append(b'' + bSrc)
#         if ghostQuit == 1:
#             args.append(b'' + ghostCommandForce.encode('utf-8'))
#             args.append(b'' + ghostCommandQuit.encode('utf-8'))
#         g = GS(*args)
#         g.exit()
#     except:
#         print("convertPDFToTiff_GS",
#                 traceback.print_exc())
#         return False, None
#     return True, dst

#Download a single file from Blob store.
#This implementation will change using Azure blob container service
@timing
def downloadFromBlobStore(fileURI, localPath):
    try:
        from azure.storage.blob import BlobServiceClient
        connect_str = cfg.getBlobConnectionString()
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        splitURI = fileURI.split("/")
        container = splitURI[1]
        blobname = "/".join([name for ind,
                             name in enumerate(splitURI) if ind > 1])
        print("container:",container,"blob:",blobname,"local:",localPath)
        blob_client = blob_service_client.get_blob_client(container=container,
                                                          blob=blobname)
        with open(localPath,"wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        return True
    except:
        print("downloadFromBlobStore",
              traceback.print_exc())
        return False

def updateDocumentApi_New(documentId, docInfo, callbackUrl):

    try:
        updUrl = callbackUrl + "/" + docUpdUrl + "/" + documentId
        data = json.dumps(docInfo)
        headers = {}
        headers["Content-Type"] = mimeJson
        retries = 0
        max_retries = 3
        import time

        while retries < max_retries:
            print("updateDocumentApi",
                  updUrl,
                  data)
            r = requests.post(updUrl,
                              data = data,
                              headers = headers,
                              verify = False)
            if r.status_code == 200:
                print("Success")
                retries = max_retries + 1
                return True,200
            elif r.status_code == 404:
                retries += 1
                time.sleep(1)
            else:
                print(r.status_code)
                retries = max_retries + 1
                return False,r.status_code
    except:
        print("updateDocumentApi",
              traceback.print_exc())
        return False,500


def updateDocumentApi(documentId, docInfo, callbackUrl):

    try:
        updUrl = callbackUrl + "/" + docUpdUrl + "/" + documentId
        data = json.dumps(docInfo)
        headers = {}
        headers["Content-Type"] = mimeJson

        print("updateDocumentApi",
              updUrl,
              data)
        r = requests.post(updUrl,
                          data = data,
                          headers = headers,
                          verify = False)
        if r.status_code == 200:
            print("Success")
            return True
        else:
            print(r.status_code)
            return False
    except:
        print("updateDocumentApi",
              traceback.print_exc())
        return False

#Upload files to Azure blob store. This is specific to Azure blob storage
def downloadFilesFromBlob(fileURIs):
    print("dwonload URL :",fileURIs)
    def __generateMessage__(hashString,
                            file_sz):
        message = {}
        message["input_hash"] = hashString
        message["file_size"] = file_sz
        message["activity"] = "Download"
        return json.dumps(message)

    #Upload to Blob store
    try:
        size_ = 1024 * 1024 * 10

        #Call Azure function to get SAS token
        #Create a json for it
        #Hash the subscriber id and allowed machines here
        hash_ = genSubscriberHash()
        message = __generateMessage__(hash_, size_)
        enc_message = encryptMessage(message).decode("utf-8")
        get_sas_message = json.dumps({"message":enc_message})
        #Give post request
        import requests
        blobUri = cfg.getBlobStorageAPI()
        header = {"Content-Type":"application/json"}
        # print("message input",
        #       message,
        #       get_sas_message,
        #       blobUri,
        #       header)
        response = requests.post(url = blobUri,
                                data = get_sas_message,
                                headers = header)
        print("download files:", response)
        if response.status_code != 200:
            return False, None
        responseObj = response.json()
        status_code = responseObj["status"]
        if status_code != 200:
            return False, None

        encrypted_response = responseObj["message"]
        dec_message = decryptMessage(encrypted_response)
        # print("decrypted message download from blob ", dec_message)
        message_obj = json.loads(dec_message)

        sas_token = message_obj["sas_token"]
        blob_acc_url = message_obj["account_url"]
        container = message_obj["container"]

        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(account_url=blob_acc_url,
                                                credential=sas_token)

        for fileURI, localPath in fileURIs:
            splitURI = fileURI.split("/")
            blobname = "/".join([name for ind,name in enumerate(splitURI) if ind > 0])
            print("Container is: ",container,"Blob Name:", blobname)
            blob_client = blob_service_client.get_blob_client(container=container,
                                                              blob=blobname)
            with open(localPath,"wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
                print("Download successfull",blobname)
                
        return True
    except:
        print("downloadFilesFromBlob",
              traceback.print_exc())
        return False

#Upload files to Azure blob store. This is specific to Azure blob storage
def uploadFilesToBlobStore(filePaths):

    def __generateMessage__(hashString,
                            file_sz):
        message = {}
        message["input_hash"] = hashString
        message["file_size"] = file_sz
        message["activity"] = "Upload"
        return json.dumps(message)

    #Upload to Blob store
    try:
        print("File Paths:", filePaths)
        size_ = 0
        for filePath in filePaths:
            print("File Path in uploadFilesToBlobStore", filePath)
            sz_ = os.path.getsize(filePath)
            if round(sz_/(1024*1024)) == 0:
                size_ += 1024*1024 + sz_

        #Call Azure function to get SAS token
        #Create a json for it
        #Hash the subscriber id and allowed machines here
        hash_ = genSubscriberHash()
        message = __generateMessage__(hash_, size_)
        enc_message = encryptMessage(message).decode("utf-8")
        get_sas_message = json.dumps({"message":enc_message})
        #Give post request
        import requests
        blobUri = cfg.getBlobStorageAPI()
        header = {"Content-Type":"application/json"}
        print("Request: ", get_sas_message, blobUri, header)
        response = requests.post(url = blobUri,
                                data = get_sas_message,
                                headers = header)
        print("Response: ", response)
        if response.status_code != 200:
            return False, None
        responseObj = response.json()
        print("response",responseObj)
        status_code = responseObj["status"]
        if status_code != 200:
            return False, None

        encrypted_response = responseObj["message"]
        dec_message = decryptMessage(encrypted_response)
        message_obj = json.loads(dec_message)

        sas_token = message_obj["sas_token"]
        blob_acc_url = message_obj["account_url"]
        container = message_obj["container"]

        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(account_url=blob_acc_url,
                                                credential=sas_token)
        blobPaths = []
        for filePath in filePaths:
            fileName = os.path.basename(filePath)
            blob_client = blob_service_client.get_blob_client(container = container,
                                                              blob = fileName)
            with open(filePath, "rb") as data:
                blob_client.upload_blob(data,
                                        overwrite=True)
            blobPaths.append(container + "/" + fileName)

        return True,blobPaths
    except:
        print("uploadFilesToBlobStore",
              traceback.print_exc())
        return False, None

def getResults(documentId,callbackUrl):

    try:
        getResultUrl = callbackUrl + "/" + docGetResultRPA_URL
        docInfo = {"id": "api.document.rpa.list",
                   "ver": "1.0",
                   "params": {
                           "msgid": ""
                           },
                           "request": {
                                   "filter": {
                                           "documentId": documentId
                                           }
                                   }}
        data = json.dumps(docInfo)
        headers = {}
        headers["Content-Type"] = mimeJson

        r = requests.post(getResultUrl,
                          data = data,
                          headers = headers,
                          verify = False)
        if r.status_code == 200:
            print("Success")
            response = r.json()
            result = response['result']
            documents = result['documents']

            return documents
        else:
            print(r.status_code)
            return None
    except:
        print("getResults",
              traceback.print_exc())
        return None

def getDocumentStatus(documentId, callbackUrl):

    try:
        docResult = getDocumentApi(documentId, callbackUrl)
        if docResult is not None:
            return docResult["result"]["document"]["status"]
        else:
            return None
    except:
        print("getDocumentStatus",
              traceback.print_exc())
        return None

def getDocumentOrgDocTypes(documentId, callbackUrl):

    try:
        docResult = getDocumentApi(documentId, callbackUrl)
        if docResult is not None:
            orgTypeId = docResult["result"]["documents"][0]["orgTypeId"]
            docTypeId = docResult["result"]["documents"][0]["docTypeId"]
            return orgTypeId,docTypeId
        else:
            return None,None
    except:
        print("getDocumentOrgDocTypes",
              traceback.print_exc())
        return None,None

def __adjustColOrder__(df,colOrder):

    df_copy = df.copy(deep = True)
    try:
        asterisk_present = False
        srcCols = list(df.columns.values)
        tgtCols = list(colOrder.keys())
        asterisk_present = "*" in tgtCols
        # tgtTranCols = list(colOrder.values())
        #Drop extra columns in the dataframe if not defined in target mapping
        # addColsSrc = list(set(srcCols) - set(tgtCols))
        addColsSrc = [l for l in srcCols if l not in tgtCols]
        print("Additional Columns in Source:",addColsSrc)
        if asterisk_present:
            tgtCols.pop(tgtCols.index("*"))
        else:
            df = df.drop(addColsSrc,axis = 1)
        #Add extra columns in the dataframe if target mapping has it but source doesn't have
        #This scenario should be avoided and try to have the same columns at both the ends
        # addColsTgt = list(set(tgtCols) - set(srcCols))
        addColsTgt = [l for l in tgtCols if l not in srcCols]
        print("Additional Columns in Target:",
              addColsTgt)
        df[addColsTgt] = ""
        df_new = df[tgtCols]
        if asterisk_present:
            df_new = df_new.join(df[addColsSrc])
        df_new = df_new.rename(colOrder,
                               axis = 1)
        return df_new
    except:
        print(traceback.print_exc())
        return df_copy

def translateResults(df_hdr,
                     df_lines,
                     orgTypeId=None,
                     docTypeId=None):

    df_hdr_copy = df_hdr.copy(deep = True)
    df_lines_copy = df_lines.copy(deep = True)
    columnOrder = None
    try:
        print("Before translating to required column names and order")

        if os.path.exists(xlsDownloadColumnsPath):
            with open(xlsDownloadColumnsPath,"r") as f:
                tmpJson = f.read()
                colOrders = json.loads(tmpJson)
                found = False
                for colOrder in colOrders:
                    orgType = colOrder.get("orgType")
                    docType = colOrder.get("docType")
                    if (orgType == orgTypeId) and (docType == docTypeId):
                        columnOrder = colOrder.get("template")
                        found = True
                        break
                if not found:
                    return df_hdr_copy,df_lines_copy
                if columnOrder is None:
                    return df_hdr_copy,df_lines_copy
                if ("LI" in columnOrder.keys()):
                    LIOrder = columnOrder["LI"]
                    columnOrder.pop("LI")
                    #Reorder line items here
                    print("Column Order LI:",LIOrder)
                    df_lines = __adjustColOrder__(df_lines,LIOrder)
                #Reorder header columns here
                print("Column Order hdr:",columnOrder)
                df_hdr = __adjustColOrder__(df_hdr,columnOrder)
        return df_hdr,df_lines
    except:
        print(traceback.print_exc())
        return df_hdr_copy,df_lines_copy

def getExtractedResults(documentId,
                        callbackUrl,
                        cleanUpFiles = True):

    print("Get Extracted Results In")
    hdr_sheet = "Header"
    line_sheet = "Lines"
    max_wd = 50
    try:
        documents = getResults(documentId,
                               callbackUrl)
        print("Results Downloaded")
        orgTypeId, docTypeId = getDocumentOrgDocTypes(documentId,
                                                      callbackUrl)

        #Keep Header and Line Items separately
        #Header is extracted from the documentresult directly

        df_hdr = pd.DataFrame(documents)
        document = documents[0]
        df_lines = pd.DataFrame()

        lines = document.get('documentLineItems',[])
        df_hdr = df_hdr.drop("documentLineItems",axis = 1)
        document.pop('documentLineItems')
        df_hdr["SNo"] = 1
        if len(lines) > 0:
            df_lines = pd.DataFrame(lines)
            df_lines["ROW_INDEX"] = df_lines["rowNumber"]
            df_lines["SNo"] = df_lines["rowNumber"]
            df_lines = df_lines.drop(["rowNumber"],
                                     axis = 1)

        print("dfHdr Columns",df_hdr.columns.values)
        if (orgTypeId is not None) and (docTypeId is not None):
            df_hdr,df_lines = translateResults(df_hdr,
                                               df_lines,
                                               orgTypeId,
                                               docTypeId)

        print("dfHdr Columns after translation",df_hdr.columns.values)
        #Save a local excel file
        fileName = documentId + '.xlsx'
        localFolder = cfg.getRootFolderPath()
        os.makedirs(localFolder,
                    exist_ok = True)
        filePath = os.path.join(localFolder,
                                fileName)
        writer = pd.ExcelWriter(filePath,
                                engine = 'xlsxwriter')
        df_hdr.to_excel(writer,
                        sheet_name = hdr_sheet,
                        index = False)
        df_lines.to_excel(writer,
                          sheet_name = line_sheet,
                          index = False)

        #Format the columns - 1. Column width should be max based on data. If the width exceeds 50, apply word wrap
        workbook  = writer.book
        wrap_format = workbook.add_format({'text_wrap': True})
        hdr_ws = writer.sheets[hdr_sheet]
        ln_ws = writer.sheets[line_sheet]
        
        sht_df_dic = {hdr_sheet:(df_hdr,hdr_ws),
                      line_sheet:(df_lines,ln_ws)}

        #Apply formatting
        for sheetname, objs in sht_df_dic.items():
            df = objs[0]
            ws = objs[1]
            for idx, col in enumerate(df):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                max_len = min([max_len,max_wd])
                ws.set_column(idx,idx,max_len,wrap_format)

        writer.save()
        writer.close()
        with open(filePath,"rb") as f:
            return f.read()

    except:
        print(traceback.print_exc())
        return None

def downloadResults(documentId,
                    callbackUrl,
                    container,
                    localFolder,
                    cleanUpFiles = True):

    updated = False
    print("Download Results In")
    try:
        documents = getResults(documentId, callbackUrl)
        print("Results Downloaded")

        #Get Document Status
        docStatus = getDocumentStatus(documentId, callbackUrl)
        print("Document Status received")

        #Keep Header and Line Items separately
        #Header is extracted from the documentresult directly

        df_hdr = pd.DataFrame(documents)
        document = documents[0]
        df_lines = pd.DataFrame()

        lines = document.get('documentLineItems',[])
        if len(lines) > 0:
            df_hdr = df_hdr.drop("documentLineItems",axis = 1)
            document.pop('documentLineItems')
            df_hdr["SNo"] = 1
            df_lines = pd.DataFrame(lines)
            df_lines["ROW_INDEX"] = df_lines["rowNumber"]
            df_lines["SNo"] = df_lines["rowNumber"]
            df_lines = df_lines.drop(["rowNumber"],
                                     axis = 1)

        print("Before translating to required column names and order")
        if os.path.exists(xlsDownloadColumnsPath):
            with open(xlsDownloadColumnsPath,"r") as f:
                tmpJson = f.read()
                columnOrder = json.loads(tmpJson)
                if "LI" in columnOrder.keys():
                    LIOrder = columnOrder["LI"]
                    columnOrder.pop("LI")
                    #Reorder line items here
                    print("Column Order LI:",LIOrder)
                    df_lines = __adjustColOrder__(df_lines,LIOrder)
                #Reorder header columns here
                print("Column Order hdr:",columnOrder)
                df_hdr = __adjustColOrder__(df_hdr,columnOrder)

        #Save a local excel file
        fileName = documentId + '.xlsx'
        os.makedirs(localFolder,exist_ok = True)
        filePath = os.path.join(localFolder,fileName)
        writer = pd.ExcelWriter(filePath,
                                engine = 'xlsxwriter')
        df_hdr.to_excel(writer,
                        sheet_name = "Header",
                        index = False)
        df_lines.to_excel(writer,
                          sheet_name = "Lines",
                          index = False)

        writer.save()
        writer.close()

        #Change to local storage

        #Update the document metadata using TAPP API
        if (docStatus is not None):
            docInfo = {"id": "api.document.update",
                       "ver": "1.0",
                       "params": {
                               "msgid": ""
                               },
                               "request": {
                                       "documentId" : documentId,
                                       "status" : docStatus,
                                       "resultDownloadLink" : filePath
                                       }
                               }
            updated = updateDocumentApi(documentId,docInfo,callbackUrl)
            if updated:
                print("Results file path updated in document metadata",
                      docInfo)
            else:
                print("Results file could not updated to document metadata store")

        return True, filePath, updated
    except:
        print("downloadResults",
              traceback.print_exc())
        return False, None, updated

def createDocumentResultsApi(documentId,
                             docResultsInfo,
                             callbackUrl):

    try:
        url = callbackUrl + "/" + docResCrtUrl
        data = json.dumps(docResultsInfo)
        headers = {}
        headers["Content-Type"] = mimeJson
        retries = 0
        max_retries = 3

        while retries < max_retries:
            r = requests.post(url,
                              data = data,
                              headers = headers,
                              verify = False)
            print("Doc Result Create:",
                  url,
                  data,
                  headers)
            if r.status_code == 200:
                print("Success",
                      "Create Document Result")
                retries = max_retries + 1
                return True
            elif r.status_code == 404:
                retries += 1
                import time
                time.sleep(1)
                print("Retrying...",
                      str(retries),
                      r.status_code)
            else:
                print("createDocumentResultsApi",
                      r.status_code,
                      r,
                      traceback.print_exc())
                retries = max_retries + 1
                return False
    except:
        print("createDocumentResultsApi",
              traceback.print_exc())
        return False

def checkValidFormat(defFormat,fldFormat):
    if len(defFormat) != len(fldFormat):
        return False
    else:
        for i,c in enumerate(list(defFormat)):
            if c.upper() != "I":
                if c != fldFormat[i]:
                    return False
        return True

def removePuncFromEnds(val):
    try:
        import string
        punc = string.punctuation + " "
        return val.strip(punc)
    except:
        return val

def validAmount(val):
    try:
        mod_val = removePuncFromEnds(val)
        if len(mod_val) != len(val):
            return False
        v = float(val)
        return True
    except:
        return False

def validAlphaNumeric(val):
    try:
        mod_val = removePuncFromEnds(val)
        if len(mod_val) != len(val):
            return False
        for v in list(str(val)):
            if v.isnumeric() or v.isalpha():
                return True
        return False
    except:
        return False

def validDate(val):
    try:
        mod_val = removePuncFromEnds(val)
        if len(mod_val) != len(val):
            return False
        from dateutil import parser
        v = parser.parse(val,
                         dayfirst = True).date()
        from datetime import datetime as dt
        today = dt.now().date()
        if v > today:
            return 200
        return 0
    except:
        return 100

#Copy files from a source location to target location
def copyFile(srcPath,destPath):
    try:
        shutil.copy(srcPath,destPath)
        return True
    except:
        print("copyFile",
              traceback.print_exc())
        return False

#Identify file extn, filename, folder location
def getFileInfo(filePath):
    try:
        fileInfo = {}
        if os.path.isfile(filePath):
            fileParts = os.path.split(filePath)
            fileInfo["fullPath"] = filePath
            fileInfo["folderLoc"] = fileParts[0]
            fileInfo["filenameExtn"] = fileParts[1]
            fileInfo["filenameWoExtn"] = os.path.splitext(fileParts[1])[0]
            fileInfo["extn"] = os.path.splitext(fileParts[1])[1]
            stats = os.stat(filePath)
            fileInfo["createdTime"] = stats.st_ctime
            fileInfo["modifiedTime"] = stats.st_mtime
            fileInfo["size"] = stats.st_size
            return fileInfo
        else:
            return None
    except:
        print("getFileInfo",
              traceback.print_exc())
        return None

def getDocumentResultApi(documentId, callbackUrl):

    getUrl = callbackUrl + "/" + docResGetUrl + "/" + documentId
    try:
        headers = {}
        headers["REQUEST"] = mimeJson
        r = requests.get(getUrl,
                         headers = headers,
                         verify = False)
        if r.status_code == 200:
            result = r.json()
            return result
        else:
            return None
    except:
        print("getDocumentResultApi",
              traceback.print_exc())
        return None

def getDocumentApi(documentId, callbackUrl):

    getUrl = callbackUrl + "/" + docGetUrl + "/" + documentId
    try:
        headers = {}
        headers["REQUEST"] = mimeJson
        r = requests.get(getUrl,
                         headers = headers,
                         verify = False)
        if r.status_code == 200:
            result = r.json()
            return result
        else:
            return None
    except:
        print("getDocumentApi:",
              traceback.print_exc())
        return None

def getFldTypeFormat():
    fldConfigFile = cfg.getSTPConfiguration()
    try:
        fldTypes = []
        with open(fldConfigFile,'r') as fldconfig:
            fldconf_obj = json.load(fldconfig)
            all_flds = fldconf_obj["DEFAULT"]
            list_flds = list(all_flds.keys())
            for fld in list_flds:
                fld_info = all_flds[fld]
                display_flag = fld_info.get("display_flag",0)
                fldType = fld_info.get("fldType","alpha-numeric")
                format_ = fld_info.get("format")
                if display_flag == 1:
                    fldTypes.append((fld,fldType,format_))
        return None if len(fldTypes) == 0  else fldTypes
    except:
        print("getFldTypeFormat",
              traceback.print_exc())
        return None


def getMandatoryFields():
    fldConfigFile = cfg.getSTPConfiguration()
    try:
        mandatoryFields = []
        with open(fldConfigFile,'r') as fldconfig:
            fldconf_obj = json.load(fldconfig)
            all_flds = fldconf_obj["DEFAULT"]
            list_flds = list(all_flds.keys())
            for fld in list_flds:
                fld_info = all_flds[fld]
                display_flag = fld_info.get("display_flag",0)
                mandatory_flag = fld_info.get("mandatory",0)
                if display_flag == 1 and mandatory_flag == 1:
                    mandatoryFields.append(fld)
        return None if len(mandatoryFields) == 0  else mandatoryFields
    except:
        print("getMandatoryFields",
              traceback.print_exc())
        return None


def pathMerger(tailPath,relativeFolder,storageType):
    if storageType == "BLOB":
        return tailPath
    elif storageType == "FOLDER":
        return os.path.join(rootFolderPath,relativeFolder,
                            tailPath.replace(rootFolderPath,
                                             "").lstrip("/").lstrip("\\\\").lstrip("\\"))
    else:
        return tailPath



def makeTiffCompressed(filepath):
    try:
        img = Image.open(filepath)
        img.load()
        imlist =[]
        for page in range(0,img.n_frames):
            img.seek(page)
            imlist.append(img.convert("RGB"))
        imlist[0].save(filepath,
                       compression="tiff_deflate",
                       save_all=True,
               append_images=imlist[1:])
        return True
    except:
        print(traceback.print_exc())
        return False


def getPostProcessConstantLabels():

    try:
        postPrcConstLabels = None
        labelPath = cfg.getPostProcessConstantLabels()
        fullPath = os.path.join(script_dir,labelPath)
        postPrcConstLabels = None
        with open(fullPath, 'rb') as hand:
            postPrcConstLabels = pickle.load(hand)
        return postPrcConstLabels
    except:
        print(traceback.print_exc())
        return None

def getPostProcessFieldLabels():

    try:
        postPrcFldLabels = None
        labelPath = cfg.getPostProcessFieldLabels()
        fullPath = os.path.join(script_dir,labelPath)
        with open(fullPath, 'rb') as hand:
            postPrcFldLabels = pickle.load(hand)
        return postPrcFldLabels
    except:
        print(traceback.print_exc())
        return None

def getPostProcessScoring():
    try:
        postPrcScoring = None
        labelPath = cfg.getPostProcessScoring()
        fullPath = os.path.join(script_dir,labelPath)
        with open(fullPath, 'rb') as hand:
            postPrcScoring = pickle.load(hand)
        return postPrcScoring
    except:
        print(traceback.print_exc())
        return None

def getSTPConfiguration():
    try:
        stpConfiguation = None
        stpConfigurationPath = cfg.getSTPConfiguration()
        fullPath = os.path.join(script_dir,stpConfigurationPath)
        with open(fullPath) as json_file:
            stpConfiguation = json.load(json_file)
        return stpConfiguation
    except:
        print(traceback.print_exc())
        return None

def getPostProcessDtTypeCheck():
    try:
        postPrcDtTypeCheck = None
        labelPath = cfg.getPostProcessDtTypeCheck()
        fullPath = os.path.join(script_dir,labelPath)
        with open(fullPath, 'rb') as hand:
            postPrcDtTypeCheck = pickle.load(hand)
        return postPrcDtTypeCheck
    except:
        print(traceback.print_exc())
        return None

#  Read the files to verify the required vendor fields against the extracted fields

def extract_bar_qr_code(image_path):
    '''
    Returns
    -------
    output : List
    '''
    decoded = {}
    output = []
    img = Image.open(image_path)
    print("Image opened")
    decoded[image_path] = decode(img)
    for bar_code in decoded[image_path]:
        data_ = {}
        data = bar_code.data.decode('utf-8')
        print("Bar Code found. Data is ", data)
        typ = bar_code.type
        if typ == 'CODE128':
            typ = "BAR CODE"
        elif typ == 'CODE39':
            typ == "BAR CODE"
        elif typ == 'QRCODE':
            typ = "QR CODE"

        data_["File Name"] = os.path.basename(image_path)
        data_["Data Type"] = typ
        data_["Raw Extracted Data"] = data
        data_["Decoded Data"] = data
        try:
            s = data.split(".")
            if len(s) == 3:
                data_1 = s[1]
                missing_padding = len(data_1) % 4
                if missing_padding:
                    data_1 += '=' * (4 - missing_padding)
                decoded_data = base64.b64decode(data_1,
                                                "-_")
                import json
                d = json.loads(decoded_data)
                print("Decoded data is:",d)
                data_["Decoded Data"] = d
        except Exception as e:
            print(e)
            pass
        
        output.append(data_)

    img = None
    return output

def extract_gstin_pattern_only(string)->str:
    """
    extracting GSTIN fromates from string and returning first identified format
    """
    try:
        pattern = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z|2]{1}[A-Z\d]{1}' 
        gstin = re.findall(pattern, string, flags = 0)
        if len(gstin)>0:
            return gstin[0]
        print("Given text does't contain GSTIN")
        return string
    except :
        print("GSTIN extraction exception:",traceback.print_exc())
        return string

def correct_gstin(GSTIN):
    GSTIN = extract_gstin_pattern_only(GSTIN)
    GSTIN_COPY = copy.deepcopy(GSTIN)
    if GSTIN and (len(GSTIN) == 15):
        # print(GSTIN[12])
        if (GSTIN_COPY[12]=="I"):
            print("Received GSTIN :",GSTIN_COPY)
            GSTIN = GSTIN[:12]+"1"+GSTIN[(12+1):]
            print("Corrected GSTIN 1173:",GSTIN)
        if (GSTIN_COPY[13]=="2"):
            print("Received GSTIN :",GSTIN_COPY)
            GSTIN = GSTIN[:13]+"Z"+GSTIN[(13+1):]
            print("Corrected GSTIN  1177:",GSTIN)
        if (GSTIN_COPY[13] == "7"):
            print("Received GSTIN :",GSTIN_COPY)
            GSTIN = GSTIN[:13]+"Z"+GSTIN[(13+1):]
            print("Corrected GSTIN 1181:",GSTIN) 

        for i in [0,7,8,9,10]:
            # print(GSTIN[i])
            if (GSTIN_COPY[i]=="O") or (GSTIN_COPY == "o") :
                print("Received GSTIN :",GSTIN_COPY)
                GSTIN = GSTIN[:i]+"0"+GSTIN[(i+1):]
                print("Corrected GSTIN 1188:",GSTIN)
            if (GSTIN_COPY[i]=="Z") or (GSTIN_COPY == "z") :
                print("Received GSTIN :",GSTIN_COPY)
                GSTIN = GSTIN[:i]+"2"+GSTIN[(i+1):]
                print("Corrected GSTIN 1192:",GSTIN)
            if (GSTIN_COPY[i]=="I"):
                print("Received GSTIN :",GSTIN_COPY)
                GSTIN = GSTIN[:i]+"1"+GSTIN[(i+1):]
                print("Corrected GSTIN 1196:",GSTIN)
        return GSTIN
            
    else:
        print("Invalid GSTIN format",GSTIN_COPY)
        return GSTIN_COPY

def identify_pan_pattern(string)->str:
    """
    extracting GSTIN fromates from string and returning first identified format
    """
    try:
        pattern = r'[A-Z]{5}\d{4}[A-Z]{1}' 
        pan_pattern = re.findall(pattern, string, flags = 0)
        if len(pan_pattern)>0:
            return pan_pattern[0]
        # print("Given text does't contain PAN")
        return string
    except :
        print("PAN extraction exception:",traceback.print_exc())
        return string
