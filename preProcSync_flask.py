# -*- coding: utf-8 -*-
import traceback
try:
    from PIL import Image
except ImportError:
    import Image

import time
import os
import shutil
import cv2
import json
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdfrw import PdfReader, PdfWriter
import TAPPconfig as cfg
import math
import preProcUtilities as putil
import uuid
import fitz
import io
from sys import argv
import sys

import kafka_producer as producer

import requests

from datetime import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

#Give an Asynchronous call to extraction (pre-processor, OCR, feature-engineering, model o/p)
#Manage it using task queues (Celery) and Broker (RabbitMQ)

# from klein import Klein
# app = Klein()

from flask import Flask, request
app = Flask(__name__)


rootFolderPath = cfg.getRootFolderPath()

#Get all the temporary folders to stored the preprocessed files from config file
splitFolder = cfg.getSplitFolder()
downloadFolder = cfg.getDownloadFolder()
tiffFolder = cfg.getTiffFolder()
preprocFolder = cfg.getPreProcAsyncFolder()

#Get all constants required for request/response parameters
docType = cfg.getDocumentType()
sysUser = cfg.getSystemUser()
tappVer = cfg.getTappVersion()
docUpdApi = cfg.getDocUpdApi()
paramStatusSuccess = cfg.getParamStatusSuccess()
paramStatusFailed = cfg.getParamStatusFailed()
statusReadyForExtract = cfg.getStatusReadyExtract()
statusFailed = cfg.getStatusFailed()

statusPreprocInprog = cfg.getStatusPreprocInProg()
statusExtInprog = cfg.getStatusExtractInProg()
statusmsgExtInit = cfg.getStatusmsgExtractInitated()

errcodePreprocFail = cfg.getErrcodePreprocFail()
errcodePreprocDocNotAcc = cfg.getErrcodePreprocDocNotAcc()

errmsgPreprocTiffConv = cfg.getErrmsgPreprocTiffConv()
errmsgPreprocFileNotDwn = cfg.getErrmsgPreprocFileNotDownload()
errmsgPreprocNotValInv = cfg.getErrmsgPreprocNotValidInv()
errmsgPreprocFail = cfg.getErrmsgPreprocFail()

stgPreproc = cfg.getStagePreproc()
stgExtraction = cfg.getStageExtract()

storageType = cfg.getStorageType()
importFolder = cfg.getFolderImport()
exportBlobFolder = cfg.getBlobStoreExport()
importBlobFolder = cfg.getBlobStoreImport()
clientBlobFolder = cfg.getBlobClientFolder()

errmsgMultiInv = cfg.getErrmsgMultiInv()

#Get mimetype and extensions
extnTiff = cfg.getExtnTiff()
extnTif = cfg.getExtnTif()

#Store all the image descriptors in memory for the known invoices

#Constant values that needs to be moved to config file or kept here
IMAGEEXTNS = ["tiff","tif"]
VGEXTNS = ["pdf"]
UI_URL = cfg.getUIServer()

appPort = cfg.getPreprocPort()


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

@timing
def convertToTiffNew(filePath):

    #Convert PDFs and other images to TIFFs. Preprocessor will work on tiff images only
    try:
        extn = os.path.splitext(os.path.split(filePath)[1])[1]
        fileNameWoExtn = os.path.splitext(os.path.split(filePath)[1])[0]
        tiffFilePath = os.path.join(tiffFolder,
                                    fileNameWoExtn + extnTiff)
        print("Tiff File Paths:",filePath,tiffFilePath)
        if "pdf" in extn:
            converted = putil.convertPDFToTiff(filePath,tiffFilePath)
            if converted:
                print("pdf conversion successful")
            else:
                return None
        elif "tif" in extn:
            #Don't convert the tiffs. Only png conversion
            fileCompressed = putil.makeTiffCompressed(filePath)
            print("File Compress Sattus : ",fileCompressed)
            shutil.copy(filePath,tiffFilePath)
        else:
            #Use image conversion techniques to convert to tiff an png
            matchedExtns = [ext for ext in IMAGEEXTNS if ext in extn]
            if len(matchedExtns) > 0:
                img = cv2.imread(filePath)
                cv2.imwrite(tiffFilePath,img)
        return tiffFilePath
    except:
        print("convertToTiffNew",
              traceback.print_exc())
        return None

@timing
def splitPages(splitFolder, fileInfo):
    #Split a document into multiple TIFFs and PNGs
    #and upload the files to blob store and return URIs
    try:
        allPageInfo = {}
        pages = []
        pageInfo = {}
        fileNameWoExtn = fileInfo["filenameWoExtn"]
        fileNameExtn = fileInfo["filenameExtn"]
        filePath = fileInfo["fullPath"]
        print("split function input:",filePath)
        if "pdf" in fileInfo["extn"].lower():
            try:
                inputpdf = PdfFileReader(open(filePath, "rb"),
                                         strict = False)
                noOfPages = inputpdf.numPages
                print("successfully read pdf",noOfPages)
            except:
                print("exception reading pdf")
                inputpdf = PdfReader(filePath)
                noOfPages = len(inputpdf.pages)

            allPageInfo["noOfPages"] = noOfPages
            if noOfPages == 1:
                pageInfo = {}
                framePath = os.path.join(splitFolder,
                                         fileNameWoExtn + "-%s.pdf" % 0)
                copied = putil.copyFile(filePath,framePath)
                if not copied:
                    return None
                pageInfo["sequence"] = "0"
                pageFileInfo = putil.getFileInfo(framePath)
                pageInfo["fileInfo"] = pageFileInfo
                pages.append(pageInfo)
                allPageInfo["pages"] = pages
                print("Split page info:", allPageInfo)
                return allPageInfo
            else:
                for i in range(noOfPages):
                    pageInfo = {}
                    try:
                        output = PdfFileWriter()
                        output.addPage(inputpdf.getPage(i))
                        framePath = os.path.join(splitFolder,
                                                 fileNameWoExtn + "-%s.pdf" % i)
                        print("Frame Path", framePath)
                        #We may have to upload the file here or the calling function would upload
                        with open(framePath, "wb") as outputStream:
                            output.write(outputStream)
                            # output.write(framePath)
                        pageInfo["sequence"] = str(i)
                        pageFileInfo = putil.getFileInfo(framePath)
                        pageInfo["fileInfo"] = pageFileInfo
                        pages.append(pageInfo)
                    except:
                        print("PdfFileWriting Failed",traceback.print_exc())
                        #Jul 12 2022 - Read using PdfReader instead of file reader
                        inputpdf = PdfReader(filePath)
                        #Jul 12 2022 - Read using PdfReader instead of file reader
                        output = PdfWriter()
                        output.addPage(inputpdf.pages[i])
                        framePath = os.path.join(splitFolder, fileNameWoExtn + "-%s.pdf" % i)
                        with open(framePath, "wb") as outputStream:
                            output.write(outputStream)
                        # output.write(framePath)
                        pageInfo["sequence"] = str(i)
                        pageFileInfo = putil.getFileInfo(framePath)
                        pageInfo["fileInfo"] = pageFileInfo
                        pages.append(pageInfo)
                allPageInfo["pages"] = pages
                print("Split page info:", allPageInfo)
                return allPageInfo
        elif "tif" in fileInfo["extn"].lower():
            fileCompressed = putil.makeTiffCompressed(filePath)
            print("File Compress Status : ",fileCompressed)
            read, imgArr = cv2.imreadmulti(filePath)
            noOfPages = len(imgArr)
            allPageInfo["noOfPages"] = noOfPages
            for frame in range(0,noOfPages):
                pageInfo = {}
                framePath = os.path.join(splitFolder, fileNameWoExtn + "-%s.tiff" % frame)

                cv2.imwrite(framePath,imgArr[frame])
                pageInfo["sequence"] = str(frame)
                pageFileInfo = putil.getFileInfo(framePath)
                pageInfo["fileInfo"] = pageFileInfo
                pages.append(pageInfo)
            allPageInfo["pages"] = pages
            return allPageInfo
        else:
            allPageInfo["noOfPages"] = 1
            pageInfo = {}
            pageInfo["sequence"] = "0"
            destPath = os.path.join(splitFolder, fileNameExtn)
            shutil.copy(filePath,destPath)
            pageFileInfo = putil.getFileInfo(destPath)
            pageInfo["fileInfo"] = pageFileInfo
            pages.append(pageInfo)
            allPageInfo["pages"] = pages
            return allPageInfo
    except:
        print("splitPages",
              traceback.print_exc())
        return None
@timing
def convertPDFIntoTiffs(src_path, dest_path):
    try:
        fileInfo = putil.getFileInfo(src_path)
        print("Print fileInfo :",fileInfo)
        allPageInfo = {}
        pages = []
        fileNameWoExtn = fileInfo["filenameWoExtn"]
        filePath = fileInfo["fullPath"]
        print("tiff conversion file path :",filePath)
        if "pdf" in fileInfo["extn"].lower():
            # if dest_path.split(".")[-1].upper() == 'PDF':
            images = putil.pdf_to_img(filePath)
            if images is not None:
                noOfPages = len(images)
                allPageInfo["noOfPages"] = noOfPages
                for idx, image in enumerate(images):
                    dest_full_path = (os.path.join(dest_path,fileNameWoExtn + "_" + str(idx)+'.tiff'))
                    image.save(dest_full_path)
                    fileCompressed = putil.makeTiffCompressed(dest_full_path)
                    print("File Compress Status : ",fileCompressed, dest_full_path)
                    pageInfo = {}
                    pageFileInfo = putil.getFileInfo(dest_full_path)
                    pageInfo["sequence"] = str(idx)
                    pageInfo["fileInfo"] = pageFileInfo
                    pages.append(pageInfo)
                allPageInfo["pages"] = pages
                return  allPageInfo
            else:
                return None

        elif "tif" in fileInfo["extn"].lower():
            fileCompressed = putil.makeTiffCompressed(filePath)
            print("File Compress Status : ",fileCompressed)
            read, imgArr = cv2.imreadmulti(filePath)
            noOfPages = len(imgArr)
            allPageInfo["noOfPages"] = noOfPages
            for frame in range(0,noOfPages):
                pageInfo = {}
                framePath = os.path.join(splitFolder, fileNameWoExtn + "_%s.tiff" % frame)

                cv2.imwrite(framePath,imgArr[frame])
                pageInfo["sequence"] = str(frame)
                pageFileInfo = putil.getFileInfo(framePath)
                pageInfo["fileInfo"] = pageFileInfo
                pages.append(pageInfo)
            allPageInfo["pages"] = pages
            return allPageInfo
        else:
            matchedExtns = [ext for ext in IMAGEEXTNS if ext in fileInfo["extn"]]
            if len(matchedExtns) > 0:
                img = cv2.imread(filePath)
                dest_full_path = os.path.join(dest_path+fileNameWoExtn+'.tiff')
                cv2.imwrite(dest_full_path,img)

                allPageInfo["noOfPages"] = 1
                pageInfo["sequence"] = "0"
                # destPath = os.path.join(splitFolder, fileNameExtn)
                # shutil.copy(filePath,dest_path)
                pageFileInfo = putil.getFileInfo(dest_full_path)
                pageInfo["fileInfo"] = pageFileInfo
                pages.append(pageInfo)
                allPageInfo["pages"] = pages
                return allPageInfo
            print("other than tiff and pdf file")
            return None
    except:
        # print("Error in Converting PDF to TIFF: ",e)
        print("convertPDFToTiff",traceback.print_exc())
        return None

def extract_images_from_pdf(file_path, file_name, out_path, doc_id):
    """
    Extracts all the images from pdf into png files to be stored in a folder
    """
    image_path = os.path.join(file_path, file_name)

    dict_out = {}
    pdf_file = fitz.open(image_path)

    for page_info in enumerate(pdf_file):
        page_num = page_info[0]
        page = page_info[1]
        image_list = page.get_images()
        for image_index, img in enumerate(image_list):
            x_ref = img[0]
            base_image = pdf_file.extract_image(x_ref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            try:
                image = Image.open(io.BytesIO(image_bytes))
                out_file_name = str(doc_id) + "_" + str(page_num) + "_" + str(image_index) + ".png"
                out_image_path = os.path.join(out_path, out_file_name)
                print("Saving Image at path:", out_image_path)
                image.save(out_image_path)
            except Exception as e:
                print("exception in extract_images_from_pdf:",e)
                pass

def returnFailure(docApiInfo):
    docParams = docApiInfo["params"]
    docResult = docApiInfo["result"]
    docParams["resmsgid"] = str(uuid.uuid1())
    docParams["status"] = paramStatusFailed
    docParams["err"] = errcodePreprocFail
    docResult["status"] = paramStatusFailed
    docResult["stage"] = stgPreproc
    response = json.dumps(docApiInfo)
    return response

def apiInit():
    docApiInfo = {}
    docApiInfo["id"] = ""
    docApiInfo["ver"] = tappVer
    docApiInfo["ts"] = str(math.trunc(time.time()))
    docResult = {}
    docParams = {}
    docApiInfo["result"] = docResult
    docApiInfo["params"] = docParams
    docResult["documentType"] = docType
    docResult["lastUpdatedBy"] = sysUser
    docResult["lastProcessedOn"] = math.trunc(time.time())
    docResult["stage"] = stgPreproc
    docParams["status"] = paramStatusFailed
    return docApiInfo


#Jul 19 2022 - Make this a flask app

#Check if input file extn is PDF or TIFF/TIF. Only then process the document
#Split Pages, Convert Images, Compare and return template ID
#Copy the file to two folders 1. for pre-processing to kick off
# @app.route('/preprocess/submit', methods=['POST'])
# def preProcess(request):

#     docApiInfo = apiInit()
#     docParams = docApiInfo.get("params")
#     docResult = docApiInfo.get("result")
#     apiId = ""
#     apiVer = ""
#     callbackUrl = ""

#     try:
#         rawContent = request.content.read()
#         encodedContent = rawContent.decode("utf-8")
#         content = json.loads(encodedContent)
#         print("Input from the UI:\n ", content)

#         #Unwrap the content
#         apiId = content["id"]
#         apiVer = content["ver"]
#         reqContent = content["request"]
#         documentId = reqContent["documentId"]

#         #Add logging
#         # old_stdout = sys.stdout
#         # old_stderr = sys.stderr
#         # log_file_path = os.path.join(rootFolderPath,
#         #                              documentId + "_preprocsync.log")
#         # log_file = open(log_file_path,"w")
#         # sys.stdout = log_file
#         # sys.stderr = log_file
#         #Add logging
        
#         documentLocation = reqContent["location"].lstrip("/")
#         print("Current Working Directory 1:",
#               os.getcwd())
#         uiRoot = cfg.getUIRootFolder()
#         print("Current Directory",os.getcwd(),
#               "documentLocation:",documentLocation,
#               "uiRoot:",uiRoot,
#               "final path:",os.path.join(uiRoot,documentLocation))
#         documentLocation = os.path.join(uiRoot, documentLocation)
#         mimeType = reqContent["mimeType"]

#         extn = "." + mimeType.split("/")[1]

#         documentLocation_renamed = documentLocation.replace("'","")
#         print("document renamed before docId replace:",
#               documentLocation_renamed)
#         docFoldPath = os.path.split(documentLocation_renamed)[0]
#         documentLocation_renamed = os.path.join(docFoldPath,
#                                                 documentId + extn)
#         print("Doc Locations new and old: ", documentLocation,
#                       documentLocation_renamed,
#                       documentLocation == documentLocation_renamed)
#         os.rename(documentLocation, documentLocation_renamed)
#         documentLocation = documentLocation_renamed
#         print("Current Working Directory 2:",
#               os.getcwd(),"document location:", documentLocation)
#         print("Location from TAPP API", documentLocation)

#         #Response parameters returned as per what came in the request
#         docApiInfo["id"] = apiId
#         docApiInfo["ver"] = apiVer
#         docParams["msgid"] = content["params"]["msgid"]
#         docParams["resmsgid"] = str(uuid.uuid1())

#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI
#         # callbackUrl = reqContent["callbackUrl"]
#         callbackUrl = UI_URL
#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI

#         docResult["mimeType"] = mimeType
#         docResult["documentId"] = documentId
#         docResult["name"] = documentId

#         #Trigger Preprocessing
#         #Download file to a folder (pre-defined in some config)
#         #File name should be the same as documentID
#         destFolder = downloadFolder
#         inputPath = os.path.join(destFolder,
#                                  str(documentId) + extn)
#         #Added this line for taking files from local folder itself
#         inputPath = documentLocation
#         orgFileLocation = ""
#         downloaded = False

#         downloaded = True
#         noOfPages = 0
#         filesToBeRemoved = []

#         if not downloaded:
#             docParams["errmsg"] = errmsgPreprocFileNotDwn
#             docResult["statusMsg"] = errmsgPreprocFileNotDwn
#             docApiInfo["responseCode"] = 500
#             return returnFailure(docApiInfo)

#         fileExtnAccepted = False
#         if (extn.lstrip(".").lower() in IMAGEEXTNS) or (extn.lstrip(".").lower() in VGEXTNS):
#             fileExtnAccepted = True
#         #Jul 05 2022 - Check for magic bytes to check if file type is correct
#         try:
#             import magic as magic
#             ALLWD_MIMES = ['image/tiff',
#                            'image/tif',
#                            'text/plain',
#                            'application/pdf']
#             identified_mime = magic.from_file(inputPath,mime = True)
#             print("Identified Mime", identified_mime)
#             if identified_mime.lower() not in ALLWD_MIMES:
#                 fileExtnAccepted = False
#         except:
#             pass
#         #Jul 05 2022 - Check for magic bytes to check if file type is correct
#         # NOTE: Added 19th October for Image extraction at preprocessor stage from PDF Files(KGS)
#         if fileExtnAccepted:
#             #Jul 12, 2022 - allow tiff file also to be uploaded
#             # if extn.lstrip(".").lower() == "pdf":
#             if (extn.lstrip(".").lower() == "pdf") or ("tif" in extn.lstrip(".").lower()):
#             #Jul 12, 2022 - allow tiff file also to be uploaded
#                 #Jul 16 2022 - convert pdf to tiff and make tiff as if it is the original file
#                 # print("input doc path befor conversion :",inputPath)
#                 if extn.lstrip(".").lower() == "pdf":
#                     try:
#                         uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
#                     except:
#                         pass
#                     # if not uploaded:
#                     #     return returnFailure(docApiInfo)
#                     convFolder = "./convert/"
#                     os.makedirs(convFolder,exist_ok = True)
#                     tiff_conversion, tiff_path = putil.convertOrignal_PDFToTiff(inputPath, convFolder)
#                     if tiff_conversion:
#                         inputPath = tiff_path
#                         print("conversion status :",tiff_conversion,"\t file_path :",inputPath)
#                 #Jul 16 2022 - convert pdf to tiff and make tiff as if it is the original file

#                 uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
#                 if not uploaded:
#                     return returnFailure(docApiInfo)

#         if not fileExtnAccepted:
#             docParams["errmsg"] = errmsgPreprocNotValInv
#             docResult["statusMsg"] = errmsgPreprocNotValInv
#             docApiInfo["responseCode"] = 404
#             return returnFailure(docApiInfo)

#         # ## Check for multi-Invoice
#         # # For Piramal Demo throw execption on MultiInvoice
#         singleInvoice = True
#         if not singleInvoice:
#             docParams["errmsg"] = errmsgMultiInv
#             docResult["statusMsg"] = errmsgMultiInv
#             docApiInfo["responseCode"] = 500
#             return returnFailure(docApiInfo)
#         splitFilesInfo = splitPages(splitFolder,
#                                     putil.getFileInfo(inputPath))

#         noOfPages = splitFilesInfo["noOfPages"]
#         pages = splitFilesInfo["pages"]
#         filesToBeRemoved.append(inputPath)
#         print("split complete")

#         pages_request = []
#         bar_qr_data = {}
#         for page in pages:
#             pageFileInfo = page["fileInfo"]
#             pagePath = pageFileInfo["fullPath"]
#             extn = os.path.splitext(os.path.basename(pagePath))[1]
#             tiffPath = convertToTiffNew(pagePath)
#             print("TIFF PATH : ", tiffPath)
#             if tiffPath is None:
#                 docParams["errmsg"] = errmsgPreprocFileNotDwn
#                 docResult["statusMsg"] = errmsgPreprocFileNotDwn
#                 docApiInfo["responseCode"] = 500
#                 return returnFailure(docApiInfo)
#             else:
#                 # Extract QR code and BAR CODE, if any
#                 try:
#                     bar_qr_data[str(page.get('sequence'))] = putil.extract_bar_qr_code(tiffPath)
#                 except:
#                     bar_qr_data[str(page.get('sequence'))] = []

#             pages_request.append(
#                 {"location":tiffPath,
#                  "pageNumber":page["sequence"]})
#         print("PAGE_REQUEST : ", pages_request)

#         if len(pages_request) != len(splitFilesInfo["pages"]):
#             docParams["errmsg"] = errmsgPreprocTiffConv
#             docResult["statusMsg"] = errmsgPreprocTiffConv
#             docApiInfo["responseCode"] = 500
#             return returnFailure(docApiInfo)

#         pageLocations = []
#         for page in pages_request:
#             filesToBeRemoved.extend(page.get("location"))
#             #upload files to blob storage
#             pageLocations.append(page.get("location"))

#         uploaded,blobPaths_tiff = putil.uploadFilesToBlobStore(pageLocations)
#         if (not uploaded) or (len(blobPaths_tiff) != len(pages_request)):
#             docParams["errmsg"] = errmsgPreprocFileNotDwn
#             docResult["statusMsg"] = errmsgPreprocFileNotDwn
#             docApiInfo["responseCode"] = 500
#             return returnFailure(docApiInfo)

#         for page_index, page in enumerate(pages_request):
#             pages_request[page_index]["location"] = blobPaths_tiff[page_index]

#         #Call the Call Extraction API. Take the auth-token
#         #and handle for completion of extraction to a celery task

#         docInfo = {}
#         docInfo["documentId"] = documentId

#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI
#         docInfo["callbackUrl"] = callbackUrl
#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI
#         if isinstance(orgFileLocation,list):
#             orgFileLocation = orgFileLocation[0]
#         docInfo["orgFileLocation"] = orgFileLocation
#         docInfo["extn"] = extn

#         docInfo["pages"] = pages_request
#         docInfo["client_blob_folder"] = ""

#         extractionInfo = {}
#         extractionInfo["documentId"] = documentId
#         extractionInfo["request"] = docInfo
#         response = putil.extractionSubmit(extractionInfo)
#         if response is None:
#             docParams["errmsg"] = "EXTRACTION NOT INITIATED"
#             docResult["statusMsg"] = "FAILED DURING AUTHENTICATION"
#             docApiInfo["responseCode"] = 500
#             return returnFailure(docApiInfo)

#         auth_token = response["auth_token"]
#         allotted_time = response["allotted_time"]
#         t = datetime.strptime(allotted_time,
#                               "%H:%M:%S")
#         delta = timedelta(hours=t.hour,
#                           minutes=t.minute,
#                           seconds=t.second)
#         # exp_time = datetime.now() + delta
#         s_delta = str(delta)
#         print("Time delta:", s_delta,delta, type(delta),type(s_delta))

#         #Kafka implementation - Send a message to Kafka for the consumer to read
#         sub_id = cfg.getSubscriberId()
#         message = {}
#         message["sub_id"] = sub_id
#         message["auth_token"] = auth_token
#         message["documentId"] = documentId
#         message["s_delta"] = s_delta
#         message["callbackUrl"] = callbackUrl
        
#         #Jun 16 2022 - Use Azure eventgrid instead of kafka
#         if cfg.getQueueType() == "EVENGRID":
#             #Call Azure grid
#             id_ = str(datetime.now())
#             data = []
#             payload = {}
#             payload["topic"] = "/subscriptions/3d34cc1f-baa0-4d2e-80b3-95a1834afe2f/resourceGroups/TAPP/providers/Microsoft.EventGrid/topics/" + cfg.getEventGridName()
#             payload["subject"] = "Microsoft.EventGrid/topics/" + cfg.getEventGridName()
#             payload["eventType"] = "Microsoft.EventGrid"
#             payload["eventTime"] = id_
#             payload["id"] = id_
#             payload["dataVersion"] = "1.0"
#             payload["metadataVersion"] = "1.0"
#             payload["data"] = message
#             data.append(payload)
#             headers = {"aeg-sas-key":cfg.getEventGridKey()}
#             data = json.dumps(data)
#             url = cfg.getEventGridAPI()
#             response = requests.post(url = url,
#                                      headers = headers,
#                                      data = data)
#             if response.status_code != 200:
#                 raise Exception()
#         elif cfg.getQueueType() == "KAFKA":
#             producer.sendMessage(message)
#         #Jun 16 2022 - Use Azure eventgrid instead of kafka

#         #Kafka Implementation - End


#         #Call tapp_client's task as a celery delayed task
#         # print("Expiration Time:", type(exp_time))
#         # pp.getExtractionResults.delay(auth_token,
#         #                               s_delta,
#         #                               documentId,
#         #                               callbackUrl)
#         #Call tapp_client's task as a celery delayed task

#         # pp.getExtractionResults.delay(auth_token,
#         #                               exp_time,
#         #                               documentId,
#         #                               callbackUrl)

#         #calling watcher
#         # json_data = json.dumps({"auth_token":auth_token,
#         #                         "document_id":documentId,
#         #                         "exp_time":str(exp_time),
#         #                         "callback_url":callbackUrl})
#         # file_path = os.path.join(preprocFolder,
#         #                          documentId + ".json")
#         # with open(file_path,"w") as f:
#         #     f.write(json_data)
#         print("Post processor request given")

#         #Prepare the parameters to be passed to Async method here
#         docInfo = {}
#         docInfo["documentId"] = documentId
#         docInfo["callbackUrl"] = callbackUrl
#         docInfo["pages"] = pages_request
#         docResult["stage"] = stgExtraction
#         docResult["status"] = statusExtInprog
#         docResult["statusMsg"] = statusmsgExtInit
#         docParams["status"] = paramStatusSuccess
#         docResult["pageCount"] = noOfPages
#         docResult['bar_qr_data'] = bar_qr_data
#         docApiInfo["responseCode"] = 200

#         response = json.dumps(docApiInfo,
#                               indent = 4,
#                               sort_keys = False,
#                               default = str)
#         #Copy the files to a folder where pre-processing and ABBYY processsing will be triggered
#         #Get PreProc Folder from a config file
#         return response
#     except:
#         print("PreprocSync", traceback.print_exc())
#         #Response parameters returned as per what came in the request
#         docParams["errmsg"] = errmsgPreprocFail
#         docResult["statusMsg"] = errmsgPreprocFail
#         docApiInfo["responseCode"] = 500
#         return returnFailure(docApiInfo)
#     finally:
#         #Close log file. Push it to cloud and delete from local disk
#         # try:
#         #     if log_file is not None:
#         #         sys.stdout = old_stdout
#         #         sys.stderr = old_stderr
#         #         log_file.close()
#         #         uploaded,orgFileLocation = putil.uploadFilesToBlobStore([log_file_path])
#         #         if not uploaded:
#         #             pass
#         #         else:
#         #             os.remove(log_file_path)
#         # except:
#         #     pass
#         #Remove all the files which are not required
#         try:
#             for fil in filesToBeRemoved:
#                 os.remove(fil)
#         except:
#             pass

@app.route('/preprocess/submit', methods=['POST'])
def preProcess():

    docApiInfo = apiInit()
    docParams = docApiInfo.get("params")
    docResult = docApiInfo.get("result")
    apiId = ""
    apiVer = ""
    callbackUrl = ""

    try:
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)

        content = json.loads(request.data,
                             strict = False)
        print("Input from the UI:\n ", content)

        #Unwrap the content
        apiId = content["id"]
        apiVer = content["ver"]
        reqContent = content["request"]
        documentId = reqContent["documentId"]

        #Add logging
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_file_path = os.path.join(rootFolderPath,
                                      "preprocSync_" + str(documentId) + ".log")
        log_file = open(log_file_path,"w")
        sys.stdout = log_file
        sys.stderr = log_file
        #Add logging

        print("Input: ",content,reqContent)
        mimeType = reqContent["mimeType"]
        extn = "." + mimeType.split("/")[1]

        documentLocation = reqContent["location"].lstrip("/")

        storageType = cfg.getStorageType()
        if storageType == "BLOB":
            print("Blob path",
                  documentLocation)
            fileExtn = os.path.splitext(documentLocation)[1]
            localFolder = os.path.join(cfg.getRootFolderPath(),
                                     "downloads")
            os.makedirs(localFolder,
                        exist_ok=True)
            localFilePath = os.path.join(localFolder,
                                         documentId + fileExtn)
            print("Local paths",
                  localFolder,
                  localFilePath)
            downloaded = putil.downloadFromBlobStore(documentLocation,
                                                     localFilePath)
            if downloaded:
                documentLocation = localFilePath
        else:
            print("Current Working Directory 1:",
                  os.getcwd())
            uiRoot = cfg.getUIRootFolder()
            print("Current Directory",os.getcwd(),
                  "documentLocation:",documentLocation,
                  "uiRoot:",uiRoot,
                  "final path:",os.path.join(uiRoot,documentLocation))
            documentLocation = os.path.join(uiRoot, documentLocation)
    
            documentLocation_renamed = documentLocation.replace("'","")
            print("document renamed before docId replace:",
                  documentLocation_renamed)
            docFoldPath = os.path.split(documentLocation_renamed)[0]
            documentLocation_renamed = os.path.join(docFoldPath,
                                                    documentId + extn)
            print("Doc Locations new and old: ", documentLocation,
                          documentLocation_renamed,
                          documentLocation == documentLocation_renamed)
            os.rename(documentLocation, documentLocation_renamed)
            documentLocation = documentLocation_renamed
            print("Current Working Directory 2:",
                  os.getcwd(),"document location:", documentLocation)

        print("New Location", documentLocation)

        #Response parameters returned as per what came in the request
        docApiInfo["id"] = apiId
        docApiInfo["ver"] = apiVer
        docParams["msgid"] = content["params"]["msgid"]
        docParams["resmsgid"] = str(uuid.uuid1())

        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        # callbackUrl = reqContent["callbackUrl"]
        callbackUrl = UI_URL
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI

        docResult["mimeType"] = mimeType
        docResult["documentId"] = documentId
        docResult["name"] = documentId

        #Trigger Preprocessing
        #Download file to a folder (pre-defined in some config)
        #File name should be the same as documentID
        destFolder = downloadFolder
        inputPath = os.path.join(destFolder,
                                 str(documentId) + extn)
        #Added this line for taking files from local folder itself
        inputPath = documentLocation
        orgFileLocation = ""
        downloaded = False

        downloaded = True
        noOfPages = 0
        filesToBeRemoved = []

        if not downloaded:
            docParams["errmsg"] = errmsgPreprocFileNotDwn
            docResult["statusMsg"] = errmsgPreprocFileNotDwn
            docApiInfo["responseCode"] = 500
            return returnFailure(docApiInfo)

        fileExtnAccepted = False
        if (extn.lstrip(".").lower() in IMAGEEXTNS) or (extn.lstrip(".").lower() in VGEXTNS):
            fileExtnAccepted = True
        #Jul 05 2022 - Check for magic bytes to check if file type is correct
        try:
            import magic as magic
            ALLWD_MIMES = ['image/tiff',
                           'image/tif',
                           'text/plain',
                           'application/pdf']
            identified_mime = magic.from_file(inputPath,mime = True)
            print("Identified Mime", identified_mime)
            if identified_mime.lower() not in ALLWD_MIMES:
                fileExtnAccepted = False
        except:
            pass
        #Jul 05 2022 - Check for magic bytes to check if file type is correct
        # NOTE: Added 19th October for Image extraction at preprocessor stage from PDF Files(KGS)
        if fileExtnAccepted:
            #Jul 12, 2022 - allow tiff file also to be uploaded
            # if extn.lstrip(".").lower() == "pdf":
            if (extn.lstrip(".").lower() == "pdf") or ("tif" in extn.lstrip(".").lower()):
            #Jul 12, 2022 - allow tiff file also to be uploaded
                #Jul 16 2022 - convert pdf to tiff and make tiff as if it is the original file
                # print("input doc path befor conversion :",inputPath)
                # if extn.lstrip(".").lower() == "pdf":
                #     try:
                #         uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
                #     except:
                #         pass
                #     convFolder = "./convert/"
                #     os.makedirs(convFolder,exist_ok = True)
                #     tiff_conversion, tiff_path = putil.convertOrignal_PDFToTiff(inputPath,
                #                                                                 convFolder)
                #     # tiff_conversion, tiff_path = putil.convertOrignal_PDFToTiff_GS(inputPath,
                #     #                                                                convFolder)
                #     if tiff_path:
                #         inputPath = tiff_path
                #         print("conversion status :",tiff_conversion,
                #               "\t file_path :",inputPath)
                #     else:
                #         print("Failed in convert pdf to tiff")
                #         return returnFailure(docApiInfo)
                #Jul 16 2022 - convert pdf to tiff and make tiff as if it is the original file

                uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
                if not uploaded:
                    return returnFailure(docApiInfo)

        if not fileExtnAccepted:
            docParams["errmsg"] = errmsgPreprocNotValInv
            docResult["statusMsg"] = errmsgPreprocNotValInv
            docApiInfo["responseCode"] = 404
            return returnFailure(docApiInfo)

        # ## Check for multi-Invoice
        # # For Piramal Demo throw execption on MultiInvoice
        singleInvoice = True
        if not singleInvoice:
            docParams["errmsg"] = errmsgMultiInv
            docResult["statusMsg"] = errmsgMultiInv
            docApiInfo["responseCode"] = 500
            return returnFailure(docApiInfo)
        # splitFilesInfo = splitPages(splitFolder,putil.getFileInfo(inputPath))
        # -> alternative to split pages 
        splitFilesInfo = convertPDFIntoTiffs(inputPath, splitFolder) 
        print("splitFilesInfo :",splitFilesInfo)

        #Jul 20 2022 - convert pdf to tiff and make tiff as if it is the original file
        # print("input doc path befor conversion :",inputPath)
        print("is split done correctly",
              splitFilesInfo is None,
              splitFilesInfo)
        if not splitFilesInfo:
            if extn.lstrip(".").lower() == "pdf":
                try:
                    uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
                    if not uploaded:
                        return returnFailure(docApiInfo)
                except:
                    pass
                convFolder = "./convert/"
                os.makedirs(convFolder,
                            exist_ok = True)
                tiff_conversion, tiff_path = putil.convertOrignal_PDFToTiff(inputPath,
                                                                            convFolder)
                if tiff_path:
                    inputPath = tiff_path
                    print("conversion status :",tiff_conversion,
                          "\t file_path :",inputPath)
                    print("Calling split again :")
                    uploaded,orgFileLocation = putil.uploadFilesToBlobStore([inputPath])
                    if not uploaded:
                        return returnFailure(docApiInfo)
                    splitFilesInfo = splitPages(splitFolder,
                                                putil.getFileInfo(inputPath))
                    print("Split info",
                          splitFilesInfo)
                else:
                    print("Failed in convert pdf to tiff")
                    return returnFailure(docApiInfo)
                
        #Jul 20 2022 - convert pdf to tiff and make tiff as if it is the original file
        if not splitFilesInfo:
            return returnFailure(docApiInfo)

        noOfPages = splitFilesInfo["noOfPages"]
        pages = splitFilesInfo["pages"]
        filesToBeRemoved.append(inputPath)
        print("split complete")

        pages_request = []
        bar_qr_data = {}
        for page in pages:
            pageFileInfo = page.get("fileInfo")
            pagePath = pageFileInfo.get("fullPath")
            # extn = os.path.splitext(os.path.basename(pagePath))[1]
            extn = pageFileInfo.get("extn")
            # tiffPath = convertToTiffNew(pagePath)
            tiffPath = pagePath
            print("TIFF PATH : ", tiffPath)
            if tiffPath is None:
                docParams["errmsg"] = errmsgPreprocFileNotDwn
                docResult["statusMsg"] = errmsgPreprocFileNotDwn
                docApiInfo["responseCode"] = 500
                return returnFailure(docApiInfo)
            else:
                # Extract QR code and BAR CODE, if any
                try:
                    bar_qr_data[str(page.get('sequence'))] = putil.extract_bar_qr_code(tiffPath)
                except:
                    print("barcode, qr code extraction exception :",traceback.print_exc())
                    bar_qr_data[str(page.get('sequence'))] = []

            pages_request.append(
                {"location":tiffPath,
                 "pageNumber":page["sequence"]})
        print("PAGE_REQUEST : ", pages_request)

        if len(pages_request) != len(splitFilesInfo["pages"]):
            docParams["errmsg"] = errmsgPreprocTiffConv
            docResult["statusMsg"] = errmsgPreprocTiffConv
            docApiInfo["responseCode"] = 500
            return returnFailure(docApiInfo)

        pageLocations = []
        for page in pages_request:
            filesToBeRemoved.extend(page.get("location"))
            #upload files to blob storage
            pageLocations.append(page.get("location"))

        uploaded,blobPaths_tiff = putil.uploadFilesToBlobStore(pageLocations)
        if (not uploaded) or (len(blobPaths_tiff) != len(pages_request)):
            docParams["errmsg"] = errmsgPreprocFileNotDwn
            docResult["statusMsg"] = errmsgPreprocFileNotDwn
            docApiInfo["responseCode"] = 500
            return returnFailure(docApiInfo)

        for page_index, page in enumerate(pages_request):
            pages_request[page_index]["location"] = blobPaths_tiff[page_index]

        #Call the Call Extraction API. Take the auth-token
        #and handle for completion of extraction to a celery task

        docInfo = {}
        docInfo["documentId"] = documentId

        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        docInfo["callbackUrl"] = callbackUrl
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        if isinstance(orgFileLocation,list):
            orgFileLocation = orgFileLocation[0]
        docInfo["orgFileLocation"] = orgFileLocation
        docInfo["extn"] = extn

        docInfo["pages"] = pages_request
        docInfo["client_blob_folder"] = ""

        extractionInfo = {}
        extractionInfo["documentId"] = documentId
        extractionInfo["request"] = docInfo
        response = putil.extractionSubmit(extractionInfo)
        if response is None:
            docParams["errmsg"] = "EXTRACTION NOT INITIATED"
            docResult["statusMsg"] = "FAILED DURING AUTHENTICATION"
            docApiInfo["responseCode"] = 500
            return returnFailure(docApiInfo)

        auth_token = response["auth_token"]
        allotted_time = response["allotted_time"]
        t = datetime.strptime(allotted_time,
                              "%H:%M:%S")
        delta = timedelta(hours=t.hour,
                          minutes=t.minute,
                          seconds=t.second)
        # exp_time = datetime.now() + delta
        s_delta = str(delta)
        print("Time delta:", s_delta,delta, type(delta),type(s_delta))

        #Kafka implementation - Send a message to Kafka for the consumer to read
        sub_id = cfg.getSubscriberId()
        message = {}
        message["sub_id"] = sub_id
        message["auth_token"] = auth_token
        message["documentId"] = documentId
        message["s_delta"] = s_delta
        message["callbackUrl"] = callbackUrl
        
        #Jun 16 2022 - Use Azure eventgrid instead of kafka
        if cfg.getQueueType() == "EVENGRID":
            #Call Azure grid
            id_ = str(datetime.now())
            data = []
            payload = {}
            payload["topic"] = "/subscriptions/3d34cc1f-baa0-4d2e-80b3-95a1834afe2f/resourceGroups/TAPP/providers/Microsoft.EventGrid/topics/" + cfg.getEventGridName()
            payload["subject"] = "Microsoft.EventGrid/topics/" + cfg.getEventGridName()
            payload["eventType"] = "Microsoft.EventGrid"
            payload["eventTime"] = id_
            payload["id"] = id_
            payload["dataVersion"] = "1.0"
            payload["metadataVersion"] = "1.0"
            payload["data"] = message
            data.append(payload)
            headers = {"aeg-sas-key":cfg.getEventGridKey()}
            data = json.dumps(data)
            url = cfg.getEventGridAPI()
            response = requests.post(url = url,
                                     headers = headers,
                                     data = data)
            if response.status_code != 200:
                raise Exception()
        elif cfg.getQueueType() == "KAFKA":
            producer.sendMessage(message)
        #Jun 16 2022 - Use Azure eventgrid instead of kafka

        #Kafka Implementation - End


        #Call tapp_client's task as a celery delayed task
        # print("Expiration Time:", type(exp_time))
        # pp.getExtractionResults.delay(auth_token,
        #                               s_delta,
        #                               documentId,
        #                               callbackUrl)
        #Call tapp_client's task as a celery delayed task

        # pp.getExtractionResults.delay(auth_token,
        #                               exp_time,
        #                               documentId,
        #                               callbackUrl)

        #calling watcher
        # json_data = json.dumps({"auth_token":auth_token,
        #                         "document_id":documentId,
        #                         "exp_time":str(exp_time),
        #                         "callback_url":callbackUrl})
        # file_path = os.path.join(preprocFolder,
        #                          documentId + ".json")
        # with open(file_path,"w") as f:
        #     f.write(json_data)
        print("Post processor request given")

        #Prepare the parameters to be passed to Async method here
        docInfo = {}
        docInfo["documentId"] = documentId
        docInfo["callbackUrl"] = callbackUrl
        docInfo["pages"] = pages_request
        docResult["stage"] = stgExtraction
        docResult["status"] = statusExtInprog
        docResult["statusMsg"] = statusmsgExtInit
        docParams["status"] = paramStatusSuccess
        docResult["pageCount"] = noOfPages
        docResult['bar_qr_data'] = bar_qr_data
        docApiInfo["responseCode"] = 200

        response = json.dumps(docApiInfo,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        #Copy the files to a folder where pre-processing and ABBYY processsing will be triggered
        #Get PreProc Folder from a config file
        return response
    except:
        print("PreprocSync", traceback.print_exc())
        #Response parameters returned as per what came in the request
        docParams["errmsg"] = errmsgPreprocFail
        docResult["statusMsg"] = errmsgPreprocFail
        docApiInfo["responseCode"] = 500
        return returnFailure(docApiInfo)
    finally:
        #Close log file. Push it to cloud and delete from local disk
        try:
            if log_file is not None:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_file.close()
                # uploaded,orgFileLocation = putil.uploadFilesToBlobStore([log_file_path])
                # if not uploaded:
                #     pass
                # else:
                #     os.remove(log_file_path)
        except:
            pass
        #Remove all the files which are not required
        # try:
        #     for fil in filesToBeRemoved:
        #         os.remove(fil)
        # except:
        #     pass

#Jul 19 2022 - Make this a flask app


@app.route('/download/*', methods = ['OPTIONS'])
def downloadAccess(request):
    request.setHeader('Access-Control-Allow-Methods', 'OPTIONS')
    request.setHeader('Access-Control-Allow-Origin', '*')
    request.setHeader('Access-Control-Allow-Methods', 'POST')
    request.setHeader('Access-Control-Request-Methods', 'POST')
    request.setHeader('Content-Type', 'application/json')
    request.setHeader('Content-Type',
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    request.setHeader('Access-Control-Allow-Headers',
                      'Content-type,Authorization,x-requested-with')
    resp = {}
    response = json.dumps(resp,
                          indent = 4,
                          sort_keys = False,
                          default = str)
    return response

#Jul 19 2022 - Make this a flask app

# @app.route('/download/results', methods = ['POST','OPTIONS'])
# def downloadResults(request):

#     request.setHeader('Access-Control-Allow-Methods', 'OPTIONS')
#     request.setHeader('Access-Control-Allow-Origin', '*')
#     request.setHeader('Access-Control-Allow-Methods', 'POST')
#     request.setHeader('Access-Control-Request-Methods', 'POST')
#     request.setHeader('Content-Type', 'application/json')
#     request.setHeader('Content-Type',
#                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#     request.setHeader('Access-Control-Allow-Headers',
#                       'Content-type,Authorization,x-requested-with')

#     docApiInfo = {}
#     docApiInfo["id"] = ""
#     docApiInfo["ver"] = tappVer
#     docApiInfo["ts"] = str(math.trunc(time.time()))
#     docResult = {}
#     docParams = {}
#     docApiInfo["result"] = docResult
#     docApiInfo["params"] = docParams
#     docApiInfo["responseCode"] = 404
#     callbackUrl = ""

#     try:
#         rawContent = request.content.read()
#         encodedContent = rawContent.decode("utf-8")
#         print("*******************")
#         print(rawContent)
#         print("*******************")
#         print(encodedContent)
#         print("*******************")
#         content = json.loads(encodedContent)

#         #Unwrap the content
#         reqContent = content.get("request","")
#         documentId = reqContent["documentId"]

#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI
#         # callbackUrl = reqContent["callbackUrl"]
#         callbackUrl = UI_URL
#         #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
#         #Use a configured Url to write back to UI

#         #Storage is always blob. This has to persist for a long time
#         #Create separate containers for each client
#         #Change until US picks up from the correct folder
#         file_content = putil.getExtractedResults(documentId,
#                                                  callbackUrl,
#                                                  cleanUpFiles = False)

#         #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path
#         # request.responseHeaders.addRawHeader(b"Content-Type",
#         #                                      b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#         # return file_content
#         local_path = os.path.join(cfg.getUIRootFolder(),
#                                   cfg.getUIDownloadFolder(),
#                                   documentId + ".xlsx")
#         print("Local Path:", local_path)
#         with open(local_path,"wb") as f:
#             f.write(file_content)
#         request.responseHeaders.addRawHeader(b"Content-Type","application/json")
#         return json.dumps({"downloaded":"true"})
#         #Jul 07 2022 - return yes/no and not file content or file path

#     except:
#         print("downloadResults",
#               traceback.print_exc())
#         #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path
#         # request.responseHeaders.addRawHeader(b"Content-Type",
#         #                                      b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#         # return None
#         request.responseHeaders.addRawHeader(b"Content-Type","application/json")
#         return json.dumps({"downloaded":"false"})
#         #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path

@app.route('/download/results', methods = ['POST','OPTIONS'])
def downloadResults():

    request.setHeader('Access-Control-Allow-Methods', 'OPTIONS')
    request.setHeader('Access-Control-Allow-Origin', '*')
    request.setHeader('Access-Control-Allow-Methods', 'POST')
    request.setHeader('Access-Control-Request-Methods', 'POST')
    request.setHeader('Content-Type', 'application/json')
    request.setHeader('Content-Type',
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    request.setHeader('Access-Control-Allow-Headers',
                      'Content-type,Authorization,x-requested-with')

    docApiInfo = {}
    docApiInfo["id"] = ""
    docApiInfo["ver"] = tappVer
    docApiInfo["ts"] = str(math.trunc(time.time()))
    docResult = {}
    docParams = {}
    docApiInfo["result"] = docResult
    docApiInfo["params"] = docParams
    docApiInfo["responseCode"] = 404
    callbackUrl = ""

    try:
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # print("*******************")
        # print(rawContent)
        # print("*******************")
        # print(encodedContent)
        # print("*******************")
        # content = json.loads(encodedContent)

        content = json.loads(request.data,
                             strict = False)

        #Unwrap the content
        reqContent = content.get("request","")
        documentId = reqContent["documentId"]

        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        # callbackUrl = reqContent["callbackUrl"]
        callbackUrl = UI_URL
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI

        #Storage is always blob. This has to persist for a long time
        #Create separate containers for each client
        #Change until US picks up from the correct folder
        file_content = putil.getExtractedResults(documentId,
                                                 callbackUrl,
                                                 cleanUpFiles = False)

        #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path
        # request.responseHeaders.addRawHeader(b"Content-Type",
        #                                      b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # return file_content
        local_path = os.path.join(cfg.getUIRootFolder(),
                                  cfg.getUIDownloadFolder(),
                                  documentId + ".xlsx")
        print("Local Path:", local_path)
        with open(local_path,"wb") as f:
            f.write(file_content)
        request.responseHeaders.addRawHeader(b"Content-Type","application/json")
        return json.dumps({"downloaded":"true"})
        #Jul 07 2022 - return yes/no and not file content or file path

    except:
        print("downloadResults",
              traceback.print_exc())
        #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path
        # request.responseHeaders.addRawHeader(b"Content-Type",
        #                                      b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # return None
        request.responseHeaders.addRawHeader(b"Content-Type","application/json")
        return json.dumps({"downloaded":"false"})
        #Jul 07 2022 - Save the file in a local folder and return yes/no and not file content or file path

#Jul 19 2022 - Make this a flask app


if __name__ == "__main__":
    if len(argv) > 1:
        appPort = int(argv[1])
        print(appPort)
    #Jun 23, 2022 - run the service only in localhost
    # app.run("0.0.0.0", appPort)
    appPort = 9100
    app.run("0.0.0.0", port = appPort)
    #Jun 23, 2022 - run the service only in localhost
