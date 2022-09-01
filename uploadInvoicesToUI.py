# -*- coding: utf-8 -*-

import time
from klein import run, route, Klein
import requests
import json
import traceback
import preProcUtilities as putil
import TAPPconfig as cfg
import os
import mimetypes
from sys import argv



"""
cfg.getDocUpdApi() #DOCUMENT_UPDATE_API_ID

"""
# Vm root folders
rootFolderPath = cfg.getRootFolderPath()
UIRootFolder = cfg.getUIRootFolder()
docGetUrl = cfg.getDocGetURL()

#Get values for request/response
docUpdApi = cfg.getDocUpdApi()
tappVer = cfg.getTappVersion()
docType = cfg.getDocumentType()
sysUser = cfg.getSystemUser()
SubscriberId = cfg.getSubscriberId()
PreprocServer =cfg.getPreprocServer()

def initiate_payload():
    payload = {
                "ver": "1.0",
                "params": {"msgid": ""},
                "request": {
                    "documentId": '',
                    "fileName": '',
                    "Vendor_Name": '',
                    "documentType": "Invoice",
                    "mimeType": " ",
                    "uploadUrl": "",
                    "size": 64185,
                    "status": "NEW",
                    "submittedBy": "system"
                    }
                }
    return payload

app = Klein()
@app.route('/ui/upload/invoice',methods = ["POST"])
def uploadDocToUI(request):
    try:
        #Open the request json from logic app

        content = json.loads(request.content.read())        
        fileLocation = content["blobname"]
        fileName = content["fileName"]

        #Using fileLocation download the file using putil.downloadFilesFromBlob
        #Define Local path as UI root folder + import
        #fileName = os.path.basename(fileLocation)
        blob_downloads = [SubscriberId +"/"+ fileLocation]
        local_downloads = [UIRootFolder + "/import/"+fileName]
        print("download folders: ", blob_downloads, local_downloads)
        downloadUrls = zip(blob_downloads,local_downloads)
        FileDownloadStatus = putil.downloadFilesFromBlob(downloadUrls)
        print("FileDownloadStatus :",FileDownloadStatus)
        if FileDownloadStatus != True:
            print("Failed to Downloaded file from blob")
            return json.dumps({"status_code":500,"message":"Failed to Downloaded file from blob"})
        
         #Call a function to post to document.add using the metadata received from logic app
        #emailData, fileLocation, consumed, etc., into request along with other required parameters for UI
        
        #Get the response from document.Add and pass it to the caller
        #In case of exception return status code 500 and a status as "Failed"
        payload = initiate_payload()
        payload = {"ver":"1.0"}
        params = {}
        params["msgid"] = "100"

        request = {}
        request["uploadUrl"] = "/import/" + fileName
        request["fileName"] = fileName
        request["mimeType"] = mimetypes.guess_type(fileName)[0]
        #request["documentId"] = content.get("documentId") #str(time.time()).replace('.','_')
        request["documentType"] = "Invoice"
        request["status"] = "NEW"
        request["submittedBy"] = "system"
        #request["size"] = content.get("size")
        request = {**request,**content}

        payload["params"] = params
        payload["request"] = request

        headers = {}
        headers["Content-Type"] = "application/json"
        # headers["access-control-allow-origin"] = "*"
        # headers["content-length"] = "573"
        add_url = "http://52.172.153.247:9999/document/add" # PreprocServer+"/document/add"

        data = json.dumps(payload)
        print("\n headrs:",headers, "\n\npayload:", data, "\n\npost url:",add_url)
        response = requests.post(add_url, headers=headers,  data=data)

        # reading response
        response_json = response.json()
        # getting result; type dictionary
        print("Request response :",response_json)
        result = response_json.get('result')
        if result.get("status"):   
            if (result['status'].lower() == 'failed'):
                return json.dumps({"status_code":500,"message":"Failed in Preprocessing"})
        return json.dumps({"status":200, "message": "success"})

    except:
        print(traceback.print_exc())
        return json.dumps({"status_code":500,"message":"Failed in request"})



if __name__ == "__main__":
    if len(argv) > 1:
        appPort = int(argv[1])
        print(appPort)
    appPort = "8585"
    app.run("0.0.0.0", appPort)