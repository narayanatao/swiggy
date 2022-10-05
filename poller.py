# -*- coding: utf-8 -*-
from polling import poll
import preProcUtilities as putil
import TAPPconfig as cfg
import traceback
import requests
import time
import sys
import datetime
import json

def pingForStatus(auth_token,
                  documentId,
                  sub_id):
    try:

        url = cfg.getExtractionGetAPI()
        enc_token = putil.encryptMessage(json.dumps({"auth_token":auth_token,
                                                     "documentId":documentId,
                                                     "sub_id":sub_id}))
        enc_token = enc_token.decode("utf-8")
        message = json.dumps({"message":enc_token})
        header = {"Content-Type":"application/json"}
        print("Get Doc Status ", url, message)
        response = requests.post(url = url, headers = header, data = message)
        print("print reponse",response, type(response))
        if response.status_code != 200:
            return None
        resp_obj = response.json()
        ###Need to have the messages encrypted in the server and decrypted here
        print("Response Object", resp_obj, type(resp_obj))
        resp_obj = putil.decryptMessage(resp_obj["message"])
        resp_obj = json.loads(resp_obj)
        print("Unwrapped object", resp_obj, type(resp_obj))
        if resp_obj["status_code"] != 200:
            print("Failed during extraction ")
            return None
        return resp_obj
        
    except:
        return None

def checkResponse(response):
    if response is None:
        print(" check resp method True 1")
        return True
    ext_status = response['status']
    # if ext_status =='Success':
    #     return True
    if (ext_status == 'Processing' or ext_status == "Submitted"):
        return False
    else:
        return True
    # if ext_status =='Extracted':
    #     print("Extracted triggred ")
    #     return True

def pollStatus(auth_token,delta,documentId,sub_id):

    try: 
        pollResult = poll(lambda : pingForStatus(auth_token,
                                                 documentId,
                                                 sub_id),
                                    check_success = checkResponse,
                                    step =10,
                                    timeout = delta)
        return pollResult
    except:
        print(traceback.print_exc())
        print("Polling TimeOutException ")
        return None
