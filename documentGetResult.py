# -*- coding: utf-8 -*-
import json
import traceback
import requests
from klein import run, route, Klein
from sys import argv
from dateutil import parser
import TAPPconfig as cfg

PREPROC_SERVER = cfg.getPreprocServer()
GET_DOCUMENT_RESULT = cfg.getDocoumentResult()
FIND_DOCUMENT = cfg.getDocumentFind()

def init_payload(request):
    content = json.loads(request.content.read()) 
    payload = {}        
    request = {}
    # filter =  {"status": "REVIEW_COMPLETED"}

    request["filter"] = content["filter"]
    request["token"] =  "" #content["token"]
    if content.get("offset"):
        request["offset"] = content.get("offset")
    else:
        request["offset"] = 0
    request["offset"] =  0 #content["offset"]
    if content.get("limit"):
        request["limit"] = content.get("limit")
    else:
        request["limit"] = 250
    if content.get("page"):
        request["page"] = content.get("page")
    else:
        request["page"] = 0
    
    payload["request"] = request
    return payload

def init_response(status_code,status_message,result):
    response = {}
    response["status_code"] = status_code
    response["status_message"] = status_message
    # if (len(doc_MetaData) >0):
    #     response["doc_MetaData"] = doc_MetaData
    # else:
    #     response["doc_MetaData"] = {}
    if (len(result)>0):
        response["result"]= result
    else:
        response["result"]= {
                        "documents": [
                            {
                            "documentId": "",
                            "fileName": "",
                            "extracted_data": {},
                            "totalFields": 0,
                            "correctedFields": 0
                            }
                        ]
                        }
    response = json.dumps(response)
    return response

def findDocResult(result):
    endPiont = PREPROC_SERVER + GET_DOCUMENT_RESULT
    print("Get Document Result URL",endPiont)
    headers ={}
    headers["Content-Type"] = "application/json"
    data = []
    try:
        dict_list = result.get("documents")
        if dict_list:
            for document in dict_list:
                doc_add_fields = {}
                extracted_data = {}
                totalRequiredFields = 0
                # totalEmptyFields = 0
                correctedFields = 0
                doc_add_fields["documentId"] = document.get("documentId")
                doc_add_fields["document_metadata"] = document              
                # doc_add_fields = {**doc_add_fields,**document}
                # print("item",doc_add_fields)
                # getting docId from document dict and retriving document result
                for field, value in document.items():
                    if (field == "documentId"):
                        url = endPiont + value
                        resp = requests.get(url,headers= headers)
                        resp = resp.json()
                        #getting list of header items from doc result 
                        headerItems = resp.get("result").get("document").get("documentInfo")
                        if headerItems:
                            for hdr_item in headerItems:
                                f_key = hdr_item.get("fieldId")
                                f_val = hdr_item.get("fieldValue")
                                extracted_data[f_key] = f_val
                                if hdr_item.get("correctedValue"):
                                    correctedFields = correctedFields + 1
                # # calculating non empty fields
                # for key, val in extracted_data.items():
                #     if (val != ''):
                #         totalRequiredFields = (totalRequiredFields + 1)
                #     else:
                #         if val == '':
                #             totalEmptyFields = totalEmptyFields + 1
                # print("Total Non-empty Required fields :",extracted_data)
                # print("Total Fields :",len(extracted_data),"\tNon-Emty Fields", totalRequiredFields,"\tEmpty Fields",totalEmptyFields)
                # if (len(extracted_data) == (totalEmptyFields+totalRequiredFields)):
                #     print("Validating Empty and None empty matched") 
                totalRequiredFields = len(extracted_data)   
                extracted_data["fileName"] = document.get("fileName")
                inv_Date = extracted_data.get("invoiceDate")
                if inv_Date :
                    extracted_data["invoiceDate"] = parser.parse(extracted_data.get("invoiceDate"), dayfirst=True).date().strftime('%d-%b-%Y')
                doc_add_fields["extracted_data"] = extracted_data
                doc_add_fields["totalFields"]= totalRequiredFields
                doc_add_fields["correctedFields"] = correctedFields
                #print("doc dict keys :",doc_add_fields.keys())
                data.append(doc_add_fields)
            #print("Recived Doc list :",len(dict_list),"\nAdded result doc list :",len(data))
            result["documents"]= data
            response = init_response(status_code= 200,status_message ="success",result= result)
            return response
        else:
            print("Empty list recived. please check the limit")
            response = init_response(status_code= 500,status_message ="Failed",result= {})
            return response
    except:
        print("document get exception :",traceback.print_exc())
        response = init_response(status_code= 500,status_message = traceback.print_exc(),result= {})
        return response

def prepare_request(payload):
    try:
        headers = {}
        headers["Content-Type"] = "application/json"
        add_url = PREPROC_SERVER + FIND_DOCUMENT
        print("add_url :",add_url)
        data = json.dumps(payload)
        # print("\nPosting url:",add_url, "\nHeadrs:",headers, "\nRequest:", data)
        
        response =  requests.post(add_url, headers=headers, data=data) 
        response = response.json()
        # print("Request response5 :",response)
        if str(response.get("params").get("status")).lower() == "success":
            docs_list = response.get('result').get("documents")
            doc_ids = []
            # print("docs list :",docs_list)
            for itms in docs_list:
                # print("doc metadata :",itms)
                # doc_ids.append({"documentId":itms.get("documentId"),"fileName":itms.get("fileName")})
                doc_ids.append(itms)
            #print("No of Doc ids :",doc_ids)
            result = response.get('result')
            result["documents"] = doc_ids
            post_result = findDocResult(result)
            post_result = json.loads(post_result)
            print("Result post :",post_result)
            print("Result post :",type(post_result))        
            if (post_result["status_code"]==200):
                count = result.get("count")
                page = result.get("page")
                perPageRecords = result.get("perPageRecords")
                print("count :",count,"\tpage :",page,"\tperPageRecords :",perPageRecords)
                print("\npage-wise result in requested format",post_result)
                return json.dumps(post_result)
            else:
                print("Failed while posting result")
                return json.dumps(post_result)
        if str(response.get("params").get("status")).lower() == "failed":
            print("Request Failed")
            response = init_response(status_code= 500,status_message = response.get("params").get("status"),result= {})
            return response          
    except:
        print("prepare_request exception")
        response = init_response(status_code= 500,status_message = traceback.print_exc(),result= {})
        return response

app = Klein()
@app.route('/document/getResult',methods = ["POST"])

def getReviewedDocs(request):
    """
    input json
    {
        "request": {
            "token": "",
            "filter": {
                "status": "REVIEW_COMPLETED" or 
                "documentId":""asfdf123"
            },
            "offset": 0,  # optional
            "limit": 250, # optional
            "page": 1  # optional
        }
    }
    """
    try:
        payload = init_payload(request)
        reps = prepare_request(payload)
        reps = json.loads(reps)
        pageCount = 1
        print("reps keys:",reps.keys())
        if (reps.get("result")) :
            count = reps.get("result").get("count")
            page = reps.get("result").get("page")
            perPageRecords = reps.get("result").get("perPageRecords")
            #print("count :",count,"\tpage :",page,"\tperPageRecords :",perPageRecords)
            while (pageCount * int(perPageRecords)) <int(count):
                pageCount = pageCount+1
                payload["request"]["page"] = pageCount
                print("Requesting Page No :",payload["request"]["page"])
                reps = prepare_request(payload)
            print("Total Recods requested")
            print("finally done")
            reps.get("result").pop("count")
            reps.get("result").pop("page")
            reps.get("result").pop("perPageRecords")
            docs = reps.get("result").get("documents")
            reps["documents"] = docs
            reps.pop("result")
            return json.dumps(reps)
        else:
            response = init_response(status_code= 500,status_message = "Failed",result={})
            return json.dumps(response)

    except:
        print("get doc id exception",traceback.print_exc())
        response = init_response(status_code= 500,status_message = traceback.print_exc(),result={})
        return json.dumps(response)



if __name__ == "__main__":
    # import time
    # tic = time.perf_counter()
    # result = getReviewedDocs(request)
    # result = findDocResult(result)
    # toc = time.perf_counter()
    # print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

    if len(argv) > 1:
        appPort = int(argv[1])
        print(appPort)
    appPort = "8588"
    # app.run("127.0.0.1", appPort) local port
    app.run("0.0.0.0", appPort)