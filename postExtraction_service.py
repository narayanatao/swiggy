# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:52:45 2022

@author: DELL
"""

import traceback
from sys import argv
import json
import TAPPconfig as cfg

from client_rules import bizRuleValidateForUi as BR
import corrections as corr
import path_finder as pfi
import format_identifier as fmi

from klein import Klein
app = Klein()

appPort = cfg.getPostExtractionSvcPort()
svcIp = cfg.getPostExtractionSvcIP()
UI_url = cfg.getUIServer()

@app.route('/document/BizRuleValidate',methods = ['POST'])
def docBizRuleValidate(request):
    
    def exceptionResponse():
        return json.dumps({"status_code":200,
                           "list_fields":[]},
                          indent = 4,
                          sort_keys = False,
                          default = str)

    def successResponse():
        return json.dumps({"status_code":200,
                           "list_fields":[]},
                          indent = 4,
                          sort_keys = False,
                          default = str)

    try:
        request.responseHeaders.addRawHeader(b"Content-Type",
                                             b"application/json")
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        content = json.loads(encodedContent)
        documentId = content["documentId"]
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        # callBackUrl = content["callBackUrl"]
        callBackUrl = UI_url
        #Jul 05 2022 - DO NOT USE Callback Url to update back to UI.
        #Use a configured Url to write back to UI
        print("Input from the UI:\n ", content)
        resp = BR(documentId,
                  callBackUrl)
        if resp is not None:
            if len(resp) == 0:
                return successResponse()
            else:
                resp_ = {}
                resp_["status_code"] = 500
                resp_["list_fields"] = []
                for res in resp:
                    r = {}
                    r["fieldId"] = res[0]
                    r["error_message"] = res[1]
                    resp_["list_fields"].append(r)
                resp_ = json.dumps(resp_,
                                   indent = 4,
                                   sort_keys = False,
                                   default = str)
                return resp_
        else:
            return exceptionResponse()
    except:
        print("docBizRuleValidate",
              traceback.print_exc())
        return exceptionResponse()

@app.route('/corrections/getOCRLines', methods=['POST'])
def get_ocr_lines(request):
    try:
        response = corr.get_ocr_lines(request)
        return response
    except:
        print("get_ocr_lines",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        request.responseHeaders.addRawHeader(b"content-type", b"application/json")
        response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
        return response

@app.route('/corrections/getOCRLinesTemp', methods=['POST'])
def get_ocr_lines_temp(request):
    try:
        response = corr.get_ocr_lines_temp(request)
        return response
    except:
        print("get_ocr_lines_temp",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        request.responseHeaders.addRawHeader(b"content-type", b"application/json")
        response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
        return response

@app.route('/path_finder/get_templates', methods=['POST'])
def get_templates(request):
    try:
        response = pfi.get_templates(request)
        return response
    except:
        print("get_templates",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['templates'] = []
        response_object['message'] = "Failure"
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response

@app.route('/path_finder/test_templates', methods=['POST'])
def test_templates(request):
    try:
        response = pfi.test_templates(request)
        return response
    except:
        print("test_templates",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object["extracted_value"] = []
        response_object['message'] = "Failure"
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response
        
@app.route('/path_finder/validate_template', methods=['POST'])
def validate_template(request):
    try:
        response = pfi.validate_template(request)
        return response
    except:
        print("validate_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['extracted_value'] = {}
        response_object['message'] = "Failure"
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response

@app.route('/path_finder/create_templates', methods=['POST'])
def insert_template(request):
    try:
        response = pfi.insert_template(request)
        return response
    except:
        print("insert_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response

@app.route('/format_identifier/refresh_format', methods=['POST'])
def refresh_format(request):
    try:
        response = fmi.refresh_format(request)
        return response
    except:
        print("refresh_format",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['refreshed_result'] = None
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response

@app.route('/format_identifier/get_suggestion', methods=['POST'])
def get_suggested_masterdata(request):
    try:
        response = fmi.get_suggested_masterdata(request)
        return response
    except:
        print("get_suggested_masterdata",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['master_data'] = None
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response


@app.route('/format_identifier/create', methods=['POST'])
def insert_masterdata_wrapper(request):
    try:
        response = fmi.insert_masterdata_wrapper(request)
        return response
    except:
        print("insert_template",
              traceback.print_exc())
        response_object = {}
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response_object['master_data'] = None
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response


@app.route('/format_identifier/validate', methods=['POST'])
def validate_masterdata_wrapper(request):
    try:
        response = fmi.validate_masterdata_wrapper(request)
        return response
    except:
        print("validate_masterdata_wrapper",
              traceback.print_exc())
        response_object = {}
        response_object['validate_result'] = "INVALID"
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['score'] = 0.0
        response_object['message'] = "Failure"
        request.responseHeaders.addRawHeader(b"content-type",
                                             b"application/json")
        response = json.dumps(response_object,
                              indent = 4,
                              sort_keys = False,
                              default = str)
        return response


if __name__ == "__main__":
    if len(argv) > 1:
        appPort = int(argv[1])
        print(appPort)
    #Jun 23, 2022 - run the service only in localhost
    # app.run("0.0.0.0", appPort)
    app.run(svcIp, appPort)
    #Jun 23, 2022 - run the service only in localhost
