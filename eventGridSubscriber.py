# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:34:03 2022

@author: DELL
"""

import TAPPconfig as cfg
import json
import traceback
from klein import Klein
# import os
import tapp_client as cli
# from datetime import datetime
app = Klein()

@app.route("/eventgrid/ping", methods = ['POST'])
def subscriber(request):
    try:
        print("Received request",request)
        rawContent = request.content.read()
        encodedContent = rawContent.decode("utf-8")
        content = json.loads(encodedContent)
        body = content["content"]

        sub_id = body["sub_id"]
        auth_token = body["auth_token"]
        documentId = body["documentId"]
        s_delta = body["s_delta"]
        callbackUrl = body["callbackUrl"]
        extraction = cli.getExtractionResults(auth_token,
                                              s_delta,
                                              documentId,
                                              callbackUrl,
                                              sub_id)
        # #Create a json file in the folder and fire file creation event to trigger poller
        # json_obj = json.dumps(body)
        # folderPath = "./pollstart"
        # os.makedirs(folderPath,exist_ok = True)
        # filePath = os.path.join(folderPath,str(datetime.now()),"_polling.json")
        # with open(filePath,"w") as f:
        #     f.write(json_obj)
        return json.dumps({"status_code":200})
    except:
        print("subscriber",traceback.print_exc())
        return json.dumps({"status_code":500})


if __name__ == "__main__":
    appPort = cfg.getEventGridSubPort()
    app.run("0.0.0.0", appPort)
