# -*- coding: utf-8 -*-

print("starting Watcher")
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import time
import os
import tapp_client as pp
import TAPPconfig as cfg
import traceback
import json
from datetime import datetime

#Get folder to monitor, URLs to call from config file
preProcFolder = cfg.getPreProcAsyncFolder()
uiServerUrl = cfg.getUIServer()
constCallbackUrl = cfg.getConstCallbackUrl()

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

callbackUrl = ""

class MyHandler(PatternMatchingEventHandler):

    def on_created(self, event):
        self.process(event)

    def process(self, event):

        try:
            print("Watcher triggered")
            json_docInfo = None
            fullPath = event.src_path
            fileNameExtn = os.path.basename(fullPath)
            documentId = os.path.splitext(fileNameExtn)[0]
            callbackUrl = ""
            callBackFile = os.path.join(preProcFolder,
                                        documentId + ".json")
            print("Callback File:", callBackFile)

            if os.path.isfile(callBackFile):

                with open(callBackFile, mode = "r") as f:
                    json_docInfo = f.read()
                    docInfo = json.loads(json_docInfo)
                    auth_token = docInfo.get("auth_token")
                    document_id = docInfo.get("document_id")
                    s_exp_time = docInfo.get("exp_time")
                    callback_url = docInfo.get("callback_url")
                    exp_time = datetime.strptime(s_exp_time,
                                                 "%Y-%m-%d %H:%M:%S.%f")
                    extraction = pp.getExtractionResults(auth_token,
                                                         exp_time,
                                                         document_id,
                                                         callback_url)
                    if extraction:
                        print("Success")
                    else:
                        print("Failure")

            else:
                print("No callback url")

        except:
            print(traceback.print_exc())
            if callbackUrl == "":
                print("Callback URL cannot be retrieved")

if __name__ == '__main__':
    print("Watcher Started", preProcFolder)
    observer = Observer()
    observer.schedule(MyHandler(), preProcFolder)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
