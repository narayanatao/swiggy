#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:25:58 2022

@author: Parmesh
"""

import requests
import json
import traceback, copy
import preProcUtilities as putil
import mimetypes
from sys import argv
from dateutil import parser
import TAPPconfig as cfg
import os,sys
import pandas as pd
from post_processor import wordshape
from dateutil.parser import parse
import datetime

SubscriberId = cfg.getSubscriberId()
PreprocServer = cfg.getPreprocServer()
UI_SERVER = cfg.getUIServer()
GET_DOCUMENT_RESULT = cfg.getDocoumentResult()
FIND_DOCUMENT = cfg.getDocumentFind()
ROOT_DIR = os.path.abspath(os.curdir)
RESULT_DOWNLOAD_FOLDER = cfg.getUIDownloadFolder()

def get_epoc_timestamp_minus_two_day():
    date = datetime.datetime.now() - datetime.timedelta(days=2)
    return int(datetime.datetime.timestamp(date)*1000)


header  = {"Content-Type":"application/json"}
request= {
            "id": "api.document.find",
            "ver": "1.0",
            "ts": 1571813791276,
            "params": {
                "msgid": ""
            },
            "request": {
                "token": "",
                "filter": {
                    "status": "REVIEW_COMPLETED",
                    "lastUpdatedOn": {">":get_epoc_timestamp_minus_two_day()}
                },
                "offset": 0,
                "limit": 0,
                "page": 1
            }
        }

def convert_epoc_time_into_date_format(epoc_time_stamp):
    try:
        date = datetime.datetime.fromtimestamp(int(epoc_time_stamp)/1000)
        return date.strftime("%d/%m/%y")
    except:
        return None

def get_reviewed_perpage(request):
    doc_ids = []
    pageNo = None
    try:
        # request["filter"]["lastUpdatedOn"] = {">":get_epoc_timestamp_minus_two_day()}
        print("request filter :",request)
        reps = requests.post(url=UI_SERVER + FIND_DOCUMENT,headers = header,data = json.dumps(request))
        # print("reps :",reps)
        if reps.status_code != 200:
            return pageNo,doc_ids
        response = reps.json()
        if str(response.get("params").get("status")).lower() == "success":
            docs_list = response.get('result').get("documents")
            print("docs list :",len(docs_list))
            pageNo = response.get("result").get("page")
            # totalRecordcount = reps.get("result").get("count")
            # perPageRecords = reps.get("result").get("perPageRecords")

            for itms in docs_list:
                doc_ids.append({"documentId":itms.get("documentId"),
                                "lastUpdatedOn":convert_epoc_time_into_date_format(itms.get("lastUpdatedOn"))})
        return pageNo,doc_ids
    except:
        print("get_reviewed_perpage exception",traceback.print_exc())
        return pageNo,doc_ids

def get_all_reviewed_docs():
    doc_ids = []
    try:
        urls =UI_SERVER+FIND_DOCUMENT
        print("urls",urls,"\nrequest :",request,"\nheader :",header)
        reps = requests.post(url = urls,headers = header,data = json.dumps(request))
        # print("reps2 :",reps)
        if reps.status_code != 200:
            return doc_ids
        # pageCount = 1
        reps = reps.json()
        # print("reps keys:",reps)
        if (reps.get("result")) :
            pageCount = reps.get("result").get("page")
            count = reps.get("result").get("count")
            perPageRecords = reps.get("result").get("perPageRecords")
            print("pageCount",pageCount,"count :",count,"\tperPageRecords :",perPageRecords)
            docs_ls = reps.get('result').get("documents")
            # print("docs list :",docs_list)
            for itms in docs_ls:
                doc_ids.append({"documentId":itms.get("documentId"),
                                "lastUpdatedOn":convert_epoc_time_into_date_format(itms.get("lastUpdatedOn"))})
            print("list of document id :",len(doc_ids))
            try:    
                while (int(count)>(pageCount * int(perPageRecords))):
                    pageCount = pageCount + 1
                    request["request"]["page"] = pageCount
                    print("Requesting Page No :",request)
                    pageno, docids = get_reviewed_perpage(request)
                    print("inside loop pageno",pageno,"\tdoc ids ",len(docids))
                    if (pageno is not None) and len(docids)>0:
                        doc_ids.extend(docids)
                    else:
                        print("Getting reviewed doc error")
                        break
            except :
                print("getting all pages docs exception")
                pass
        return doc_ids
    except:
        print("get_all_reviewed_docs exception ",traceback.print_exc())
        return doc_ids

def getDocumentMeta(documentId):

    try:
        get_result = putil.getDocumentApi(documentId,
                                          UI_SERVER)
        if get_result is None:
            print("Issue in calling document/get API",
                  traceback.print_exc())
            return None
        else:
            # obj = json.loads(get_result)
            obj = get_result
            responseCode = obj.get("responseCode")
            if str(responseCode).upper() != "OK":
                print("Internal error in document/get API")
                return None
            else:
                result = obj.get("result")
                document = result.get("document")
                # if document.get("rawPrediction"):
                #     document.pop("rawPrediction")
                return document
    except:
        print("getDocumet Failed",
              traceback.print_exc())
        return None

def get_document_result(documentId):
    try : 
        endPiont = UI_SERVER + GET_DOCUMENT_RESULT + documentId
        # print("Get Document Result URL",endPiont)
        headers = {}
        headers["Content-Type"] = "application/json"
        doc_rs = requests.get(endPiont)
        # print("doc_rs :",doc_rs)
        if doc_rs.status_code != 200:
            return None
        #doc_rs = json.loads(doc_rs.text)
        # print("str",doc_rs.text)
        doc_rs = doc_rs.json()
        # documentInfo = doc_rs.get("result").get("document").get("documentInfo")
        # print("documentInfo :",documentInfo)
        return doc_rs
    except :
        print("get_document_result exception ",traceback.print_exc())
        return None

def dwonload_pred_files(doc_id:str):
    try:
        file = doc_id +"_pred.csv"
        blob_downloads = [os.path.join(SubscriberId,file)]
        # Define local download path and file name RESULT_DOWNLOAD_FOLDER
        local_downloads = [os.path.join(ROOT_DIR,RESULT_DOWNLOAD_FOLDER,file)]
        print("download folders: ", blob_downloads, local_downloads)
        downloadUrls = zip(blob_downloads,local_downloads)
        FileDownloadStatus = putil.downloadFilesFromBlob(downloadUrls)
        if FileDownloadStatus:
            return local_downloads[0]
        return None
    except:
        print("file dwn exp ")
        return None

def get_reference_data(doc_list: list):
    col=[]
    df = pd.DataFrame(col, columns=['vendor_id','vendor_name','field_name','field_shape','doc_id','submitted_on','status'])
    try:
        for idx,i in enumerate(doc_list):
            print("\nRefreshing Master data for :",idx," --> ",i["documentId"])
            s=str(i["documentId"])
            res =  get_document_result(s)
            if not(res):
                continue
            
            m={} # invoice number
            n={} # invoice date
            if len(df[df['vendor_name'] == res["result"]['document']['documentInfo'][3]['fieldValue']])<=20:


                m['vendor_name']=res["result"]['document']['documentInfo'][3]['fieldValue']
                m['vendor_id']=res["result"]['document']['documentInfo'][2]['fieldValue']
                m['submitted_on']=i["lastUpdatedOn"]
                m['field_name']='invoiceNumber'
                m['doc_id']=s
                if 'correctedValue' in (res["result"]['document']['documentInfo'][0]).keys():
                    k=res["result"]['document']['documentInfo'][0]['correctedValue']
                    ## call dwn pre doc 
                    pred_file_path = dwonload_pred_files(str(i["documentId"]))
                    print("pred_file_path :",pred_file_path)
                    if not(pred_file_path):
                        continue
                    df1 = pd.read_csv(pred_file_path)
                    #df1=df1.to_string(index=False)
                    df2=df1[df1['text']==k]
                    if df2.shape[0]>0:
                        m['left']= (df2['left'] * df2['image_widht']).values[0]

                        m['right']=(df2['right'] * df2['image_widht']).values[0]
                        m['top'] =(df2['top'] * df2['image_height']).values[0]
                        m['bottom']=(df2['bottom'] * df2['image_height']).values[0]
                        m["field_shape"]=wordshape(str(res["result"]['document']['documentInfo'][0]['correctedValue']))
                        if df2.shape[0]>1:
                            m['status']=0
                        else:
                            m['status']=1
                        
                        df = df.append(m, ignore_index = True)
                else:
                    m["field_shape"]=wordshape(str(res["result"]['document']['documentInfo'][0]['fieldValue']))
                    m['left']=res["result"]['document']['documentInfo'][0]['boundingBox']['left']
                    m['right']=res["result"]['document']['documentInfo'][0]['boundingBox']['right']
                    m['top']=res["result"]['document']['documentInfo'][0]['boundingBox']['top']
                    m['bottom']=res["result"]['document']['documentInfo'][0]['boundingBox']['bottom']
                    m['status']=1
                    df = df.append(m, ignore_index = True)
                
                # invoice date matching
                n['vendor_name']=res["result"]['document']['documentInfo'][3]['fieldValue']
                n['vendor_id']=res["result"]['document']['documentInfo'][2]['fieldValue']
                n['submitted_on']=i["lastUpdatedOn"]
                n['field_name']='invoiceDate'
                n['doc_id']=s
                if 'correctedValue' in (res["result"]['document']['documentInfo'][1]).keys():
                    l=res["result"]['document']['documentInfo'][1]['correctedValue']
                    pred_file_path = dwonload_pred_files(str(i["documentId"]))
                    # print("pred_file_path :",pred_file_path)
                    if not(pred_file_path):
                        print("pred file not found / download error")
                        continue
                    df1=pd.read_csv(pred_file_path)

                    dt = parse(l)
                    
                    l1=(dt.strftime('%d/%m/%Y'))
                    df2=df1[df1['is_date']==1]
                    I=0
                    for index, row in df2.iterrows():
                        dt=parse(row['text'])
                        row['text']=(dt.strftime('%d/%m/%Y'))
                        #print(row['text'],'kk')
                    #df2=df2[df2['text']==l1]
                    #print(df2)
                        
                        # print(I)
                        if l1==row['text']:
                            I=I+1
                            # print(I,'kl')
                    for index, row in df2.iterrows():
                        dt=parse(row['text'])
                        row['text']=(dt.strftime('%d/%m/%Y'))
                        # print(row['text'],l1)
                    #df2=df2[df2['text']==l1]
                    #print(df2)
                        if row['text']==l1:
                            
                            n['left']= (row['left'] * row['image_widht'])
                            n['right']=(row['right'] * row['image_widht'])
                            n['top'] =(row['top'] * row['image_height'])         
                            n['bottom']=(row['bottom'] * row['image_height'])
                            n["field_shape"]=wordshape(str(res["result"]['document']['documentInfo'][1]['correctedValue']))
                            if I>1:
                                #more than 1 bb is present
                                n['status']=0
                                # print('kk')
                            else:
                                n['status']=1
                                
                                # filt=pd.read_csv("E:/sample.csv")
                                # filt = filt[filt["vendor_id"] == n['vendor_id']]
                                # filt = filt[filt['field_name']==n['field_name']]
                                # filt = filt[filt['status']==1]
                                # print('kk')
                                # for k,l in filt.iterrows():
                                #     if ((int(n['left'])>=int(l['right'])) | (int(n['right'])<=int(l['left'])) | (int(n['top'])>=int(l['bottom'])) | (int(n['bottom'])<=int(l['top']))):
                                #         l['status']=0
                                #         print('need to update')
                                #         a=filt.index[filt['right'] == l['right']].tolist()
                                #         for kk in a:
                                #             filt1.at[kk,'status']=0
                                #         print(filt1)
                                #         print(l,'kko',a)
                                #     else:
                                #         print('no problem')
                                        
                            df = df.append(n, ignore_index = True)
                else:
                    n["field_shape"]=wordshape(str(res["result"]['document']['documentInfo'][1]['fieldValue']))
                    n['left']=res["result"]['document']['documentInfo'][1]['boundingBox']['left']
                    n['right']=res["result"]['document']['documentInfo'][1]['boundingBox']['right']
                    n['top']=res["result"]['document']['documentInfo'][1]['boundingBox']['top']
                    n['bottom']=res["result"]['document']['documentInfo'][1]['boundingBox']['bottom']
                    n['status']=1
                    # filt1=pd.read_csv("E:/sample1.csv")
                    # filt = filt1[filt1["vendor_id"] == n['vendor_id']]
                    # filt = filt[filt['field_name']==n['field_name']]
                    # filt = filt[filt['status']==1]
                    # print('kk2090')
                    # for k,l in filt.iterrows():
                    #     if ((int(n['left'])>=int(l['right'])) | (int(n['right'])<=int(l['left'])) | (int(n['top'])>=int(l['bottom'])) | (int(n['bottom'])<=int(l['top']))):
                    #         #l['status']=0
                    #         print(filt['right'],(l['right']),'lp')
                    #         a=filt.index[filt['right'] == l['right']].tolist()
                    #         for kk in a:
                    #             filt1.at[kk,'status']=0
                    #         print(filt1)
                    #         print(l,'kko',a)
                    #     else:
                    #         print('no problem')
                    df = df.append(n, ignore_index = True)
            try:
                os.remove(pred_file_path)
                print("file removed :",pred_file_path)
            except :
                pass
        return df
    except:
        print('there is an exception',traceback.print_exc())
        return df

def keep_per_vendor_10_records(df):
    df_copy = copy.deepcopy(df)
    try:
        droping_index = []
        df1 = df[df['vendor_id'].notna()]
        # converting to date col
        df1['submitted_on'] =  pd.to_datetime(df1['submitted_on'])    
        for item in df["vendor_id"].unique():
            df2 = df1[df1["vendor_id"]== item]
            df2 = df2.sort_values(by=["submitted_on"],ascending=True)
            if not(df2.shape[0]):
                continue
            elif df2.shape[0]<10:
                continue
            else:
                counter = 0
                for row in df2.itertuples():
                    #print(row)
                    if counter > 9:
                        droping_index.append(row.Index)
                    counter = counter + 1
        print("num rows to delete :",len(droping_index))
        print("df shape b4r removing records",df.shape)
        df.drop(droping_index,axis=0, inplace=True)
        print("df shape after removing records",df.shape)
        # print(df["vendor_id"].value_counts())
        return df
    except:
        print("removing record more than 10. exception",traceback.print.exc())
        return df_copy



def main():
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        log_file_path = os.path.join(ROOT_DIR, "./files/Reference_Data_Auto_Update.log")
        log_file = open(log_file_path,"w")
        sys.stdout = log_file
        sys.stderr = log_file
        print("Strt Refreshing reference data :",datetime.datetime.now())
        REFERENCE_MASTER_DATA_PATH = cfg.getReferenceMasterData()  
        REFERENCE_MASTER_DATA = os.path.join(ROOT_DIR,REFERENCE_MASTER_DATA_PATH)
        print("REFERENCE_MASTER_DATA :",REFERENCE_MASTER_DATA)
        document_ids = get_all_reviewed_docs()
        print("document_ids :",len(document_ids))
        df = get_reference_data(document_ids)
        print("df :",df.shape)
        if not(df.shape[0]):
            return
        ref_data = pd.read_csv(REFERENCE_MASTER_DATA, encoding='unicode_escape')
        print("ref_data :",ref_data.shape)
        #ref_data.append(df,ignore_index=True)
        duplicate_doc = set(df["doc_id"]).intersection(set(ref_data["doc_id"]))
        print("duplicate_doc :",len(duplicate_doc))
        for val in duplicate_doc:
            ref_data.drop(ref_data.index[ref_data["doc_id"]== val],inplace=True)
        print("ref_data shape after removing duplicate befor appending df :",ref_data.shape)
        ref_data = pd.concat([ref_data,df], axis = 0)
        ref_data = keep_per_vendor_10_records(ref_data)
        print("ref_data after appending :",ref_data.shape)
        ref_data.to_csv(REFERENCE_MASTER_DATA,index = False)
        return 
    except:
        print("main function exception ",traceback.print_exc())
        return 
    finally:
        if log_file is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()


if __name__ == '__main__':

    main()
