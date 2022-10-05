import sys
from post_processor import post_process
from post_processor import build_final_json
import preProcUtilities as putil
import pandas as pd
import os
import time

docMetaData = {
  'id': 'api.document.get',
  'ver': '1.0',
  'ts': 1647583464002,
  'params': {
    'resmsgid': '',
    'msgid': '',
    'status': 'Success',
    'err': '',
    'errmsg': ''
  },
  'responseCode': 'OK',
  'result': {
    'document': {
      '_id': '623420b1527d56ef0117e1e6',
      'documentId': 'doc_1647583407781_a9aa8b98a8b',
      'fileName': 'e invoice SIHPT2021-156 Sodexo.pdf',
      'documentType': 'Invoice',
      'mimeType': 'application/pdf',
      'uploadUrl': '/import/e invoice SIHPT2021-156 Sodexo.pdf',
      'size': 51339,
      'orgType': 'ORG_TYPE_001',
      'orgTypeId': 'ORG_001',
      'docType': 'DOC_TYPE_001',
      'docTypeId': 'DOC_001',
      'status': 'EXTRACTION_INPROGRESS',
      'submittedBy': 'system',
      'submittedOn': 1647583409182,
      'createdOn': 1647583409184,
      'lastUpdatedOn': 1647583417939,
      # 'bar_qr_data': {
      #   '0': [
      #     {
      #       'File Name': 'doc_1647583407781_a9aa8b98a8b-0.tiff',
      #       'Data Type': 'QR CODE',
      #       'Raw Extracted Data': 'eyJhbGciOiJSUzI1NiIsImtpZCI6IjQ0NDQwNUM3ODFFNDgyNTA3MkIzNENBNEY4QkRDNjA2Qzg2QjU3MjAiLCJ0eXAiOiJKV1QiLCJ4NXQiOiJSRVFGeDRIa2dsQnlzMHlrLUwzR0JzaHJWeUEifQ.eyJkYXRhIjoie1wiU2VsbGVyR3N0aW5cIjpcIjM2QUFEQ1M0MDE0SjFaSFwiLFwiQnV5ZXJHc3RpblwiOlwiMzZBQUFDUjI1NDdRMVpYXCIsXCJEb2NOb1wiOlwiU0lIUFQvMjAyMS0xNTZcIixcIkRvY1R5cFwiOlwiSU5WXCIsXCJEb2NEdFwiOlwiMTAvMTEvMjAyMFwiLFwiVG90SW52VmFsXCI6NDU3OC4wLFwiSXRlbUNudFwiOjEsXCJNYWluSHNuQ29kZVwiOlwiMTUxNTkwNDBcIixcIklyblwiOlwiN2UwMGE1ZGNjOTg3OGQxYmZkODk2ZTg0YWZjM2FjNjM5NjVmNmU1NDI5NjQyOTM0Y2YzMDgxOTA3NzRjYjRhOFwiLFwiSXJuRHRcIjpcIjIwMjAtMTEtMTAgMTI6MzY6MDBcIn0iLCJpc3MiOiJOSUMifQ.j6bQoNAe28Nk6HJPQrUwZCFpASDS7X6TKjGJF67zFjngBbp6qrarmZ4NJXXFbfW2-GaUG6m-sh1ochB2dh0PngEvs797CX21DtDE8OthMgUbiZSf9TIybChigAYncsxIXmzPueg4bmFNSqjxkldfCRIoW8AxanwD2M3P7ZZPVFDg33Y4xjL1qABJrI37aJrTDIRDo3qmDvUnE6IXoZEsCYIePFWdtODDXWof847V49kLc8Uf_gug8yxiLkYigvUHRNVNHnsrBaTWsy4Ff-1IT2npNTFwjxEAgetYfhQDPX9rj1S7uOgE7-4NSM6Rk80kQNAW_Yp2LWf1u6UFrtasrw',
      #       'Decoded Data': {
      #         'data': '{"SellerGstin":"36AADCS4014J1ZH","BuyerGstin":"36AAACR2547Q1ZX","DocNo":"SIHPT/2021-156","DocTyp":"INV","DocDt":"10/11/2020","TotInvVal":4578.0,"ItemCnt":1,"MainHsnCode":"15159040","Irn":"7e00a5dcc9878d1bfd896e84afc3ac63965f6e5429642934cf308190774cb4a8","IrnDt":"2020-11-10 12:36:00"}',
      #         'iss': 'NIC'
      #       }
      #     },
      #     {
      #       'File Name': 'doc_1647583407781_a9aa8b98a8b-0.tiff',
      #       'Data Type': 'BAR CODE',
      #       'Raw Extracted Data': '112010145055519',
      #       'Decoded Data': '112010145055519'
      #     }
      #   ]
      # },
      'lastProcessedOn': 1647583409,
      'lastUpdatedBy': 'system',
      'name': 'doc_1647583407781_a9aa8b98a8b',
      'pageCount': 1,
      'stage': 'EXTRACTION',
      'statusMsg': 'Extraction process initiated',
      'lastSubmittedOn': 1647583417939
    }
  }
}

def helper_main(file_path, docMetaData):
    """
    Read entire OCR output and separate into different files
    Extract FileName first
    """

    # Check no LI Items
    DF = pd.read_csv(file_path, index_col = 0)

    DF['FileName']='hi'
    print("preprocessing!!!")
    prediction,stp,crit,format_= post_process(DF, docMetaData=docMetaData)

    prediction = build_final_json(prediction)
    print("^^^^^^^^^^^^^^^^^")
    print(prediction)
    mean_ocr_conf = DF['conf'].mean()
    median_ocr_conf = DF['conf'].median()
    std_ocr_conf = DF['conf'].std()

    return prediction, mean_ocr_conf, median_ocr_conf, std_ocr_conf

def get_output_dict(pred, vendor_id, document_id, mean_ocr_conf, median_ocr_conf, std_ocr_conf):
    data={}
    data["Vendor_id"]=vendor_id
    data["Document_id"]=document_id
    data['status']="FileFound"
    data['mean_ocr_conf'] = mean_ocr_conf
    data['median_ocr_conf'] = median_ocr_conf
    data['std_ocr_conf'] = std_ocr_conf
    datta=[]

    pred_header = pred['documentInfo']
    pred_lineitem = pred['documentLineItems']

    for p in pred_header:
        data[p['fieldId']] = p['fieldValue']
        data[p['fieldId']+ "_conf"] = p['confidence']
        data[p['fieldId']+ "_ocr_conf"] = p['OCRConfidence']

    for p in pred_lineitem:
        dat={}
        dat["Vendor_id"] = vendor_id
        dat["Document_id"] = document_id
        dat['status'] = "FileFound"
        dat['page_num'] = p['pageNumber']
        dat['line_item_num'] = p['rowNumber']
        dat['mean_ocr_conf'] = mean_ocr_conf
        dat['median_ocr_conf'] = median_ocr_conf
        dat['std_ocr_conf'] = std_ocr_conf
        l_ = p['fieldset']
        for l in l_:
            dat[l['fieldId']] = l['fieldValue']
            dat[l['fieldId'] + "_conf"] = l['confidence']
            dat[l['fieldId'] + "_ocr_conf"] = l['OCRConfidence']
        datta.append(dat)
    print([data],"\n\n\n\n\n\n\n\n")
    print(datta,"\n\n\n\n\n\n\n\n")
    # assert(1==2)
    return [data],datta


def generate_bulk(csv_file_path, folder_path):
    DF_FILEINFO=pd.read_csv(csv_file_path)
    DF_FILEINFO.dropna(inplace=True)
    DF_HEADER = pd.DataFrame()
    DF_LINEITEMS = pd.DataFrame()
    for idx,rows in DF_FILEINFO.iterrows():
        vendor_ID = rows['Vendor_ID']
        document_ID = rows['Document_ID']
        file_name = document_ID+"_pred.csv"

        file_path = os.path.join(folder_path, file_name)
        print("Bulk processing document: ", file_path)
        exist=os.path.isfile(file_path)
        if exist:
            pred, mean_ocr_conf, median_ocr_conf, std_ocr_conf = helper_main(file_path, docMetaData)

            list_headers, list_line_items = get_output_dict(pred, vendor_ID, document_ID,
                mean_ocr_conf, median_ocr_conf, std_ocr_conf)
            temp = pd.DataFrame(list_headers)

            # Append temp to DF_HEADER
            DF_HEADER=DF_HEADER.append(temp,ignore_index=True)
            temp = pd.DataFrame(list_line_items)
            # Append temp to DF_LINEITEMS
            DF_LINEITEMS=DF_LINEITEMS.append(temp,ignore_index=True)
        else:
            print("HELLOOOOOOOOOO")
            d=[]
            datta={}
            datta["Vendor_id"]=vendor_ID
            datta["Document_id"]=document_ID
            datta['status']="FileNotFound"
            d.append(datta)
            print(datta)
            DF_HEADER=DF_HEADER.append(d,ignore_index=True)
            print(DF_HEADER['status'])

            DF_LINEITEMS=DF_LINEITEMS.append(d,ignore_index=True)

    print("##############################")
    print(DF_HEADER)
    print(DF_LINEITEMS)
    print(temp)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    DF_HEADER.to_csv("../header_item8_"+timestr+".csv",mode='a')
    DF_LINEITEMS.to_csv("../line_items_"+timestr+".csv",mode='a')


def main():
    """
    Script needs input folder path to iterate over
    :return:
    """
    
    file_path = r"/Volumes/Macintosh HD - Data/workspace/Swiggy_Data/pred_files/c6525271-3848-40eb-9f5e-6907df727354_pred.csv"
    # document_id = os.path.basebame(file_path).split("_pred")[0]
    # document_id = "doc_1640339955202_a88baa89989"
    # docMetaData = putil.getDocumentApi(document_id, callbackUrl='http://13.71.23.200:9999')
    print("docMetaData: ", docMetaData)
    csv_file_path= ""

    args = sys.argv
    print(args)
    if len(args) == 1:
        helper_main(file_path, docMetaData)
    elif (len(args) == 3):
        generate_bulk(args[1],args[2])
    else:
        print("invalid arguments")
    print("Exiting main!!!")


if __name__ == "__main__":
    main()
