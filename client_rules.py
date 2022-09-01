# from calendar import SATURDAY
from dataclasses import field
import json
import os
# from pickle import TRUE
import re
from tabnanny import check
# from tkinter.messagebox import NO
from dateutil import parser
from difflib import SequenceMatcher
import extract_data as extd
import pandas as pd
import copy
import ast
from business_rules import add_new_field, reduce_field_confidence
from business_rules import reduce_amount_fields_confidenace, reduction_confidence_taxes
import preProcUtilities as putil
import traceback
# from collections import OrderedDict


# import TAPPconfig as config

# Read Client Configurataions File

script_dir = os.path.dirname(__file__)

ClientConfigFilePath = os.path.join(script_dir,
                              "Utilities/client_config.json")

ClientFieldMappingPath = os.path.join(script_dir,
                              "Utilities/CLIENT_FIELD_MAPPING_ORDERING.json")

date_fields = ["invoiceDate", "dueDate"]

dict_org = {"VEPL": {"GSTIN":"29AAFCV1464P2ZM","NAME":"VELANKANI ELECTRONICS"},
"VISL": {"GSTIN":"29AABCV0552G1ZF","NAME":"VELANKANI INFORMATION"},
"OTERRA": {"GSTIN":"29AABCV0552G1ZF","NAME":"THE OTERRA"},
"BYD": {"GSTIN":"29AABCB5845J1ZE","NAME":"BYDESIGN"},
"46OUNCES": {"GSTIN":"29AABCV0552G3ZD","NAME":"46 OUNCES"}}


imageTemplatePath = os.path.join(script_dir,
                              "Utilities/IMAGE_TEMPLATE.csv")

IMAGE_TEMPLATE = pd.read_csv(imageTemplatePath, encoding='unicode_escape')

def find_similarity_words(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def read_json(json_file_path = ClientConfigFilePath):
    """
    """
    rules = {}
    with open(json_file_path) as json_file:
        rules = json.load(json_file)
    return rules

CONFIGURATIONS = read_json()

CLIENT_FIELD_MAPPING = read_json(ClientFieldMappingPath)
SPECIAL_CHARS_LIST = CONFIGURATIONS["SPECIAL_CHARS_REMOVE"]

def extract_barcode(DF):
    """
    New method added to extract line items from model output
    Row Number is taken from line_row
    """
    # Get list of all tokens in invoice
    list_matcher = CONFIGURATIONS["BARCODE_PTN"]
    list_of_tokens = DF['text'].astype(str).to_list()
    filtered_values = []
    for matcher in list_matcher:
        l = list(filter(lambda v: re.match(matcher, v), list_of_tokens))
        filtered_values.extend(l)

    # Reassign row numbers
    final_candidates_ = {}
    if len(filtered_values)>0:
        extracted_value = filtered_values[0]
        final_candidates_['line_num'] = 0
        final_candidates_["prob_Preform"] = 1
        final_candidates_['text'] = extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = 0
        final_candidates_['right'] = 1
        final_candidates_['conf'] = 1
        final_candidates_['top'] = 0
        final_candidates_['bottom'] = 1

        final_candidates_['page_num'] = 0
        final_candidates_['image_height'] = 1
        final_candidates_['image_widht'] = 1

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = 1

        final_candidates_['final_confidence_score'] = 1
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = False

        return {"barCode": final_candidates_}
    else:
        return None

def extract_preform(DF):
    """
    New method added to extract line items from model output
    Row Number is taken from line_row
    """
    # Get list of all tokens in invoice
    list_of_tokens = DF['text'].astype(str).to_list()
    # list_of_tokens = [remove_special_charcters(token) for token in list_of_tokens]
    list_of_tokens = [i.upper() for i in list_of_tokens if i.isalpha()]
    list_of_tokens = list(set(list_of_tokens))
    preform_tokens = CONFIGURATIONS["PREFORM_TXT"]
    preform_match =  any(item in list_of_tokens for item in preform_tokens)

    final_candidates_ = {}
    if preform_match:
        extracted_value = "1"
        final_candidates_['line_num'] = 0
        final_candidates_["prob_Preform"] = 1
        final_candidates_['text'] = extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = 0
        final_candidates_['right'] = 1
        final_candidates_['conf'] = 1
        final_candidates_['top'] = 0
        final_candidates_['bottom'] = 1

        final_candidates_['page_num'] = 0
        final_candidates_['image_height'] = 1
        final_candidates_['image_widht'] = 1

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = 1

        final_candidates_['final_confidence_score'] = 1
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = False

        return {"Preform": final_candidates_}
    else:
        return None


def extract_org(DF):
    """
    """
    extracted_value = "NONE"
    confidence_score = 0.0
    DF['text'] = DF['text'].str.upper()
    DF = DF.sort_values(['page_num', 'line_num', 'word_num'])
    list_words = list(DF['text'])

    dict_match = {}
    dict_org_extracted = {}
    threshold = 0.8
    for org, val in dict_org.items():
        gstin = val['GSTIN']
        name = val['NAME']

        name_list = name.split(' ')
        N = len(name_list)
        list_name_search = [' '.join(list_words[i: i + N]) for i in range(len(list_words)- N + 1)]
        list_gstin_search = list_words

        name_score = max([find_similarity_words(s, name) for s in list_name_search])
        gstin_score = max([find_similarity_words(s, gstin) for s in list_gstin_search])
        final_score = 0.7*gstin_score + 0.3*name_score
        dict_match[org] = {"NAME": name_score, "GSTIN": gstin_score, "FINAL": final_score}
        if final_score > threshold:
            dict_org_extracted[org] = final_score

    if len(dict_org_extracted) == 1:
        extracted_value = list(dict_org_extracted.keys())[0]
        confidence_score = dict_org_extracted[extracted_value]
    elif (len(dict_org_extracted) == 2) and ("VISL" in dict_org_extracted) and ("OTERRA" in dict_org_extracted):
        # Case 1: Vendor -> VISL: GSTIN and and VISL and No Oterra in the document.
        # Name Match Score for OTERRA will be lower than threshold
        if (dict_match['OTERRA']['NAME'] < threshold) and (dict_match['VISL']['NAME'] >= threshold):
            # Mark as VISL document
            del dict_org_extracted["OTERRA"]
            extracted_value = list(dict_org_extracted.keys())[0]
            confidence_score = dict_org_extracted[extracted_value]
        elif (dict_match['OTERRA']['NAME'] >= threshold) and (dict_match['VISL']['NAME'] < threshold):
            # Mark as VISL document
            del dict_org_extracted["VISL"]
            extracted_value = list(dict_org_extracted.keys())[0]
            confidence_score = dict_org_extracted[extracted_value]

    final_candidates_ = {}
    final_candidates_['line_num'] = 0
    final_candidates_["prob_Preform"] = 1
    final_candidates_['text'] = extracted_value
    final_candidates_["Label_Present"] = True
    final_candidates_['word_num'] = 0

    final_candidates_['left'] = 0
    final_candidates_['right'] = 1
    final_candidates_['conf'] = 1
    final_candidates_['top'] = 0
    final_candidates_['bottom'] = 1

    final_candidates_['page_num'] = 0
    final_candidates_['image_height'] = 1
    final_candidates_['image_widht'] = 1

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = confidence_score

    final_candidates_['final_confidence_score'] = confidence_score
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = False

    return {"ORG": final_candidates_}

def insurance_validation(DF, prediction):
    """


    Returns
    -------
    None.

    """
    def map_fields(dict_insurance, mapping_dict, res_dict=None):

        res_dict =  {}
        print(dict_insurance.items())
        for k, v in dict_insurance.items():
            print("Key: ", k)
            # if isinstance(v, dict):
            #     v = map_fields(v, mapping_dict[k])
            if k in mapping_dict.keys():
                k = str(mapping_dict[k])
            res_dict[k] = v
        return res_dict


    dict_insurance = extd.extract_data(DF)
    _image_height = DF.iloc[0]['image_height']
    _image_width = DF.iloc[0]['image_widht']
    print("DICT INSURANCE")
    print(dict_insurance)
    if dict_insurance['doc_type'] == "UNKNOWN":
        return {}
    name_map = {'doc_type': 'Document', 'doc_number': 'Document No',
                'DOB': 'Date of Birth', "NAME" : "Name"}
    dict_insurance = map_fields(dict_insurance, name_map)
    insurance_items = list(dict_insurance.items())
    print(insurance_items)
    all_insurance_candidates = {}
    for item in insurance_items:
        _extracted_value = item[1]
        _left = 0
        _right = 1
        _top = 0
        _bottom = 1
        _conf = 1
        if isinstance(item[1],dict):
            _extracted_value = item[1]['extracted_value']
            _left = item[1]['left']
            _right = item[1]['right']
            _top = item[1]['top']
            _bottom = item[1]['bottom']
            _conf = item[1]['conf']
        final_candidates_ = {}
        final_candidates_['line_num'] = 0
        final_candidates_["prob_"+str(item[0])] = 1
        final_candidates_['text'] = _extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = _left
        final_candidates_['right'] = _right
        final_candidates_['conf'] = _conf
        final_candidates_['top'] = _top
        final_candidates_['bottom'] = _bottom

        final_candidates_['page_num'] = 0
        final_candidates_['image_height'] = _image_height
        final_candidates_['image_widht'] = _image_width

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = _conf

        final_candidates_['final_confidence_score'] = _conf
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = False
        all_insurance_candidates.update({item[0] : final_candidates_})
        print(all_insurance_candidates)
    return all_insurance_candidates


def remove_chars(s, chars):
    """
    """
    return re.sub('[' + re.escape(''.join(chars)) + ']', '', s)

def remove_special_chars(s):
    """
    """
    return re.sub('[^A-Za-z0-9]+', '', s)

def remove_special_chars_alphabets(s):
    """
    """
    return re.sub('[^0-9]+', '', s)

def clean_HSNCode(prediction):
    """
    Method to remove special characters from HSNCode
    """
    print("Inside clean_HSNCode!!!")
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            predicted_value = item[col_name]
                            if col_name == "HSNCode":
                                predicted_text = item[col_name]['text']
                                updated_text = remove_special_chars(str(predicted_text))
                                item[col_name]['text'] = str(updated_text)
    return prediction

def clean_PONumber(prediction):
    """
    Keep just the digits in PONumber
    """
    for key, val in prediction.items():
        if key == "poNumber":
            if val is not None:
                text = val['text']
                updated_text = remove_special_chars_alphabets(text)
                val['text'] = updated_text
    return prediction

def convert_dates(prediction):
    """
    """
    print("convert_dates")
    for key, val in prediction.items():
        if key in date_fields:
            print(key, val)
            if val is not None:
                text = val['text']
                try:
                    #for indian invoices
                    converted_text = parser.parse(text, dayfirst=True).date().strftime('%d/%m/%Y')
                    #for us invoices
                    #converted_text = parser.parse(text, dayfirst=False).date().strftime('%m/%d/%Y')
                    
                    val['text'] = converted_text
                except:
                    print("Convert_dates",traceback.print_exc())
                    val['prob_invoiceDate'] = 0.4
                    val['model_confidence'] = 0.4
                    val['final_confidence_score'] = 0.4
                    pass

    return prediction

def discard_additional_LI_rows(prediction):
    """
    return > Prediction
    """
    try: 
        print("Discarding unwanted rows in line item prediction")
        mandatory_fields = set(['itemQuantity', 'unitPrice', 'itemValue'])
        mandatory_fields = set(['itemValue'])
        #only for best choice

        pred = {}
        rows_to_discard = []
        po = prediction.get("poNumber")
        if po is not None:
            ponumber = po.get("text")
            if ponumber is None:
                ponumber = " "
            if "wo" not in ponumber.lower():
                mandatory_fields = set(['itemQuantity',
                                        'unitPrice',
                                        'itemValue'])

        for key, val in prediction.items():
            if key == "lineItemPrediction":
                if val is not None:
                    changed_pred = {}
                    for page, page_prediction in val.items():
                        row_cols = []
                        changed_page_prediction = {}
                        for row, row_prediction in page_prediction.items():
                            rows = list(page_prediction.keys())
                            row_pred = {}
                            for item in row_prediction:
                                col_name = list(item.keys())[0]
                                predicted_value = item[col_name]['text']
                                if col_name == "itemValue":
                                    pred_value_ = predicted_value.replace(",","")
                                    pred_value_ = pred_value_.replace(".","")
                                    pred_value_ = pred_value_.replace(" ","")
                                    if pred_value_.isdigit():
                                        if int(pred_value_) > 0:
                                            row_pred[col_name] = predicted_value
                                else:
                                    row_pred[col_name] = predicted_value
                            row_fields = set(list(row_pred.keys()))
                            if not mandatory_fields.issubset(row_fields):
                                row_cols = [elem for elem in list(mandatory_fields) if elem in list(row_fields)]
                                if set(row_cols) != mandatory_fields:
                                    rows_to_discard.append(row)
                            rows_to_keep = [i for i in rows if i not in rows_to_discard]
                        for i in rows_to_keep:
                            changed_page_prediction = {**changed_page_prediction,**{i:page_prediction[i]}}
                        changed_pred[page] = changed_page_prediction
                    pred[key] = changed_pred
            else:
                pred[key] = val

        return prediction
    except :
        print(" Did't went through discard_additional_LI_rows function")
        return prediction
# added spacifically for BCP demo
def discard_lines_without_mandatory_fields(prediction):
    pred = copy.deepcopy(prediction)
    try:
        mandatory_fields_set1 =set(["itemDescription","itemValue"])
        mandatory_fields_set2 = set(["itemDescription","unitPrice","itemQuantity"])
        line_items = pred.get("lineItemPrediction")
        for page, page_val in line_items.items():
            if page_val is not None:
                for line,line_val in page_val.items():
                    row_col = []
                    for item in line_val:
                        x = list(item.keys())
                        row_col = row_col + x
                    #rint("row_cols",row_col)
                    if not mandatory_fields_set1.issubset(row_col):
                        print("row_cols not subset of of mandatory_fields_set1",row_col)
                        del prediction["lineItemPrediction"][page][line]
                    elif not mandatory_fields_set2.issubset(row_col):
                        print("row_cols not subset of of mandatory_fields_set2",row_col)
                        del prediction["lineItemPrediction"][page][line]

        return prediction 
    except:
        print("discard_lines_without_mandatory_fields exception ",traceback.print_exc())
        return pred
        
def remove_LI_fields(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
        return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def demo_change(prediction):
    """
    Remove field from line items
    
    """
    try:
        mandatory=[]
        pred=prediction.get('lineItemPrediction')
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    for key,val in field.items():
                        mandatory.append(key)
                        
                        
                    
        
        mandatory=(set(mandatory))
        print(mandatory)
        if len(mandatory)==1:
            pred.clear()
            return prediction
        else:
            return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def remove_LI_field_po(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        po=prediction.get("poNumber")
        print(po)
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
            if po["text"]=="984583":
                print("yes")
                del (pred[key][3])
                del (pred[key][4])
                del (pred[key][5])
                return prediction
            else:
                return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def remove_LI_field_AUZ(prediction):
    """
    Remove field from line items
    
    """
    try:
        pred=prediction.get('lineItemPrediction')
        po=prediction.get("poNumber")
        print(po)
        for key,val in pred.items():
            for row,fields in val.items():
                for field in fields:
                    print(field)
            if po["text"]=="64618":
                print("yes")
                del (pred[key][27])
                return prediction
            else:
                return prediction
    except:
        print("LineItemPrediction is None")
        return prediction

def present_doc_output(prediction, doc_type, org_type):
    """
    Filtering document result based on docType and orgType
    """
    print("Filtering document result based on {} and {}".format(doc_type, org_type))
    if org_type.upper().strip()=="KYC" and doc_type.upper() in tuple(["PAN","AADHAR","PASSPORT"]):
        keys_to_keep =  ['Document','Document No','Date of Birth',"Name","lineItemPrediction"]
        keys_to_discard = [key for key in prediction.keys() if key not in keys_to_keep]
    elif org_type.upper().strip()=="ACC PAYABLE" and doc_type.upper() in tuple(["INVOICE"]):
        keys_to_discard = ['Document','Document No','Date of Birth',"Name"]
    else:
        keys_to_discard = []
    print(keys_to_discard)
    changed_prediction = {key:val for key, val in prediction.items() if key not in keys_to_discard}
    print("New prediction based on doc type")
    print(changed_prediction)
    return changed_prediction


def make_vendor_info_editable(prediction):
    """
    Make vendorName and vendorAddress editable
    """
    print("make_vendor_info_editable")
    for key, val in prediction.items():
        if (val is not None) and (key in ["vendorName", "vendorAddress","vendorGSTIN"]):
            val['extracted_from_masterdata'] = False

    return prediction

def extract_image(prediction, vendor_id):
    """
    """
    print("Inside extract_image")
    global IMAGE_TEMPLATE
    IMAGE_TEMPLATE = pd.read_csv(imageTemplatePath, encoding='unicode_escape')
    print(vendor_id)
    print(IMAGE_TEMPLATE)
    print(IMAGE_TEMPLATE.columns)
    print(dict(IMAGE_TEMPLATE.iloc[0]))
    TEMP = IMAGE_TEMPLATE.loc[IMAGE_TEMPLATE["VENDOR_ID"] == vendor_id]
    
    extracted_images = {}
    for idx_, row in TEMP.iterrows():
        template = dict(row)
        image_name = template["IMAGE_NAME"]
        page_num = template["PAGE_NUM"]
        image_num = template["IMAGE_NUM"]

        final_candidates_ = {}
        final_candidates_['line_num'] = 0
        final_candidates_["prob_"+ image_name] = 1
        final_candidates_["field_type"] = "IMAGE"
        final_candidates_["page_num"] = page_num
        final_candidates_["image_num"] = image_num
        final_candidates_["text"] = str(page_num) + "_" + str(image_num)

        final_candidates_['line_num'] = 0
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = 0
        final_candidates_['right'] = 1
        final_candidates_['conf'] = 1
        final_candidates_['top'] = 0
        final_candidates_['bottom'] = 1

        final_candidates_['page_num'] = 0
        final_candidates_['image_height'] = 1
        final_candidates_['image_widht'] = 1

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = 1

        final_candidates_['final_confidence_score'] = 1
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = False

        extracted_images[image_name] = final_candidates_

    return extracted_images

########### qr extractions ############
# Get Unified labels from qr code json

def getUnifiedDict(dictionary,Uniform_lbl_dict):
    Unified_QRCode_Json = copy.deepcopy(dictionary)      
    for key, val in dictionary.items():
        if (isinstance(val, dict)):
            #print("values", val)
            for Nested_Key in val: 
                #print("Nested_Key",Nested_Key)
                for new_key, values in Uniform_lbl_dict.items():
                    #print('Key :: ', new_key)
                    if(isinstance(values, list)):
                        for value in values:
                            if value == Nested_Key:
                                print("match new key",new_key)       
                                print("key_match val",value)
                                #print(" Neated key",Nested_Key )
                                #Unified_QRCode_Json[key]
                                for u_k, u_v in  Unified_QRCode_Json.items():
                                    if u_k == key:
                                    #if (isinstance(u_v,list)):
                                        u_v[new_key] = u_v.pop(Nested_Key)
                                        #del Unified_QRCode_Json[Nested_Key]
                                        print("update key",u_v[new_key])
    return Unified_QRCode_Json

def SerializeKeys(dictionary, prefix):
    serial_key_dict = copy.deepcopy(dictionary)
    i = 1
    print("Serializing dicts keys",dictionary)
    for k, v in dictionary.items():
        print(" b4 serialise", k)
        if (k == "QRCODE_STATUS" or k == "BARCODE_STATUS"):
            pass
        else:
            new_key = prefix + str(i)
            print("New Key ", new_key)
            #serial_key_dict.update(new_key,v)
            serial_key_dict[new_key] = serial_key_dict[k]
            del serial_key_dict[k]
            i = i+1
    return serial_key_dict

def deleteDuplicates(dictionary):
    temp = []
    res = dict()
    print("res valu",res )
    for key, val in dictionary.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    temp = None
    
    return res

##### get Barcode QR Code Data ######
def Get_BAR_QR_CodeData(docMetaData):
    """
    returns Bar/ code jsons
    """
    Uniform_lbl_dict = {"invoiceNumber" : ["invoice_code","invoice number","Invoice No","Bill No","Bill number","DocNo"],
           "invoceDate" : ["invoice_date","invoice date","created","Bill Date","Dated","DocDt"],
           "totalAmount" : ["total","gross total","grand total","TotInvVal"],
           "irnNumber": ['Irn','irn'],
           "irnDate":['IrnDt'],
           "vendorGSTIN":['SellerGstin'],
           "buyerGSTIN":['BuyerGstin'],
           "HSNCode":['MainHsnCode']

          }


    docMetaData = docMetaData.get('result')
    if docMetaData is None:
        return None
    docMetaData = docMetaData.get("document")
    if docMetaData is None:
        return None
    #print("BARCodeDataResult",docMetaData.keys())
    QRCodeData = {}
    BARCodeData = {}
    docMetaData = docMetaData.get('bar_qr_data')
    #print("BARCodeData",docMetaData)
    for k1 , v1 in docMetaData.items():
        print("k1", k1)
        if str(v1) == '[]':
            #QRCodeData[k1] = {"QRCodeStatus":"NotDetected"}
            pass
        else:
            #print(qr_coded_details)
            barcodes_json = {}
            if isinstance(v1,list):
                #print("key values",v1)
                for item in v1:
                    if item:
                        #print("qwrrtt",item)
                        #print(item.keys())
                        if item['Data Type'] =='QR CODE':
                            print('QR code Found')

                            if type(item["Decoded Data"]) == str:
                                #print("String",item['Decoded Data'])
                                try: 
                                    item_dict = ast.literal_eval(re.search('({.+})', item["Decoded Data"]).group(0))
                                    print("extracted from string",item_dict.keys())
                                    for ks in item_dict.keys():
                                        if ks == 'data':
                                            dict_item = item_dict['data']
                                            print("data",type(dict_item))
                                            print(dict_item)
                                            #if type(item["data"]) == str:
                                            try :
                                                QRCodeData[k1] = ast.literal_eval(re.search('({.+})', dict_item).group(0))
                                            except AttributeError as e:
                                                print("string does not contain dict")
                                                QRCodeData[k1] = dict_item


                                except AttributeError as e:
                                    QRCodeData[k1] = {'QRCodeStatus':'Notreadable'}
                                    print("string does not contain diitemct",QRCodeData)
                            else:
                                item_dict = item["Decoded Data"]
                                #print(" print_dict ",item_dict.keys())
                                for ks in item_dict.keys():
                                    if ks == 'data':
                                        dict_item = item_dict['data']
                                        #print("data",type(dict_item))
                                        #print(dict_item)
                                        try :
                                            QRCodeData[k1] = ast.literal_eval(re.search('({.+})', dict_item).group(0))
                                        except AttributeError as e:
                                            #print(" is dict ")
                                            QRCodeData[k1] = dict_item
                        #else:
                         #   QRCodeData[k1] = {"QRCodeStatus":"NotDetected"}
                        if item['Data Type'] == "BAR CODE":
                            BARCodeData[k1] = item["Decoded Data"]

    print("QRCode b4 delte duppictes",QRCodeData)
    
    print("BARCodeData b4 delte duppictes",BARCodeData)
    QRCodeData = deleteDuplicates(QRCodeData)
    BARCodeData = deleteDuplicates(BARCodeData)
    print("BARCodeData after delte duppictes",BARCodeData)
    print("qtrdt",len(QRCodeData))
    if len(QRCodeData) == 0:
        QRCodeData = {"QRCODE_STATUS": "NotDetected"}
    if len(BARCodeData) == 0:
        BARCodeData = {"BARCODE_STATUS": "NotDetected"}
   
    QRCodeData = SerializeKeys(QRCodeData, prefix = "QRCODE_")
    QRCodeData = getUnifiedDict(QRCodeData,Uniform_lbl_dict)
    
    BARCodeData =  SerializeKeys(BARCodeData, prefix = "BARCODE_")
    print("Serialized QR keys",BARCodeData)

    BAR_QR_CodeData = {"QRCodeData": QRCodeData, "BARCodeData":BARCodeData}
    print("BAR_QR_CodeData keys",BAR_QR_CodeData.keys())
            
    return BAR_QR_CodeData


# build QR code final json
def build_final_QRCode_json(prediction, docMetaData):
    '''
    
    return type:
    '''
    try:
        requiredFieldsFromQRCode = ['QRCODE_STATUS','BARCODE_STATUS',
                                    'invoiceNumber', 'invoiceDate','poNumber',
                                    'irnNumber','irnDate','totalAmount',
                                    'vendorGSTIN','buyerGSTIN','HSNCode']
        BR_QR_CodeJson = Get_BAR_QR_CodeData(docMetaData)
        if BR_QR_CodeJson is None:
            print("QR code data is None to add into pred, so written pred only")
            return prediction
        qr_candidates = {}
        print("BR_QR_CodeJson",BR_QR_CodeJson)
        #qr_candidates['QRCode_Extraction'] = []
        #if 
        
        for item_, value_ in BR_QR_CodeJson.items():
            dictiory = {}
            print("value_ ",value_)
            if item_ == "QRCodeData":
                print("inside qr")
                for k, v in value_.items():
                    if isinstance(v,dict):
                        print("inside sub dict",v)
                        for k1, v1 in v.items():
                            print(" k1 ",k1, "v1",v1)
                            if k1 in requiredFieldsFromQRCode:
                                # dictiory[item_+"_QR"] 
                                if str(v1) == "nan" :
                                    dictiory[k1 + "_QR"] = ""
                                else:
                                    dictiory[k1 + "_QR"] = str(v1)
                    else:
                        dictiory[k]= v
            qr_candidates.update(dictiory)
                                    
            if item_ == "BARCodeData":
                for k, v in value_.items():
                    dictiory[k] = str(v)
            qr_candidates.update(dictiory)
            
        print("QR candidates final: ", qr_candidates)

        # {}
        # qr fields extracted need to be formed in the post processor json structure
        # Before returning prediction, append qr fields
        qrjson = {}
        for key, val in qr_candidates.items():
            print(key)
            final_candidates_ = {}
            final_candidates_['line_num'] = 0
            final_candidates_["text"] = str(val)
            final_candidates_["prob_"+ str(key)] = val
            final_candidates_['line_num'] = 0
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = 0

            final_candidates_['left'] = 0
            final_candidates_['right'] = 1
            final_candidates_['conf'] = 1
            final_candidates_['top'] = 0
            final_candidates_['bottom'] = 1

            final_candidates_['page_num'] = 0
            final_candidates_['image_height'] = 1
            final_candidates_['image_widht'] = 1

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = 1
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False      
            qrjson[key] = final_candidates_
            #print("finalzzzzz",qrjson)
        print(" qrjson",qrjson)
        return {**prediction,**qrjson}
    except:
        print(" it did't went through form final QR Jsson")
        return prediction

####### validating and replacing model predition with QR Code data ######
def validate_Model_Prediction_with_QRCode_Data(docMetaData,prediction):
    """
    functions compares the validationFields of model prediction against the QR code 
    extracted and replaceswith QR code if its not noatches, and
    returns updated prediction

    """
    try:
        validationFields = ['invoiceNumber', 'invoiceDate','totalAmount','vendorGSTIN']
        
        uiDisplayFormate ={
                            'page_num': 0,
                            'line_num': 0,
                            'prob_': 1,
                            'text': '0',
                            'Label_Present': 0,
                            'word_num': 1,
                            'left': 0,
                            'right': 0,
                            'top': 0,
                            'bottom': 0,
                            'conf': 0,
                            'height': 0,
                            'width': 0,
                            'image_height': 1122,
                            'image_widht': 79,
                            'label_confidence': 0.0,
                            'wordshape': '',
                            'wordshape_confidence': 0.0,
                            'Odds': 0.4789099371265383,
                            'model_confidence': 1,
                            'final_confidence_score': 0,
                            'vendor_masterdata_present': False,
                            'extracted_from_masterdata': False
                        }

        QRCodeData = Get_BAR_QR_CodeData(docMetaData)
        print("QRCodeData only ",QRCodeData)
        if QRCodeData is None:
            print("QR code data is None to validate prediction")
            return prediction
        QRCodeData = QRCodeData.get("QRCodeData")
        if QRCodeData is None:
            print("QR code meta data is None to validate prediction")
            return prediction
        
        for key, val in prediction.items():
            #print(key)
            if isinstance(val,dict):
                for k, v in QRCodeData.items():
                    if isinstance(v,dict):
                        QRCodeData = v
                        print("qr dict 1 ",QRCodeData)
                        if key in QRCodeData.keys() and validationFields :
                            print("model prediction : ", key , val["text"])
                            print("qr extracted : ",key, QRCodeData[key])
                            if QRCodeData[key]:
                                if val["text"] != QRCodeData[key]:
                                    val["text"] = QRCodeData[key]
                                    print("Updated model prediton :",val["text"])
                                    break

            else:
                for k, v in QRCodeData.items():
                    if isinstance(v,dict):
                        QRCodeData = v
                        print("qr dict 2 ",QRCodeData)
                        if key in QRCodeData.keys() and validationFields :
                            
                            print("model prediction : ", key , val)
                            print("qr extracted : ",key, QRCodeData[key])
                            if QRCodeData[key]:
                                val =  uiDisplayFormate
                                val['prob_'+ key] = val.pop("prob_")
                                #print(val.keys())
                                val["text"] = QRCodeData[key]
                                print("Updated Model pred key : ",key, val["text"])
                                prediction[key] = val
                                print(" braking for loop")
                                break
                        
        return prediction 
    except:
        print("Prediction does not went throuh Validate model prediction against QR code data")
        return prediction

########### end of Qr Barcode extraction##########
##### bill / ship to Name and GISTIN extraction #########
# >>>>>>>>>> getting final candidte for GSTIN 
def final_candidates(row,text_col,prob_col):
    final_candidates_ = {}
    final_candidates_['page_num'] = row['page_num']
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["text"] = str(row[text_col])
    final_candidates_["prob_"+ str(prob_col)] = 1
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["Label_Present"] = True
    final_candidates_['word_num'] = row['word_num']

    final_candidates_['left'] = row['left']
    final_candidates_['right'] = row['right']
    final_candidates_['conf'] = row['conf']
    final_candidates_['top'] = row['top']
    final_candidates_['bottom'] = row['bottom']

    final_candidates_['image_height'] = row['image_height']
    final_candidates_['image_widht'] = row['image_widht']

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = 1

    final_candidates_['final_confidence_score'] = 1
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False      
    #final_candidate[candidate] = final_candidates_
    return final_candidates_

# >>>>>>>>>>>  getting final candidates for ship/bill to Name
def bill2shipName_final_candidates(row,text_col,prob_col):
    final_candidates_ = {}
    final_candidates_['page_num'] = row['page_num']
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["text"] = str(row[text_col])
    final_candidates_["prob_"+ str(prob_col)] = 1
    final_candidates_['line_num'] = row['line_num']
    final_candidates_["Label_Present"] = True
    final_candidates_['word_num'] = row['word_num']

    final_candidates_['left'] = row['line_left']
    final_candidates_['right'] = row['line_right']
    final_candidates_['conf'] = row['conf']
    final_candidates_['top'] = row['line_top']
    final_candidates_['bottom'] = row['line_down']

    final_candidates_['image_height'] = row['image_height']
    final_candidates_['image_widht'] = row['image_widht']

    final_candidates_["label_confidence"] = None
    final_candidates_["wordshape"] = None
    final_candidates_["wordshape_confidence"] = None
    final_candidates_["Odds"] = None
    final_candidates_['model_confidence'] = 1

    final_candidates_['final_confidence_score'] = 1
    final_candidates_['vendor_masterdata_present'] = False
    final_candidates_['extracted_from_masterdata'] = False      
    #final_candidate[candidate] = final_candidates_
    return final_candidates_

##>>>>>>>>>>>>>>> getting Billing GSTIN
def get_billingGSTIN(df, prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        for i, r in df.iterrows():
            pageNo = r["page_num"]
            temp = df[df["page_num"]== pageNo]
            start = 0
            current_add_region = 0
            candidate = None
            s2_label_left_bx = None
            temp = temp.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True])
            breaker = None
            for idx, row in temp.iterrows():
                if row["contains_ship_to_name"] == 1:
                    s2_label_left_bx = row["line_left"]
                if row["contains_bill_to_name"] == 1:
                    start = 1
                    billingName = row["line_text"]
                    b2_label_left_bx = row["line_left"]
                    current_add_region = int(row["region"])
                    #print("current_add_region :",current_add_region)

                if start == 1 &  current_add_region == int(row["region"]) & int(row["is_gstin_format"]) == 1:
                    print(" inside 1 cond")
                    if s2_label_left_bx is not None:
                        if row['left'] < s2_label_left_bx:
                            print("gstin picked with less than ship to cor..")
                            candidate = final_candidates(row,text_col ="text",prob_col = "billingGSTIN")                
                            breaker = True
                            break
                elif start == 1 & int(row["is_gstin_format"]) == 1:
                    print("inside 2 cond")
                    candidate = final_candidates(row,text_col ="text",prob_col = "billingGSTIN")                
                    breaker = True
                    break
            if breaker == True:
                break
        final_candidate["billingGSTIN_test"] = candidate
        prediction = {**prediction, **final_candidate }
        return prediction
    except:
        print(" billing Gstin exception :",traceback.print_exc()) 
        return prediction 

# >>>>>>>>>>>>>> getting shipping GSTIN
def get_shippingGSTIN(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        for i, r in df.iterrows():
            pageNo = r["page_num"]
            temp = df[df["page_num"]== pageNo]
            temp = temp.sort_values(['page_num','line_num','word_num'], ascending=[True,True,True])
            start = 0
            current_add_region = 0
            final_candidate = {}
            candidate = None
            breaker = False
            ship2_left_bx = None
            for idx, row in temp.iterrows():
                if row["is_bill_to_name"] == 1:
                    bill2_left_bx = row["line_left"]
                if row["contains_ship_to_name"] == 1:
                    start = 1
                    current_add_region = int(row["region"])
                    line_left_bounding_box = row["line_left"]
                    #print("current_add_region :",current_add_region)
                if start == 1 and current_add_region == int(row["region"]) and int(row["is_gstin_format"]) == 1:
                    print(" Inside 2 if")
                    if row["left"] >= line_left_bounding_box:
                        line_left_bx = row["left"]
                        print("shippingGSTIN :", row["text"], "shipping lable left_bx :",line_left_bounding_box, "GSTIN_line_left_bx :",line_left_bx, "current_add_region :",current_add_region," GSTIN Region",row["region"])            
                        candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                        breaker = True
                        break
                elif start == 1 & int(row["is_gstin_format"]) == 1:
                    if row["left"] >= line_left_bounding_box:
                        line_left_bx = row["left"]
                        print("line_left_bounding_box :",line_left_bounding_box, row["left"])              
                        print("shippingGSTIN :", row["text"], "shipping lable left_bx :",line_left_bounding_box, "GSTIN_line_left_bx :",line_left_bx, "current_add_region :",current_add_region," GSTIN Region",row["region"])            
                        candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                        breaker = True
                        break
                    else: 
                        if ship2_left_bx is not None:
                            if row["left"]>= bill2_left_bx:
                                print('row["is_gstin_format"]',row["is_gstin_format"])
                                candidate = final_candidates(row,text_col ="text",prob_col = "shippingGSTIN")                
                                breaker = True
                                break
                        
            if breaker:
                break
        final_candidate["shippingGSTIN_test"] = candidate
        return {**prediction, **final_candidate}
    except: 
        print(traceback.print_exc())
        return prediction

# >>>>>>>>>>>>> getting shipping Name
def get_shippingName(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        candidate = None
        for idx, row in df.iterrows():
            if row["is_ship_to_name"] == 1:
                candidate = bill2shipName_final_candidates(row,text_col ="line_text",prob_col = "shippingName")                
                break 
        final_candidate["shippingName_test"] = candidate
        return {**prediction, **final_candidate}
    except: return prediction 

# >>>>>>>>> getting billing Name
def get_billingName(df,prediction ):
    try:
        df = df[df["contains_bill2ship2_feature"] == 1]
        final_candidate = {}
        candidate = None
        for idx, row in df.iterrows():
            if row["is_bill_to_name"] == 1:
                candidate = bill2shipName_final_candidates(row,text_col ="line_text",prob_col = "billingName")                
                break 
        final_candidate["billingName_test"] = candidate
        return {**prediction, **final_candidate }
    except: return prediction

# >>> calling bill/ship to name, gistin 
def getBill2Shop2Details(df, prediction):
    prediction = get_billingName(df,prediction)
    prediction = get_shippingName(df,prediction)
    prediction = get_billingGSTIN(df,prediction)
    prediction = get_shippingGSTIN(df,prediction)
    return prediction
# added spacially for BCP demo
def supress_fields(prediction):
    pred = copy.deepcopy(prediction)
    try:
        pred_keys = pred.keys()
        not_required_fields = ["poNumber","paymentTerms","shippingName","billingName","shippingAddress",
                                "billingAddress","freightAmount","CGSTAmount","SGSTAmount","IGSTAmount",
                                "vendorGSTIN","shippingGSTIN","billingGSTIN","subTotal"]
        for item in not_required_fields :
            for k in pred_keys:
                if item == k:
                    del prediction[item]
        print("supressed fields")
        return prediction
    except:
        print("supress_fields exception",traceback.print_exc())
        return pred

# validated GST fields from  master data

def field_val_from_prediction(fieldName,prediction):
    # print("filedName :",fieldName, prediction.get(fieldName))
    if prediction.get(fieldName):
        fieldName = prediction.get(fieldName).get("text")
        fieldName = putil.correct_gstin(fieldName)
        return fieldName
    else:
        # print("field Not in prediction")
        fieldName = None
        return fieldName

def get_GSTIN_fields(DF, prediction, ADDRESS_MASTERDATA,VENDOR_MASTERDATA):
    pred_copy = copy.deepcopy(prediction)
    try:
        print("actual df shape :",DF.shape)
        F_DF = DF[DF["is_gstin_format"]==1]
        DF = F_DF[F_DF["page_num"] == 0]
        print("First page df shape :",DF.shape)
        if DF.shape[0] ==0 or DF.shape[0] is None:
            DF = F_DF[F_DF["page_num"] == 1]
            print("Second page df shape :",DF.shape)
        print("page df shape :",DF.shape)

        unique_gstin = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))
        print("total unique GSTIN : ", len(unique_gstin),"\t:",unique_gstin)
        label_frequency = DF["predict_label"].value_counts().to_dict()
        lbl_vendorGSTIN = label_frequency.get("vendorGSTIN")
        lbl_billingGSTIN = label_frequency.get("billingGSTIN")
        lbl_shipingGSTIN = label_frequency.get("shippingGSTIN")
        print("lbl_vendorGSTIN :",lbl_vendorGSTIN)
        print("lbl_billingGSTIN :",lbl_billingGSTIN)
        print("lbl_shipingGSTIN :",lbl_shipingGSTIN)

        print("Frequency of predicted labels :",type(label_frequency), label_frequency)
        # print("prediction :",prediction.keys())
        total_gstin = DF["is_gstin_format"].sum()
        print("total_gstin :",total_gstin, "\tunique_gstin :",len(unique_gstin))
        print("DF Shape after filtering data :",DF.shape)
      
        B_Assigned = None
        S_Assigned = None
        V_Assigned = None
        # gstin_matched_with_master_data = {}
        for idx ,row in DF.iterrows():
            GSTIN = putil.correct_gstin(row["text"])
            row["text"] = GSTIN
            print("\n\nGetting Prediction For :",row["text"])
            print("B_Assigned :",B_Assigned,"S_Assigned :",S_Assigned,"V_Assigned :",V_Assigned)
            vendorGSTIN = field_val_from_prediction("vendorGSTIN",prediction)
            print("predicted vendorGSTIN :",vendorGSTIN)
            billingGSTIN = field_val_from_prediction("billingGSTIN",prediction)
            print("predicted billingGSTIN :",billingGSTIN)
            shippingGSTIN = field_val_from_prediction("shippingGSTIN",prediction)
            print("predicted shippingGSTIN :",shippingGSTIN)
            predict_label = row["predict_label"]
            print("GSTIN :",GSTIN, "\tPredicted label :",predict_label)
            # print("row : ",type(row))
            # Matching GSTIN with buyers address master data
            GSTIN_FROM_ADDRESS_MASTERDATA = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == GSTIN]

            # Matching GSSTIN with Vendor master data 
            GSTIN_FROM_VENDOR_MASTERDATA = VENDOR_MASTERDATA[VENDOR_MASTERDATA['VENDOR_GSTIN']==GSTIN]
            print("Num of Records found in Vendor address data :\t",GSTIN_FROM_VENDOR_MASTERDATA.shape[0])
            print("Num of Records found in buyers address data :\t",GSTIN_FROM_ADDRESS_MASTERDATA.shape[0])
            if (lbl_vendorGSTIN is not None) and (lbl_vendorGSTIN > 1):
                # print("inside vendorGSTIN > 1")
                if GSTIN_FROM_ADDRESS_MASTERDATA.shape[0]>0: # and GSTIN_FROM_VENDOR_MASTERDATA.shape[0]<1:
                    if (shippingGSTIN is None and total_gstin > 2):
                        if S_Assigned == None:
                            S_Assigned = GSTIN
                            print("assign shipping GSTIN:", GSTIN)
                            prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                            continue
                    if(billingGSTIN is None):
                        if B_Assigned == None:
                            B_Assigned = GSTIN
                            print("assign billing GSTIN:", GSTIN)
                            prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                            continue
                if ((GSTIN_FROM_ADDRESS_MASTERDATA.shape[0] < 1) and (GSTIN_FROM_VENDOR_MASTERDATA.shape[0] > 0)):
                    print("inside match vendor data")
                    if V_Assigned is None:
                        V_Assigned = GSTIN
                        print("assigning Vendor GSTIN")
                        prediction.update(add_new_fields("vendorGSTIN",row,from_entity=True))
                        continue

            # print("finding match into master data")
            if (GSTIN_FROM_ADDRESS_MASTERDATA.shape[0]>0):
                # GSTIN_Matched_In_BuyesData = GSTIN_FROM_ADDRESS_MASTERDATA.iloc[0].to_dict()
                print("inside buyers master data")
                if vendorGSTIN == GSTIN:
                    print("Entity GSTIN predicted as vendor removing GSTIN prediction")
                    prediction["vendorGSTIN"] = None
                    prediction["vendorName"] = None
                if predict_label == "shippingGSTIN":
                    if S_Assigned == None:
                        S_Assigned = GSTIN
                        print("shipping GSTIN assigned")
                        prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                        continue
                    else:
                        print("already assinged as shipping:",GSTIN)
                        if B_Assigned is None:
                            B_Assigned = GSTIN
                            print("Assigned shipping GSTIN")
                            prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                            continue
                if predict_label == "billingGSTIN":
                    if B_Assigned == None:
                        B_Assigned = GSTIN
                        print("Assigned billing GSTIN")
                        prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                        continue
                    else:
                        print("already assinged as billing:",GSTIN)
                        if S_Assigned is  None:
                            S_Assigned = GSTIN
                            print("Assigned shipping GSTIN")
                            prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                            continue
                if (predict_label not in ["vendorGSTIN","shippingGSTIN","billingGSTIN"]):
                    if total_gstin > 2 : #and len(unique_gstin) ==2: 
                        if (billingGSTIN is None) and (shippingGSTIN is not None):
                            if B_Assigned == None:
                                B_Assigned = GSTIN
                                print("Assigned billing GSTIN")
                                prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                                continue
                        if (billingGSTIN is not None) and (shippingGSTIN is None):
                            if S_Assigned == None:
                                S_Assigned = True
                                print("Assigned shipping GSTIN")
                                prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                                continue
                        if (billingGSTIN is None) and (shippingGSTIN is None):
                            print("Both are unknown or VendorGSTIN")
                            if (B_Assigned == None):
                                B_Assigned = GSTIN
                                print("BillingGSTIN assigned", GSTIN)
                                prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                                continue
                            else:
                                if S_Assigned == None:
                                    S_Assigned =True
                                    print("ShippingGSTIN assigned", GSTIN)
                                    prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                                    continue
                if B_Assigned is None and billingGSTIN is None:
                    B_Assigned = GSTIN
                    print("BillingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                    continue
                if (S_Assigned is None) and (B_Assigned is not None):
                    S_Assigned = GSTIN
                    print("ShippingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                    continue
                if (B_Assigned is None) and (S_Assigned is not None):
                    B_Assigned = GSTIN
                    print("BillingGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                    continue

            if GSTIN_FROM_VENDOR_MASTERDATA.shape[0]>0:            
                if billingGSTIN == GSTIN:
                    print("Entity GSTIN predicted as billing removing GSTIN prediction")
                    prediction["billingGSTIN"] = None
                if shippingGSTIN == GSTIN:
                    print("Entity GSTIN predicted as shipping removing GSTIN prediction")
                    prediction["shippingGSTIN"] = None
                GSTIN_Matched_in_vendorData = GSTIN_FROM_VENDOR_MASTERDATA.iloc[0].to_dict()
                print("Matched in Vendor Address masterdata :", GSTIN_Matched_in_vendorData)
                if V_Assigned == None:
                    V_Assigned = GSTIN
                    print("vendorGSTIN assigned", GSTIN)
                    prediction.update(add_new_fields("vendorGSTIN",row,from_entity=True))
                    continue
                print("Vendor GSTIN already assigned :",V_Assigned,GSTIN)
            if (GSTIN_FROM_VENDOR_MASTERDATA.shape[0]<1) and (GSTIN_FROM_ADDRESS_MASTERDATA.shape[0] < 1):
                print("GSTIN not there in Vendor and Address Master data")
                # if (predict_label not in ["vendorGSTIN","shippingGSTIN","billingGSTIN"]):
                #     print("inside matching based on two GSTIN assinging third one") 
                #     print("vendorGSTIN :",vendorGSTIN, "\tshippingGSTIN :",shippingGSTIN,"\tbillingGSTIN :",billingGSTIN)
                #     if total_gstin > 2:
                #         if (billingGSTIN is not None) and (shippingGSTIN is not None):
                #             if (GSTIN != billingGSTIN) and (GSTIN != shippingGSTIN):
                #                 prediction.update(add_new_fields("vendorGSTIN",row,from_Vendor=True))
                #                 print("based on bill2ship2 GSTIN assinging third one vendor gstin")
                #                 return prediction
                #         if (billingGSTIN is not None) and (vendorGSTIN is not None):
                #             if (billingGSTIN != GSTIN ) and (vendorGSTIN != GSTIN):
                #                 prediction.update(add_new_fields("shippingGSTIN",row,from_entity=True))
                #                 print("based on 2 GSTIN assinging third one")
                #                 return prediction                    
                #         if (vendorGSTIN is not None) and (shippingGSTIN is not None):
                #             if (vendorGSTIN != GSTIN) and (shippingGSTIN != GSTIN):
                #                 prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))
                #                 print("based on 2 GSTIN assinging third one")
                #                 return prediction
                #     if total_gstin == 2:
                #         if ((vendorGSTIN is not None) and (billingGSTIN is None) and (shippingGSTIN is None)):
                #             prediction.update(add_new_fields("billingGSTIN",row,from_entity=True))

                #     print("Not matched codition inside based on 2 GSTIN assinging third one")

            print("Moving to the next iteration\n")
        #print("Prediction b4r update :",pred_copy)
        return prediction
    except:
        print("Get GSTIN Numbers Exception :",traceback.print_exc())
        return pred_copy

def add_new_fields(field_name,row = None,from_Vendor=False,from_entity = False):
    """
    """
    if (row is not None):
        final_candidates_ = {}

        extracted_value = str(row["text"])
        final_candidates_['line_num'] = row["line_num"]
        final_candidates_["prob_"+field_name] = 1
        final_candidates_['text'] = extracted_value
        final_candidates_["Label_Present"] = True
        final_candidates_['word_num'] = 0

        final_candidates_['left'] = row["left"]
        final_candidates_['right'] = row["right"]
        final_candidates_['conf'] = row["conf"]
        final_candidates_['top'] = row["top"]
        final_candidates_['bottom'] = row["bottom"]

        final_candidates_['page_num'] = row["page_num"]
        final_candidates_['image_height'] = row["image_height"]
        final_candidates_['image_widht'] = row["image_widht"]

        final_candidates_["label_confidence"] = None
        final_candidates_["wordshape"] = None
        final_candidates_["wordshape_confidence"] = None
        final_candidates_["Odds"] = None
        final_candidates_['model_confidence'] = 1

        final_candidates_['final_confidence_score'] = 1
        final_candidates_['vendor_masterdata_present'] = True
        final_candidates_['extracted_from_masterdata'] = from_Vendor
        final_candidates_['extracted_from_entitydata'] = from_entity

        return {field_name: final_candidates_}

def get_vendor_buyers_name(DF,prediction,ADDRESS_MASTERDATA,VENDOR_MASTERDATA):
    vendorGSTIN = None
    billingGSTIN = None
    shippingGSTIN = None
    vendorName = None
    billingName = None
    shippingName = None
    # clean_GSTIN = ['/',':','(',')','.',"'",","]        
    if prediction.get("vendorGSTIN"):
        vendorGSTIN = prediction.get("vendorGSTIN").get("text")
        vendorGSTIN = putil.correct_gstin(vendorGSTIN)
        # print("vendorGSTIN : ",vendorGSTIN)
    if prediction.get("billingGSTIN"):
        billingGSTIN = prediction.get("billingGSTIN").get("text")
        billingGSTIN = putil.correct_gstin(billingGSTIN) 
        # print("billingGSTIN : ",billingGSTIN)
    if prediction.get("shippingGSTIN"):
        shippingGSTIN = prediction.get("shippingGSTIN").get("text")
        shippingGSTIN = putil.correct_gstin(shippingGSTIN)
        # print("shippingGSTIN : ",shippingGSTIN)
    if prediction.get("vendorName"):
        vendorName = prediction.get("vendorName").get("text")
    if prediction.get("billingName"):
        billingName = prediction.get("billingName").get("text")
    if prediction.get("shippingName"):
        shippingName = prediction.get("shippingName").get("text")

    DF = DF[DF["is_company_name"]==1]
    C_Names = DF["line_text"].unique()
    C_Names = [x.upper() for x in C_Names]
    print("Company names :",C_Names)

    if billingName is None:
        print("billingName is None")
        if billingGSTIN is not None:
            # Matching GSTIN with buyers address master data
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == billingGSTIN]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = row["NAME"].upper()
                    print("Matched Name :", row["NAME"])
                    if B_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name Billing name from address master data",row["NAME"])
                                prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in buyes data")
                    else:
                        prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))                       
            else:
                print("billing GSTIN Match not found")
        else:
            print("billing GSTIN is None :",billingGSTIN)
    else:
        print("billingName is Not None")
        if billingGSTIN is not None:
            print("billing GSTIN Not None")
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == billingGSTIN]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = str(row["NAME"]).upper()
                    print("GSTIN Matched Names :",row["NAME"])
                    if B_Name_frm_buyersData.shape[0] > 1:
                        print("Partial billing Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name",row["NAME"])
                                prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(billingName).upper() in row["NAME"]:
                            print("updating partial matched Name",row["NAME"])
                            prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("Updating Billing Name by matching GSTIN ",row["NAME"])
                            prediction.update(add_new_field("billingName",row["NAME"],from_Vendor=True,from_entity=True))
            else:
                print("No Match data found")
        else:
            print("billing GSTIN is None :",billingGSTIN)           
    
    if shippingName is None:
        if shippingGSTIN is not None:
            # Matching GSTIN with buyers address master data
            B_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == shippingGSTIN]
            print("Match DF shape :",B_Name_frm_buyersData.shape[0])
            if B_Name_frm_buyersData.shape[0] > 0:
                for idx, row in B_Name_frm_buyersData.iterrows():
                    row["NAME"] = str(row["NAME"]).upper()
                    if B_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name :",row["NAME"])
                                prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in buyes data")
                    else:
                        print("Updating shipping Name with extract GSTIN match :",row["NAME"])
                        prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
            else:
                print("No Match found")
        else:
            print("Shipping GSTIN is None :",shippingGSTIN)
    else:
        print("shippingName is Not None")
        if shippingGSTIN is not None:
            print("shipping GSTIN Not None")
            S_Name_frm_buyersData = ADDRESS_MASTERDATA[ADDRESS_MASTERDATA["GSTIN"] == shippingGSTIN]
            print("Match DF shape :",S_Name_frm_buyersData.shape[0])
            if S_Name_frm_buyersData.shape[0] > 0:
                for idx, row in S_Name_frm_buyersData.iterrows():
                    row["NAME"] = str(row["NAME"]).upper()
                    print("GSTIN Matched Names :",row["NAME"])
                    if S_Name_frm_buyersData.shape[0] > 1:
                        print("Partial vendor Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["NAME"]:
                                print("updating Matched Name",row["NAME"])
                                prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(shippingName).upper() in row["NAME"]:
                            print("updating partial matched Name",row["NAME"])
                            prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("updating shipping Name by exact GSTIN match",row["NAME"])
                            prediction.update(add_new_field("shippingName",row["NAME"],from_Vendor=True,from_entity=True))                            
            else:
                print("No Match data found")
        else:
            print("shipping GSTIN is None :",shippingGSTIN)           

    if vendorName is None:
        print("vendorName is None")
        if vendorGSTIN is not None:
            # Matching GSTIN with buyers address master data
            V_Name_frm_buyersData = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"] == vendorGSTIN]
            print("Match DF shape :",V_Name_frm_buyersData.shape[0])
            if V_Name_frm_buyersData.shape[0] > 0:
                for idx, row in V_Name_frm_buyersData.iterrows():
                    row["VENDOR_NAME"] = str(row["VENDOR_NAME"]).upper()
                    print("GSTIN Matched Names :",row["VENDOR_NAME"])
                    if V_Name_frm_buyersData.shape[0] > 1:
                        for item in C_Names:
                            # print("items :",item)
                            if item in row["VENDOR_NAME"]:
                                print("updating Matched Name",row["VENDOR_NAME"])
                                prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity = True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        print("Updating vendor Name with extract GSTIN match :",row["VENDOR_NAME"])
                        prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))                        
            else:
                print("No Match data found")
        else:
            print("vendorName GSTIN is None :",vendorGSTIN)           
    else:
        print("VendorName is Not None")
        if vendorGSTIN is not None:
            print("Vendor GSTIN Not None")
            V_Name_frm_buyersData = VENDOR_MASTERDATA[VENDOR_MASTERDATA["VENDOR_GSTIN"] == vendorGSTIN]
            print("Match DF shape :",V_Name_frm_buyersData.shape[0])
            if V_Name_frm_buyersData.shape[0] > 0:
                for idx, row in V_Name_frm_buyersData.iterrows():
                    row["VENDOR_NAME"] = str(row["VENDOR_NAME"]).upper()
                    print("GSTIN Matched Names :",row["VENDOR_NAME"])
                    if V_Name_frm_buyersData.shape[0] > 1:
                        print("Partial vendor Name Not matche so matching with the all the invoice company name")
                        for item in C_Names:
                            print("items :",item)
                            if item in row["VENDOR_NAME"]:
                                print("updating Matched Name",row["VENDOR_NAME"])
                                prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))
                            else:
                                print("Match not foud in vendor data")
                    else:
                        if str(vendorName).upper() in row["VENDOR_NAME"]:
                            print("updating partial matched Name",row["VENDOR_NAME"])
                            prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))
                        else:
                            print("updating vendor Name by exact GSTIN match",row["VENDOR_NAME"])
                            prediction.update(add_new_field("vendorName",row["VENDOR_NAME"],from_Vendor=True,from_entity=True))                            
            else:
                print("No Match data found")
        else:
            print("vendorName GSTIN is None :",vendorGSTIN)           

    return prediction

# Removing 
def clean_GSTIN(prediction):
    for key, val in prediction.items():
        if key in ["vendorGSTIN","billingGSTIN","shippingGSTIN"]:
            if val:
                if val["text"]!='':
                    print("GSTINS befor cleaning :",val["text"])
                    val["text"] = putil.correct_gstin(val["text"])
                    prediction[key] = val
                    print("GSTIN clenned",val["text"])
    return prediction

# extracting VendorPAN from vendor GSTIN
def extract_vendorPAN(prediction):
    vendor_gstin = prediction.get("vendorGSTIN")
    if vendor_gstin:
        gstin = vendor_gstin["text"]
        if gstin != '':
            if len(gstin) == 15:
                print("getting PAN from GSTIN :",gstin)
                vendorPAN = gstin[2:12]
                print("vendorPAN",vendorPAN)
                VendorPAN = add_new_field("vendorPAN",vendorPAN,from_Vendor=True)
                print("VendorPAN :",VendorPAN)
                prediction.update(VendorPAN)
                print("vendorPAN",prediction.get("vendorPAN"))
            print("Incorrect GSTIN",gstin)
        else: print("Empty GSTIN field Reccived")
    return prediction

# adding Mandatory fields flag from stg config into field result
def adding_mandatory_fieldFlag(prediction):
    pred_copy = copy.deepcopy(prediction)
    try:
        STP_CONFIGURATION = putil.getSTPConfiguration()
        STP_CONFIGURATION = STP_CONFIGURATION.get("DEFAULT")
        display_fields = [key for key,val in STP_CONFIGURATION.items() if (val["display_flag"] == 1)]
        # print("rcvd pred :",prediction)
        for key, val in pred_copy.items():
            if (key in display_fields) and (val is not None):   
                # print("key :",key, "\tval :",val)
                mandatory_field = STP_CONFIGURATION.get(key).get("mandatory")
                val.update({"mandatory":mandatory_field})
                prediction[key] = val
        # print("prediction after adding mdtr :",prediction)
        return prediction
    except:
        print("adding_mandatory_fieldFlag exception :",adding_mandatory_fieldFlag)
        return pred_copy

## set prediction  into custom order
def get_sequence_list():
    stg_config = putil.getSTPConfiguration()   
    stg_config = stg_config.get("DEFAULT")
    # print("config data :",stg_config)
    sequence_list = {}
    for k ,v in stg_config.items():
        if (v.get("display_flag")==1) and (v.get("order")):
            sequence_list[v["order"]] = k
    sequence_list = sorted(sequence_list.items())
    sequence_list = [x[1] for x in sequence_list]

    return sequence_list

def custom_sorting_prediction(prediction):
    sequence_list = get_sequence_list()
    # print("prediction oredered keys :",sequence_list)
    not_in_sequence = []
    sorted_prediction = {}
    try:
        pred_keys = list(prediction.keys())
        for i in pred_keys:
            if i not in sequence_list:
                not_in_sequence.append(i)
        # print("keys not in sequence list :",not_in_sequence)
        for item in sequence_list:
            val = prediction.get(item)
            sorted_prediction.update({item:val})
        for item_ in not_in_sequence:
            val_ = prediction.get(item_)
            sorted_prediction.update({item_:val_})
            
        return sorted_prediction
    except:
        print("Sorting prediction  exception :",traceback.print_exc())
        return prediction

def calculate_total(DF, prediction)-> dict:
    
    import math
    pred_copy = copy.deepcopy(prediction) 
    try:
        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                        "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                field_values.update({f:float(prediction.get(f).get("text"))})
            else:
                field_values.update({f : None}) 

        '''    
        -> calculating copying subtotal as total if all taxes is None.
        -> subtracting discount if it is not None
        '''
        CGSTAmount = field_values.get("CGSTAmount")
        SGSTAmount = field_values.get("SGSTAmount")
        IGSTAmount = field_values.get("IGSTAmount")
        discountAmount = field_values.get("discountAmount")
        additionalCessAmount = field_values.get("additionalCessAmount")
        CessAmount = field_values.get("CessAmount")
        subTotal = field_values.get("subTotal")
        total = field_values.get("totalAmount")

        # for key, val in field_values.items():
        #     print(key," : ",val)
        noCgst = (CGSTAmount is None) or (CGSTAmount == 0.0)
        noSgst = (SGSTAmount is None) or (SGSTAmount == 0.0)
        noIgst = (IGSTAmount is None) or (IGSTAmount == 0.0)
        noDiscount = (discountAmount is None) or (discountAmount == 0.0)
        noAdCess = (additionalCessAmount is None) or (additionalCessAmount == 0.0)
        noCess = (CessAmount is None) or (CessAmount == 0.0)
        noTotal = (total is None) or (total == 0.0)
        noSubTotal = (subTotal is None) or (subTotal == 0.0)

        if noCgst and noSgst and noIgst and noDiscount and noAdCess and noCess:
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts
            df_filt = DF[(DF["line_row"] == 0) & (DF["extracted_amount"] > 0.0)]
            extracted_amounts = list(set(list(df_filt["extracted_amount"])))
            stp_check = False
            if len(extracted_amounts) <= 2:
                m1 = max(extracted_amounts)
                m2 = min(extracted_amounts)
                if m1 - m2 < 1:
                    stp_check = True
            elif len(extracted_amounts) > 2:
                extracted_amounts = sorted(extracted_amounts,
                                           reverse = True)
                first = extracted_amounts[0]
                second = extracted_amounts[1]
                if first - second < 1:
                    rem_amounts = extracted_amounts[2:]
                else:
                    rem_amounts = extracted_amounts[1:]
                sum_amounts = sum(rem_amounts)
                if abs(sum_amounts - first) < 1:
                    stp_check = True
            #Aug 02 2022 - code to allow STP if total = subtotal and no taxes are found in header amounts

            if noTotal:
                print("inside all taxes none")
                if subTotal:
                    if subTotal > 0.0:
                        print("sub total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))

                            #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            #Make other amount fields as 100%

                        else:
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                value = subTotal,
                                final_confidence_score = 0.4,
                                calculated = not stp_check))

            elif noSubTotal:
                if total:
                    if total > 0.0:
                        print("total is > 0")
                        if stp_check:
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 1.0,
                                calculated = not stp_check))

                            #Make other amount fields as 100%
                            prediction.update(add_new_field(
                                field_name = "totalAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "SGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "IGSTAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "CessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            prediction.update(add_new_field(
                                field_name = "additionalCessAmount",
                                final_confidence_score = 1.0,
                                calculated = not stp_check))
                            #Make other amount fields as 100%

                        else:
                            prediction.update(add_new_field(
                                field_name = "subTotal",
                                value = total,
                                final_confidence_score = 0.4,
                                calculated = not stp_check))
                            
            elif subTotal > 0.0 and total > 0.0 and math.isclose(total,
                                                                 subTotal,
                                                                 abs_tol = 1):
                prediction.update(add_new_field(
                    field_name = "totalAmount",
                    value = subTotal,
                    final_confidence_score = 0.4,
                    calculated = not stp_check))
                

        return prediction

    except :
        print("Calculate total exception \n",
              traceback.print_exc())
        return pred_copy

def calculate_total2(DF, prediction)-> dict:

    pred_copy = copy.deepcopy(prediction) 
    try:
        '''
        -> calculating total by adding taxes and subtracting discount
        -> validating calculated total with first and sencond max amount 
        '''

        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                  "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                try:
                    field_values.update({f:float(prediction.get(f).get("text"))})
                except :
                    field_values.update({f : 0})
            else:
                field_values.update({f : 0}) 

        discountAmount = field_values.get("discountAmount")
        total = field_values.get("totalAmount")

        for key, val in field_values.items():
            print(key," : ",val)

        total = field_values.get("totalAmount")
        fields.remove("discountAmount")
        print("Addition fields :",fields) 
        add_fields_sum = 0
        for key, val in field_values.items():
            print("prediction field ",key," : ",val)
            if key != "totalAmount":
                add_fields_sum =  add_fields_sum + val
        print("add_fields_sum :",add_fields_sum, "DiscountAmount :",discountAmount)
        if (discountAmount >0):       
            total_cal = add_fields_sum - discountAmount
        else:
            total_cal = add_fields_sum
        print("Total calculated amount :",total_cal, "\tTotalAmount :",total)
        if abs(total- total_cal) >2:
            first_max_amt = DF[DF["is_amount"]==1]
            first_max_amt = first_max_amt[first_max_amt["line_row"]==0]
            amount_list = first_max_amt['text']
            final_amount_list = []
            for i in amount_list:
                try:
                    i = float(str(i).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                    final_amount_list.append(i)
                except:
                    print("flaot conversion error :",i)
                    pass
            final_amount_list = sorted(set(final_amount_list),reverse= True)
            if len(final_amount_list)>0:
                print("final_amount_list ;",final_amount_list,"\total_cal",total_cal)
                counter = 0
                for amount in final_amount_list:
                    print("abs with first max amount :",abs(amount - total_cal))
                    if abs(amount - total_cal) < 2:
                        prediction.update(add_new_field(field_name = "totalAmount",
                                                        final_confidence_score =1,
                                                        value = amount,calculated=False))
                        field_values.update({"totalGSTAmount":0})
                        for key, val in field_values.items():
                            if (val == 0):
                                prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                        break
                    if counter == 0:
                        break
                    counter = counter +1
        else:
            print("Absulate difference is grater than 2 :",abs(total- total_cal))             
        return prediction
    except :
        print("Calculate total exception \n",
              traceback.print_exc())
        return pred_copy

def check_if_future_Date(prediction:dict) -> dict:
    '''
    check if future date if it is then reduce confidence to 40 %
    '''
    pred_copy = copy.deepcopy(prediction)
    try:
        from datetime import datetime
        invdt = prediction.get("invoiceDate")
        if (invdt) and (invdt.get("text")!=''):
            formate_date = parser.parse(invdt.get("text"), dayfirst=True).date().strftime('%d/%m/%Y')
            print("orignal formate ;",invdt.get("text"),"\tFromate date",formate_date)
            if datetime.strptime(formate_date,'%d/%m/%Y') > datetime.now():
                print("future date reducing confidence")
                prediction = reduce_field_confidence(prediction,"invoiceDate")
                return prediction
        return pred_copy
    except:
        print("check future date exception ;",traceback.print_exc())
        return pred_copy

def check_left_above_invdt_ngbr(df,prediction):
    prediction_copy = prediction
    try:
        df1=df[(df.invdt_prediction == True)]
        df = df1[df1['page_num']==0]
        if df1.shape[0]==0:
            df = df1[df1['page_num']==1]
        inv_dt = prediction.get("invoiceDate")
        print("filter df :",df.shape)
        if inv_dt:
            print("invoiceDate values :",inv_dt)
            for row in df1.itertuples():
                row_date = parser.parse(row.text, dayfirst=True).date().strftime('%d/%m/%Y')
                print("converted date :",row_date, "bf4 con dt ;",row.text)
                if row_date == inv_dt.get('text'):
                    print("left ;",row.left,"\tright :",row.right,"\ttop ;",row.top,"\tbottom :",row.bottom)
                    print("above ngbr:",row.above_processed_ngbr,"left ngbr :",row.left_processed_ngbr)
                    check1=("invoice" in row.above_processed_ngbr.lower() or "inv" in row.above_processed_ngbr.lower())
                    check2=("invoice" in row.left_processed_ngbr.lower() or "inv" in row.left_processed_ngbr.lower())
                    print("check1 :",check1,"check2:",check2)
                    if not (check1) and not(check2):
                        print("invoice name not present in invoiceDate label. so reducing confidence")
                        prediction = reduce_field_confidence(prediction,"invoiceDate")
                        return prediction
                # else:
                #     print("else block")
                #     print("inv date :",row.text,"inve predicted dt",inv_dt.get('text'))
                #     # print("left ;",row.left,"\tright :",row.right,"\ttop ;",row.top,"\tbottom :",row.bottom)
                #     print("above ngbr:",row.above_processed_ngbr,"left ngbr :",row.left_processed_ngbr)

        return prediction
    except:
        print("check_left_above_invdt_ngbr :",traceback.print_exc())
        return prediction_copy


def copy_gstin(DF, prediction):
    pred_copy = copy.deepcopy(prediction)
    try:
        print("actual df shape :",DF.shape)
        DF = DF[DF["is_gstin_format"]==1]
        print("page df shape :",DF.shape)

        unique_gstin = list(set([putil.correct_gstin(s) for s in list(DF[DF["is_gstin_format"]==1]["text"].unique())]))
        print("total unique GSTIN : ", len(unique_gstin),"\t:",unique_gstin)
        if len(unique_gstin) == 2:
            billingGSTIN = prediction.get("billingGSTIN")
            shippingGSTIN = prediction.get("shippingGSTIN")
            if (shippingGSTIN is not None) and (billingGSTIN is None):
                if shippingGSTIN.get("text") != '':
                    prediction.update(add_new_field(field_name = "billingGSTIN",
                                        value = shippingGSTIN.get("text"),
                                        from_entity = shippingGSTIN.get("extracted_from_entitydata")))
                    print("Copied  Shipping GSTIN to Billing GSTIN")
            if (billingGSTIN is not None) and (shippingGSTIN is None):
                if billingGSTIN.get("text") != '':
                    prediction.update(add_new_field(field_name = "shippingGSTIN",
                                                    value = billingGSTIN.get("text"),
                                                    from_entity = billingGSTIN.get("extracted_from_entitydata")))
                    print("Copied Billing GSTIN to Shipping GSTIN")
        else:
            print("Total unique GSTIN less than two or more")
        return prediction
    except :
        print("Copy GSTIN exception",traceback.print_exc())
        return pred_copy

def validating_amount_fields_increasing_confidence(DF,prediction)-> dict:

    pred_copy = copy.deepcopy(prediction) 
    try:
        '''
        -> calculating total by adding taxes and subtracting discount
        -> validating calculated total with first and sencond max amount 
        '''

        fields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount",
                  "CessAmount","additionalCessAmount","discountAmount"]
        field_values = {}
        for f in fields:
            if (prediction.get(f)) and (prediction.get(f).get("text") != ''):
                try:
                    field_values.update({f:float(prediction.get(f).get("text"))})
                except :
                    field_values.update({f : 0})
            else:
                field_values.update({f : 0}) 
        GSTAmount = 0
        Cess = 0
        for key, val in field_values.items():
            print(key," : ",val)
            if val is not None:
                if key in ["CGSTAmount","SGSTAmount","IGSTAmount"]:
                    GSTAmount = GSTAmount + val
                    pass
                if key in ["CessAmount","additionalCessAmount"]:
                    Cess = Cess + val
                    pass
        
        discountAmount = field_values.get("discountAmount")
        total = field_values.get("totalAmount")
        subtotal = field_values.get("subTotal")
        calculatedSubTotal = total - (GSTAmount + Cess + discountAmount)
        print("calcualted subtotal :",calculatedSubTotal)
        if subtotal ==0:
            if DF[DF["second_max_amount"]==1].shape[0]>0:
                second_max_amt = DF[DF["second_max_amount"]==1] 
                print("inside subtotal check df :",second_max_amt.shape)
                second_max_amt = second_max_amt["text"].iloc[0]
                print("second_max_amt ;",second_max_amt,"calcualted subtotal :",calculatedSubTotal)
                try:
                    second_max_amt = float(str(second_max_amt).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                    print("abs with second max amount :",abs(second_max_amt -calculatedSubTotal))
                    if abs(second_max_amt -calculatedSubTotal) < 2:
                        prediction.update(add_new_field(field_name = "subTotal",final_confidence_score =1,value = second_max_amt,calculated=False))
                        for key, val in field_values.items():
                            if (val is not None):
                                prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                        prediction = reduce_field_confidence(prediction, "totalGSTAmount",model_confidence = 1,final_confidence_score = 1)
                    
                except:
                    print("flaot conversion error ")
                    pass

        calculatedTotal =  (GSTAmount + Cess + subtotal)-discountAmount
        abs_sub = abs(subtotal- calculatedSubTotal)
        abs_total = abs(total-calculatedTotal)
        if (abs_sub < 2) and (abs_total <2):
            for key, val in field_values.items():
                if val == 0.0:
                    # print("increasing field confidence :",key)
                    prediction = reduce_field_confidence(prediction, key,model_confidence = 1,final_confidence_score = 1)
                    prediction = reduce_field_confidence(prediction, "totalGSTAmount",model_confidence = 1,final_confidence_score = 1)
                
        return prediction
    except :
        print("validating_amount_fields_increasing_confidence :",traceback.print_exc())
        return pred_copy

def extract_missing_left_label_amount_field_from_table(df,subStngToMatch:str):
    """
    Extracting tax amounts only if there is no left label if values present in tabular form
    and if there is no extraction for thes fields
    """
    try:
        temp = df[df["is_amount"]==1]
        temp = temp[temp["line_row"]==0]
        # print("2161",temp.shape)
        # temp["keysMatch"] = temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)
        # temp = temp[temp["keysMatch"]==True]
        temp = temp[temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)==True]
        # print("2163",temp.shape)
        # print("shape of final candidates :",temp.shape)
        if not(temp.shape[0]):
            temp = df[df["is_amount"]==1]
            temp = temp[temp["above_processed_ngbr"].str.contains(subStngToMatch,case=False,regex=True)==True]
            # print("2170",temp.shape)
            # temp.to_csv(subStngToMatch+".csv")
            if not(temp.shape[0]):
                return None
        # temp = temp.drop_duplicates('text').sort_index().sort_values(by=["text"],ascending = True)
        amt_lst = []
        for item in list(set(temp["text"])):
            if str(item).replace(',','').replace(':','').replace('.','').replace(u'\u20B9','').isdigit():
                x = float(str(item).replace(',','').replace(':','').replace('/','').replace(u'\u20B9',''))
                amt_lst.append(x)
        amt_lst = sorted(amt_lst,reverse=True)
        print("amt_lst :",amt_lst)
        if len(amt_lst)>0:
            return amt_lst[0]
        return None
    except :
        print(" exception extract_missing_left_label_amount_field_from_table",traceback.print_exc())
        return None

def validating_extracted_amount_fields_without_left_label(prediction:dict,df)->dict:
    pred_copy = copy.deepcopy(prediction)
    try:
        print("validating_extracted_amount_fields_without_left_label")
        is_CGST_SGST = df["is_CGST_SGST"][0]
        is_IGST = df["is_IGST"][0]
        # print("is_CGST_SGST :",is_CGST_SGST)
        total =  prediction.get("totalAmount")
        if total and total.get("text")!= '':
            total = float(total.get("text"))
        subtotal =  prediction.get("subTotal")
        if subtotal and subtotal.get("text")!= '':
            subtotal = float(subtotal.get("text"))
        cgst = prediction.get("CGSTAmount")
        sgst = prediction.get("SGSTAmount")
        igst = prediction.get("IGSTAmount")
        cess = prediction.get("CessAmount")
        if cess and cess.get("text")!= '':
            cess = float(cess.get("text"))
        addCess = prediction.get("additionalCessAmount")
        if addCess and addCess.get("text")!= '':
            addCess = float(addCess.get("text"))
        print("total :",total,"/subtotal :",subtotal)
        print("befor extracting")
        print("cgst :",cgst,"\nsgst :",sgst,"\nigst :",igst,"\ncess :",cess,"\naddCess :",addCess)
        CSGT_subStngToMatch = 'CGST AMT|CGST'
        SGST_subStngToMatch = 'SGST AMT|SGST'
        IGST_subStngToMatch = "IGST AMT|IGST"
        subtotal_subStngToMatch = "TAXABLE"

        if not(cgst) and not(sgst)and not(igst):
            cgst = extract_missing_left_label_amount_field_from_table(df,CSGT_subStngToMatch)
            sgst = extract_missing_left_label_amount_field_from_table(df,SGST_subStngToMatch)
            igst = extract_missing_left_label_amount_field_from_table(df,IGST_subStngToMatch)
            if not (subtotal):
                subtotal_ext = extract_missing_left_label_amount_field_from_table(df,subtotal_subStngToMatch)
            else :
                subtotal_ext = extract_missing_left_label_amount_field_from_table(df,subtotal_subStngToMatch)
                print("subtotal_ext:",subtotal_ext)

            print("total :",total,"\nsubtotal :",subtotal)
            print("affter extracting")
            print("cgst :",cgst,"\nsgst :",sgst,"\nigst :",igst)

            add_sum = 0
            if total:
                if (is_CGST_SGST ==1)and (is_IGST==0):
                    if cgst and sgst:
                        if subtotal and (abs(total-(cgst+sgst+subtotal))<2):
                            prediction.update(add_new_field(field_name = "CGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "SGSTAmount",value = cgst))
                            print("matched CGST SGST extracted")
                        elif subtotal_ext and (abs(total-(cgst+sgst+subtotal_ext))<2):
                            prediction.update(add_new_field(field_name = "CGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "SGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "subTotal",value = subtotal_ext))
                            print("Extract new subtotal matched")
                        else:
                            print("Difference is greater than 2 : total ",total,":subtotal ",subtotal,": subtotal_ext ",subtotal_ext)
                if (is_CGST_SGST ==0)and (is_IGST==1):
                    if igst:
                        if subtotal and (abs(total-(igst+subtotal))<2):
                            prediction.update(add_new_field(field_name = "IGSTAmount",value = cgst))
                            print("match igst extracted")
                        elif subtotal_ext and (abs(total - (igst+subtotal_ext))):
                            prediction.update(add_new_field(field_name = "IGSTAmount",value = cgst))
                            prediction.update(add_new_field(field_name = "subTotal",value = subtotal_ext))
                            print("matched with calculated subtotal")
                        else:
                            print("Difference is greater than 2 :",total,add_sum)

        return prediction
    except :
        print("validating_extracted_amount_fields_without_left_label :",traceback.print_exc())
        return pred_copy
        

def apply_client_rules(DF, prediction, docMetaData,ADDRESS_MASTERDATA,VENDOR_MASTERDATA,format_= None):
    print("Started applying  client rules")
    # prediction = discard_lines_without_mandatory_fields(prediction)
    prediction = discard_additional_LI_rows(prediction)
    prediction = demo_change(prediction)
    #check_preform = extract_preform(DF)
    #if check_preform is not None:
     #   prediction = {**prediction, **check_preform}
    #out_dict = extract_barcode(DF)
    #if out_dict is not None:
     #   prediction = {**prediction, **out_dict}
    #prediction = clean_HSNCode(prediction)
    #prediction = clean_PONumber(prediction)
    prediction = convert_dates(prediction)
    # prediction = make_vendor_info_editable(prediction)
    # prediction = get_billingName(DF,prediction)
    # prediction = get_shippingName(DF,prediction)
    # prediction = get_billingGSTIN(DF,prediction)
    # prediction = get_shippingGSTIN(DF,prediction)
    #prediction = build_final_QRCode_json(prediction,docMetaData)
    # if qr_pred is not None:
    #      prediction = qr_pred
    #print("extractedQRCodeData",prediction)
    #prediction = validate_Model_Prediction_with_QRCode_Data(docMetaData,prediction)
    #extracted_org = extract_org(DF)
    #prediction = {**prediction, **extracted_org}
    # prediction = getBill2Shop2Details(DF, prediction)
    # print("predcition before gsting extraction :",prediction)
    prediction = get_GSTIN_fields(DF, prediction, ADDRESS_MASTERDATA,VENDOR_MASTERDATA)
    prediction = copy_gstin(DF, prediction)
    prediction = get_vendor_buyers_name(DF,prediction,ADDRESS_MASTERDATA,VENDOR_MASTERDATA)
    prediction = extract_vendorPAN(prediction)
    # prediction = validating_extracted_amount_fields_without_left_label(prediction,DF)
    prediction = calculate_total(DF,prediction)
    prediction = calculate_total2(DF, prediction)
    prediction = check_if_future_Date(prediction)
    prediction = check_left_above_invdt_ngbr(DF,prediction)

    # calling confidence reduction fuction aftrer applying all rules
    prediction = reduction_confidence_taxes(prediction)
    prediction = reduce_amount_fields_confidenace(prediction)
    return prediction

def bizRuleValidateForUi(documentId,callBackUrl):

    def mandatoryMsg(fldId):
        return fldId + " cannot be blank"

    def invalidAmtMsg(fldId):
        return fldId + " is an amount field and must contain only numbers"

    def invalidDtMsg(fldId):
        return fldId + " is a date field and must contain a valid date format starting with day"

    def invalidFormat(fldId,format_):
        return "The format of " + fldId + " is " + format_ + ". Please ensure the value is in this format."


    try:
        result = []
        docResult = putil.getDocumentResultApi(documentId,
                                               callBackUrl)
        if docResult is None:
            return None
        extractedFields = docResult["result"]["document"]["documentInfo"]
        mandatoryFields = putil.getMandatoryFields()
        fldTypes = putil.getFldTypeFormat()
        # print("Field Types",fldTypes)
        if mandatoryFields is None:
            raise Exception
        flds = {}
        for idx,fld in enumerate(extractedFields):
            fldId = fld["fieldId"]
            fldVal = fld["correctedValue"] if "correctedValue" in fld.keys() else fld["fieldValue"]
            flds[fldId] = fldVal
            #Do Mandatory check
            if fldVal == "" and fldId in mandatoryFields:
                # return mandatoryMsg(fldId)
                result.append((fldId,mandatoryMsg(fldId)))
            #Do Alpha-numeric check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "alpha-numeric"]:
                if not putil.validAlphaNumeric(fldVal):
                    result.append((fldId,
                                   fldId + " is an alpha-numeric field and must contain at-least one alphabet or number"))
            #Do numeric check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "numeric"]:
                if not putil.validAmount(fldVal):
                    result.append((fldId,
                                    invalidAmtMsg(fldId)))
            #Do date check
            if fldId in [fld_[0] for fld_ in fldTypes if fld_[1].lower() == "date"]:
                if putil.validDate(fldVal) == 100:
                    result.append((fldId,
                                    invalidDtMsg(fldId)))
                elif putil.validDate(fldVal) == 200:
                    result.append((fldId,
                                   "Invoice Date cannot be a future date"))
            #Do format check
            fldFormat = "".join(["X" if c.isalpha() else ("0" if c.isnumeric() else c) for c in list(fldVal)])
            defFormat = [fld_[2] for fld_ in fldTypes if fld_[2] is not None and fld_[0] == fldId]
            # print("Format ",fldId,defFormat)
            if len(defFormat) == 1:
                defFormat_ = defFormat[0]
                if defFormat_ != "":
                    if not putil.checkValidFormat(defFormat_,fldFormat):
                        result.append((fldId,
                                       invalidFormat(fldId,
                                                     defFormat_)))

        #Functional rule
        vendorGSTIN = flds.get("vendorGSTIN")
        billingGSTIN = flds.get("billingGSTIN")
        IGSTAmount = flds.get("IGSTAmount")
        SGSTAmount = flds.get("SGSTAmount")
        CGSTAmount = flds.get("CGSTAmount")
        totalGSTAmount = flds.get("totalGSTAmount")
        discountAmount = flds.get("discountAmount")
        cessAmount = flds.get("CessAmount")
        additionalCessAmount = flds.get("additionalCessAmount")
        subTotal = flds.get("subTotal")
        totalAmount = flds.get("totalAmount")
        try:
            convIGSTAmount = float(IGSTAmount) if IGSTAmount is not None else 0.0
        except:
            convIGSTAmount = 0.0
        try:
            convSGSTAmount = float(SGSTAmount) if SGSTAmount is not None else 0.0
        except:
            convSGSTAmount = 0.0
        try:
            convCGSTAmount = float(CGSTAmount) if CGSTAmount is not None else 0.0
        except:
            convCGSTAmount = 0.0
        try:
            convTotalGST = float(totalGSTAmount) if totalGSTAmount is not None else 0.0
        except:
            convTotalGST = 0.0
        try:
            convSubTotal = float(subTotal) if subTotal is not None else 0.0
        except:
            convSubTotal = 0.0
        try:
            convTotalAmt = float(totalAmount) if totalAmount is not None else 0.0
        except:
            convTotalAmt = 0.0
        try:
            convDiscAmt = float(discountAmount) if discountAmount is not None else 0.0
        except:
            convDiscAmt = 0.0
        try:
            convCessAmt = float(cessAmount) if cessAmount is not None else 0.0
        except:
            convCessAmt = 0.0
        try:
            convAddlCess = float(additionalCessAmount) if additionalCessAmount is not None else 0.0
        except:
            convAddlCess = 0.0
        calcTotalGST = convIGSTAmount + convSGSTAmount + convCGSTAmount
        calcAddlTax = convCessAmt + convAddlCess

        print("Total Tax",calcTotalGST)
        print("Sub Total",convSubTotal)
        print("Addl Tax", calcAddlTax)
        print("Total Amt",convTotalAmt)

        if convTotalAmt == 0:
            result.append(("totalAmount",
                          "Total Amount should contain a value greater than zero"))
        if calcTotalGST != convTotalGST:
            result.append(("totalGSTAmount",
                          "Total GST Amount should be sum of individual GST Amounts"))
        if calcTotalGST > 0.0:
            if vendorGSTIN is not None and billingGSTIN is not None:
                print("vendorGSTIN",vendorGSTIN)
                print("billingGSTIN",billingGSTIN)
                print("CGST",CGSTAmount,
                      "SGST",SGSTAmount,
                      "IGST",IGSTAmount)
                print("condition",
                      vendorGSTIN[:2].upper() == billingGSTIN[:2].upper(),
                      ((convCGSTAmount == 0) or (convSGSTAmount == 0)))
                if (vendorGSTIN[:2].upper() == billingGSTIN[:2].upper()) and ((convSGSTAmount == 0) or (convCGSTAmount == 0)):
                    if convCGSTAmount == 0:
                        result.append(("CGSTAmount",
                                      "CGST Amount cannot be blank or zero when total tax is not empty"))
                    if convSGSTAmount == 0:
                        result.append(("SGSTAmount",
                                      "SGST Amount cannot be blank or zero when total tax is not empty"))
                elif (vendorGSTIN[:2].upper() != billingGSTIN[:2].upper()) and (convIGSTAmount == 0):
                    if convIGSTAmount == 0:
                        result.append(("IGSTAmount",
                                      "IGST Amount cannot be blank or zero when total tax is not empty"))
        if convSubTotal > 0 or calcTotalGST > 0 or calcAddlTax > 0:
            netAmount = convSubTotal - convDiscAmt + calcTotalGST + calcAddlTax
            import math
            print("Net Amount",
                  netAmount,
                  "convTotalAmount",
                  convTotalAmt,
                  math.isclose(netAmount,
                               convTotalAmt,
                               abs_tol = 0.5))
            if not math.isclose(netAmount,
                                convTotalAmt,
                                abs_tol = 0.5):
                result.append(("totalAmount",
                              "Total Amount should be a sum of subtotal - Discount + GST + Cess. A tolerance of .5 Rs is acceptable"))
        if convSubTotal == 0:
            result.append(("subTotal",
                           "Sub Total should be greater than 0"))
        return result
    except:
        print("bizRuleValidateForUi",
              traceback.print_exc())
        return None

def isMasterDataPresent(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId, callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["vendorGSTIN",
                                           "billingGSTIN",
                                           "shippingGSTIN"]:
                    masterData = docInfo_["entityMasterdata"]
                    if not masterData:
                        return False
            return True
        else:
            return False
    except:
        print("getDocumentMasterDataPresent",
              traceback.print_exc())
        return False

def isInvoiceNumberAnAmount(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                               callbackUrl)
        print("docResult isInvoiceNumberAnAmount ",docResult)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["invoiceNumber"]:
                    if putil.validAmount(docInfo_["fieldValue"]) or putil.validDate(docInfo_["fieldValue"]):
                        print("isInvoiceNumberAnAmount -> validAmount : ", True)
                        return True
            # print("isInvoiceNumberAnAmount : ", False)       
            return False
        else:
            print("isInvoiceNumberAnAmount-> getDocumentResultApi is None : ", False)
            return False
    except:
        print("isInvoiceNumberAnAmount",
              traceback.print_exc())
        print("isInvoiceNumberAnAmount exception : ", False)
        return True

def isTotOrSubCalc(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"] in ["totalAmount","subTotal"]:
                    if docInfo_.get("calculated_field"):
                        if docInfo_.get("calculated_field") == 1:
                            return True
            return False
        else:
            return False
    except:
        print("isTotOrSubCalc",
              traceback.print_exc())
        return True

def is_equal_subTotal_TotalAmount(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        total = None
        subtotal = None
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                if docInfo_["fieldId"]=="totalAmount":
                    if docInfo_.get("fieldValue"):
                        total = float(docInfo_.get("fieldValue"))
                if docInfo_["fieldId"] == "subTotal":
                    if docInfo_.get("fieldValue"):
                        subtotal = float(docInfo_.get("fieldValue"))
                if (total is not None) and (subtotal is not None):
                    print("total :",total,"subtotal :",subtotal,"abs :",abs(total-subtotal))
                    if (abs(total-subtotal)) < 1:
                        return False
        return None 
    except:
        print("isTotOrSubCalc",
              traceback.print_exc())
        return None

def check_date_field_stp_score(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                if docInfo_["fieldId"]=="invoiceDate":
                    print("checking invoiceDate stp confidence",docInfo_.get("confidence"))
                    if docInfo_.get("confidence") < 60:
                        print("Date confidance not meet the stp score")
                        return False
        return True
    except:
        print("check_date_field_stp_score exception",
              traceback.print_exc())
        return False

def check_multi_invoices_stp(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                #print(docInfo_["fieldId"])
                if docInfo_["fieldId"]=="invoiceNumber":
                    print("checking multiple invoiceNumber flag :",docInfo_.get("multi_invoices"))
                    if docInfo_.get("multi_invoices") == 1:
                        print("multiple invoices present")
                        return False
        return True
    except:
        print("Multi invoices check exception",
              traceback.print_exc())
        return False

def getBuyerStateCode(documentId, callbackUrl):

    try:
        docResult = putil.getDocumentResultApi(documentId,
                                                callbackUrl)
        if docResult is not None:
            docInfo = docResult["result"]["document"]["documentInfo"]
            for docInfo_ in docInfo:
                if docInfo_["fieldId"]=="billingGSTIN":
                    billingGSTIN = docInfo_["fieldValue"]
                    if billingGSTIN:
                        if len(billingGSTIN) > 2:
                            billingStateCode = billingGSTIN[:2]
                            return billingStateCode
        return None
    except:
        print("check_date_field_stp_score exception",
              traceback.print_exc())
        return None

