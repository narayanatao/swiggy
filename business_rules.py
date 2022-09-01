from copy import deepcopy
from traceback import print_tb
import traceback
import preProcUtilities as putil
import re
import pickle
import pandas as pd
import operator
import numpy as np
from difflib import SequenceMatcher
import os
import json
import math
import collections

from string import punctuation
# Read Client Configurataions File
script_dir = os.path.dirname(__file__)

ClientConfigFilePath = os.path.join(script_dir,
                              "Utilities/client_config.json")

FieldCorrectionFilePath = os.path.join(script_dir,
                              "Utilities/field_corrections.json")
def read_rules(json_file_path = ClientConfigFilePath):
    """
    """
    rules = {}
    with open(json_file_path) as json_file:
        rules = json.load(json_file)
    return rules

CONFIGURATIONS = read_rules()
FIELD_CORRECTIONS = read_rules(FieldCorrectionFilePath)


# Read Product Mapping File
ProductMappingFilePath = os.path.join(script_dir,
                              "Utilities/PRODUCT_MAPPING.csv")

PRODUCT_MAPPING = pd.read_csv(ProductMappingFilePath, encoding='unicode_escape')
PRODUCT_MAPPING = PRODUCT_MAPPING.astype(str)
unique_HSNCodes = set(PRODUCT_MAPPING['HSNCode'].unique())


# with open('post_processor_field_labels.pkl', 'rb') as handle:
#     FIELD= pickle.load(handle)

FIELD = putil.getPostProcessFieldLabels()
FIELD = {'date_fields': ['invoiceDate', 'dueDate'],
'header_fields': ['invoiceNumber', 'poNumber', 'paymentTerms'],
'vendor_specific_addrefinal_candidates_fields': ['vendorAddress'],
'address_fields': ['shippingAddress', 'billingAddress'],
'vendor_specific_header_fields': ['vendorGSTIN', 'vendorName', 'vendorEmail', 'currency'],
'amount_fields': ['totalAmount', 'taxAmount', 'subTotal','SGSTAmount', 'CGSTAmount', 'IGSTAmount',
'freightAmount','discountAmount', 'TCSAmount','CessAmount','insuranceAmount'],
'tax_rate_fields': ['taxRate'],
'lineitem_header_labels': ['hdr_itemCode', 'hdr_itemDescription', 'hdr_itemQuantity', 'hdr_unitPrice', 'hdr_itemValue',
'hdr_taxAmount', 'hdr_taxRate','hdr_CGSTAmount','hdr_SGSTAmount','hdr_IGSTAmount', 'hdr_UOM', 'hdr_HSNCode'],
'lineitem_value_labels': ['LI_itemQuantity', 'LI_unitPrice', 'LI_itemValue', 'LI_taxAmount', 'LI_taxRate', 'LI_itemCode',
'LI_itemDescription','LI_CGSTAmount', 'LI_SGSTAmount', 'LI_IGSTAmount','LI_UOM', 'LI_HSNCode'],
'total_fields': ['totalAmount', 'subTotal'], 'lbl_total_fields': ['lblTotalAmount', 'lblSubTotal']}


STP_CONFIGURATION = putil.getSTPConfiguration()

LI_amount_list = ['itemQuantity', 'unitPrice', 'itemValue', 'CGSTAmount', 'SGSTAmount', 'IGSTAmount', "CessAmount"]
def correct_fields(prediction, format_):
    """
    This is just for header fields
    """
    print("correct_fields", format_)
    if format_ in FIELD_CORRECTIONS:
        val = FIELD_CORRECTIONS[format_]
        for f, shape in val.items():
            if (f in prediction) & (prediction[f] is not None):
                extracted_value = prediction[f]['text']
                if len(str(extracted_value)) == len(str(shape)):
                    start = re.search(r'[^X]', shape).start()
                    if start is not None:
                        end = shape[start:].find('X')
                        str_to_replace = shape[start: start+end]
                        new_value = (extracted_value[:start] + str_to_replace +
                            extracted_value[start+end:])
                        prediction[f]['text'] = new_value
    return prediction

def add_empty_field(field_name,value):
    """
    """
    final_candidates_ = {}

    extracted_value = value
    final_candidates_['line_num'] = 0
    final_candidates_["prob_"+field_name] = 0
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
    final_candidates_['model_confidence'] = 0

    final_candidates_['final_confidence_score'] = 0
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = False

    return {field_name: final_candidates_}


def update_stp_configuration(default_config, vendor_specific_config):
    """
    """
    for k, v in vendor_specific_config.items():
        if isinstance(v, collections.abc.Mapping):
            default_config[k] = update_stp_configuration(default_config.get(k, {}), v)
        else:
            default_config[k] = v
    return default_config


def get_stp_configuration_for_VENDOR_ID(vendor_id):
    """
    """
    STP_CONFIGURATION = putil.getSTPConfiguration()
    default_config = STP_CONFIGURATION["DEFAULT"]
    if vendor_id is not None:
        if vendor_id in STP_CONFIGURATION:
            vendor_specific_config = STP_CONFIGURATION[vendor_id]
            default_config = update_stp_configuration(default_config, vendor_specific_config)
    return default_config


def get_stp_configuration(vendor_masterdata):
    """
    """
    STP_CONFIGURATION = putil.getSTPConfiguration()
    default_config = STP_CONFIGURATION["DEFAULT"]
    if vendor_masterdata is not None:
        vendor_id = vendor_masterdata['VENDOR_ID']
        if vendor_id in STP_CONFIGURATION:
            vendor_specific_config = STP_CONFIGURATION[vendor_id]
            default_config = update_stp_configuration(default_config, vendor_specific_config)
    return default_config

def client_required_fields(prediction, vendor_masterdata):
    """
    """
    stp_config = get_stp_configuration(vendor_masterdata)
    #print("client_required_fields162:", stp_config)
    display_fields = [key for key,val in stp_config.items() if (val["display_flag"] == 1)]
    print("Fields to be displayed:", display_fields)
    header_required_fields = {v for v in display_fields if ("LI_" not in v)}
    lineitem_required_fields = {v for v in display_fields if ("LI_" in v)}
    lineitem_required_fields = {str(v).replace("LI_","") for v in lineitem_required_fields}
    print("Header Required Fields:", header_required_fields)
    print("LI Required Fields:", lineitem_required_fields)
    header_fields_present = set()
    header_fields = set()

    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        row_columns = set()
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            row_columns.add(col_name)
                        LI_fields_to_remove = row_columns - lineitem_required_fields
                        LI_fields_to_add = lineitem_required_fields - row_columns
                        while len(LI_fields_to_remove)>0:
                            f = LI_fields_to_remove.pop()
                            for item in row_prediction:
                                col_name = list(item.keys())[0]
                                if col_name == f:
                                    row_prediction.remove(item)
                                    break
                        for f in LI_fields_to_add:
                            row_prediction.append(add_empty_field(f,''))
        else:
            if val is not None:
                header_fields_present.add(key)
            header_fields.add(key)
    # To remove required fileds with None
    # header_fields_to_remove = header_fields_present - header_required_fields
    
    # Keeping required field even if it has None value
    header_fields_to_remove = header_fields - header_required_fields
    header_fields_to_add = header_required_fields - header_fields_present
    print("header_fields_to_remove :",header_fields_to_remove)
    print("header_fields_to_add :",header_fields_to_add)

    for f in header_fields_to_remove:
        prediction.pop(f, None)
        print("Removed Not required filed :",f)
    amoutFields = ["totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount","CessAmount",
                    "discountAmount", "totalGSTAmount","additionalCessAmount",]
    fields_to_add = {}
    breaker = None
    for f in header_fields_to_add:
        for item in amoutFields:
            if item == f:
                out_dict = add_empty_field(f,value = 0)
                fields_to_add.update(out_dict)
                breaker = True
                break
        if breaker:
            continue
        out_dict = add_empty_field(f, value = '')
        fields_to_add.update(out_dict)
    prediction = {**prediction, **fields_to_add}

    ## replacing None Val with empty to show the required field into UI
    for item in display_fields:
        if (prediction.get(item) is None):
            print("Key with None val :",prediction.get(item))
            prediction.update(add_empty_field(item, value = ''))
            print("Added None key value with empty :",item)

    return prediction

def add_new_field(field_name, value,model_confidence =1,probOfField = 1,final_confidence_score =1,from_Vendor=False,from_entity = False,calculated =False):
    """
    """
    final_candidates_ = {}

    extracted_value = str(value)
    final_candidates_['line_num'] = 0
    final_candidates_["prob_"+field_name] = probOfField
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
    final_candidates_['model_confidence'] = model_confidence

    final_candidates_['final_confidence_score'] = final_confidence_score
    final_candidates_['vendor_masterdata_present'] = True
    final_candidates_['extracted_from_masterdata'] = from_Vendor
    final_candidates_['extracted_from_entitydata'] = from_entity
    final_candidates_['calculated_field'] = calculated 

    return {field_name: final_candidates_}


def add_lineitem_dummy_row(prediction, vendor_masterdata):
    """
    Add one dummy lineitem row if no lineitems are there
    """
    stp_config = get_stp_configuration_for_VENDOR_ID(vendor_masterdata)
    # print("add_lineitem_dummy_row:", prediction)
    display_fields = [key for key,val in stp_config.items() if (val["display_flag"] == 1)]
    lineitem_required_fields = {v for v in display_fields if ("LI_" in v)}
    lineitem_required_fields = {str(v).replace("LI_","") for v in lineitem_required_fields}
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if (val is None) or (len(val) == 0):
                print("Adding Dummy LI field")
                row_prediction = []
                for f in lineitem_required_fields:
                    row_prediction.append(add_empty_field(f,''))
                val = {0:{1:row_prediction}}
                prediction['lineItemPrediction'] = val

    return prediction


def extract_vendorGSTIN(DF, prediction):
    """
    """
    chars_to_keep = '[^0-9A-Za-z]'
    DF['text'] = DF['text'].str.replace(chars_to_keep, '', regex=True)
    gstin_regex = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}'
    DF['text'] = DF['text'].astype(str)
    DF['text'] = DF['text'].str.replace(':','')
    DF['GSTIN_LEN'] = (DF['text'].str.len() == 15)
    DF['GSTIN_CHECK'] = DF['text'].str.match(gstin_regex).astype(bool)
    DF['GSTIN_CHECK'] = DF['GSTIN_CHECK'] & DF['GSTIN_LEN']

    TEMP = DF.loc[DF['GSTIN_CHECK'] == True]
    print("extract_vendorGSTIN", DF.loc[DF['GSTIN_CHECK'] == True][['text']])

    if TEMP.shape[0]==0:
        # If no valid GSTIN found, do nothing
        return prediction

    vendorGSTIN = None
    billingGSTIN = None
    shippingGSTIN = None
    for key, val in prediction.items():
        if (key == "vendorGSTIN") & (val is not None):
            if val['extracted_from_masterdata']:
                # No need to further extract vendorGSTIN if it has already been extracted from
                # Vendor MasterData
                return prediction
            vendorGSTIN = val['text']
        elif (key == "shippingGSTIN") & (val is not None):
            shippingGSTIN = val['text']
        elif (key == "billingGSTIN") & (val is not None):
            billingGSTIN = val['text']



    TEMP["billingGSTIN_MS"] = TEMP['text'].apply(find_similarity_words, b=billingGSTIN)
    TEMP["shippingGSTIN_MS"] = TEMP['text'].apply(find_similarity_words, b=shippingGSTIN)

    TEMP = TEMP.loc[TEMP['billingGSTIN_MS'] < 0.8]
    TEMP = TEMP.loc[TEMP['shippingGSTIN_MS'] < 0.8]

    # TEMP["vendorGSTIN_MS"] = 0
    # if vendorGSTIN is not None:
    #     TEMP["vendorGSTIN_MS"] = TEMP['text'].apply(find_similarity_words, b=vendorGSTIN)

    # TEMP = TEMP.loc[TEMP['vendorGSTIN_MS'] > 0.8]
    if TEMP.shape[0] == 1:
        # TEMP.sort_values('vendorGSTIN_MS', ascending=False, inplace = True)
        dict_ = TEMP[['text', 'page_num', 'line_num', 'word_num', 'left', 'right', 'top', 'bottom',
                    'conf','image_height', 'image_widht', 'height', 'width']].iloc[0].to_dict()

        d_ = {'Label_Present': False, 'label_confidence': 0.0, 'wordshape': '',
         'wordshape_confidence': 0.0, 'Odds': 1, 'model_confidence': 1,
         'final_confidence_score': 1, 'vendor_masterdata_present': False,
         'extracted_from_masterdata': False, 'prob_vendorGSTIN':1}

        dict_ = {**dict_, **d_}

        prediction["vendorGSTIN"] = dict_

    return prediction


def find_similarity_words(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def reduce_confidence_LI(prediction):
    """
    return => dictionary 
    """
    try:
            
        #print("PRODUCT_MAPPING:", PRODUCT_MAPPING)
        for key, val in prediction.items():
            if key == "lineItemPrediction":
                if val is not None:
                    for page, page_prediction in val.items():
                        for row, row_prediction in page_prediction.items():
                            row_pred = {}
                            min_row_top = 1.0
                            max_row_bottom = 0.0
                            for item in row_prediction:
                                col_name = list(item.keys())[0]
                                predicted_value = item[col_name]['text']
                                row_pred[col_name] = predicted_value
                            if all(x in list(row_pred.keys()) for x in ['itemQuantity', 'unitPrice', 'itemValue']):
                                print("Check itemValue = itemQuantity * unitPrice!!!")
                                itemValue = float(row_pred['itemValue'])
                                unitPrice = float(row_pred['unitPrice'])
                                itemQuantity = float(row_pred['itemQuantity'])
                                print(itemValue, unitPrice, itemQuantity)
                                derived_itemValue = unitPrice * itemQuantity
                                if not math.isclose(itemValue, derived_itemValue, rel_tol=0.1):
                                    print("Reducing confidence!!!!")
                                    for item in row_prediction:
                                        col_name = list(item.keys())[0]
                                        if col_name in ['itemQuantity', 'unitPrice', 'itemValue']:
                                            item[col_name]['prediction_probability'] = 0.40
                                            item[col_name]['model_confidence'] = 0.40
                                            item[col_name]['model_confidence'] = 0.40
                                            print(item[col_name])
        return prediction
    except :
        return prediction
        
def increase_confidence_LI(prediction, fields=[]):
    """
    """
    #print("PRODUCT_MAPPING:", PRODUCT_MAPPING)
    try:
        print("INCREASING CONFIDENCE")
        for key, val in prediction.items():
            if key == "lineItemPrediction":
                if val is not None:
                    for page, page_prediction in val.items():
                        for row, row_prediction in page_prediction.items():
                            row_pred = {}
                            min_row_top = 1.0
                            max_row_bottom = 0.0
                            for item in row_prediction:
                                print(item)
                                col_name = list(item.keys())[0]
                                predicted_value = item[col_name]['text']
                                row_pred[col_name] = predicted_value
                            if len(fields)>0:
                                print("I am here")
                                print(fields)
                                for field in fields:
                                    print(field)
                                    var = str(row_pred[field])
                                    print(var)
                                    for item in row_prediction:
                                        col_name = list(item.keys())
                                        if field in col_name:
                                            c_score = np.random.uniform(0.8,0.99)
                                            item[field]['prediction_probability'] = c_score
                                            item[field]['model_confidence'] = c_score
                                            item[field]['model_confidence'] = c_score
                                            print(item[field])

        return prediction
    except:
        return prediction

def extract_HSNCode(DF, prediction):
    """
    """
    #print("PRODUCT_MAPPING:", PRODUCT_MAPPING)

    pred = {}
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                changed_pred = {}
                for page, page_prediction in val.items():
                    changed_page_prediction = {}
                    for row, row_prediction in page_prediction.items():
                        row_columns = []
                        min_row_top = 1.0
                        max_row_bottom = 0.0
                        changed_row_prediction = []
                        # print(row_prediction)
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            predicted_value = item[col_name]
                            row_columns.append(col_name)
                            min_row_top = min(min_row_top, predicted_value['top'])
                            max_row_bottom = max(max_row_bottom, predicted_value['bottom'])
                        #print(row_columns, min_row_top, max_row_bottom)
                        if "HSNCode" not in row_columns:
                            TEMP = DF.loc[(DF['top'] >= min_row_top) &
                            (DF['bottom'] <= max_row_bottom)]
                            list_of_tokens = TEMP['text'].astype(str).to_list()
                            list_of_tokens = [i.upper() for i in list_of_tokens]

                            print("Extract HSNCode from Product Mapping")
                            print(list_of_tokens)
                            print(unique_HSNCodes)
                            identified_HSNCodes = unique_HSNCodes.intersection(set(list_of_tokens))
                            print("identified_HSNCodes:", identified_HSNCodes)

                            if len(identified_HSNCodes) == 0:
                                string_tokens = ''.join(list_of_tokens)
                                for c in unique_HSNCodes:
                                    if c in string_tokens:
                                        identified_HSNCodes.add(c)

                            HSN_CODES = PRODUCT_MAPPING.loc[PRODUCT_MAPPING['HSNCode'].isin(identified_HSNCodes)]
                            HSN_CODES.reset_index(inplace=True)
                            match_scores = {}
                            for idx, r in HSN_CODES.iterrows():
                                item_code = r["ItemCode"]
                                item_desc = r["ItemDescription"]
                                s = str(item_code) + " "+ str(item_desc)
                                s = set(s.upper().split())
                                s_match = s.intersection(set(list_of_tokens))
                                match_scores[idx] = len(s_match)/len(s)
                            #print(match_scores)
                            if len(match_scores) > 0:
                                predicted_HSNIndex = max(match_scores.items(), key=operator.itemgetter(1))[0]
                                predicted_HSNCode = HSN_CODES.iloc[predicted_HSNIndex]['HSNCode']
                                #print("predicted_HSNCode:", predicted_HSNCode)
                                row_prediction.append({"HSNCode": {'text': predicted_HSNCode,
                                 'left': 0.0, 'right': 1.0, 'top': min_row_top, 'bottom': max_row_bottom,
                                'conf': 1.0, 'prediction_probability': 1.0,
                                 'model_confidence': 1.0, 'image_height': 1, 'image_widht': 1}})
    return prediction

def clean_trailing_punctuations(prediction):
    #removes special characters from poNumber,invoiceNumber,invoiceDate if they appear before or after string
    try:
        #pred = {}
        header_fields = ["poNumber","invoiceNumber","invoiceDate","totalAmount","subTotal","CGSTAmount","SGSTAmount","IGSTAmount"]
        for key, val in prediction.items():
            if key in header_fields:
                if val is not None:
                        text=val["text"]
                        text = list(text[::-1])
                        text_ = text.copy()
                        for i,s in enumerate(text_):
                            if s.isalpha() or s.isnumeric():
                                break
                            else:
                                del text[0]
                        text = list(text[::-1])
                        text_ = text.copy()
                        for i,s in enumerate(text_):
                            if s.isalpha() or s.isnumeric():
                                break
                            else:
                                del text[0]
                        updated_text=("".join(text))
                        val['text'] = updated_text
                        #pred[key] = val
                        prediction[key] = val

            return prediction
    except:
        return prediction


def delete_characters_amount(text):
    """
    """
    chars_to_keep = '[^0123456789.]'
    updated_text = re.sub(chars_to_keep, '', text)
    split = updated_text.split('.')
    if len(split) > 2:
        updated_text = ''.join(split[:-1])
        updated_text = updated_text + "."
        updated_text = updated_text + str(split[-1])

    return updated_text


def clean_amounts(prediction):
    for key, val in prediction.items():
       	if key in FIELD['amount_fields']:
            if val is not None:
               text = val['text']
               updated_text = delete_characters_amount(text)
               val['text'] = updated_text
        elif key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            predicted_value = item[col_name]
                            if col_name in LI_amount_list:
                                text = predicted_value['text']
                                updated_text = delete_characters_amount(text)
                                predicted_value['text'] = updated_text

    return prediction

def scale_bounding_boxes(DF, prediction):
    """
    """
    print("Inside rescaling method")
    #img_h_scaler = DF.iloc[0]['height_scaler']
    #img_w_scaler = DF.iloc[0]['width_scaler']
    img_h_scaler = DF.iloc[0].get('height_scaler',1.0)
    img_w_scaler = DF.iloc[0].get('width_scaler',1.0)
    print("Rescaling --> height by {} and width by {}".format(img_h_scaler, 
                                                              img_w_scaler))
    pred = prediction.copy()
    for key, val in pred.items():
        if key != "lineItemPrediction":
            if val is not None:
                val['left']*= img_w_scaler
                val['right']*= img_w_scaler
                val['top']*= img_h_scaler
                val['bottom']*= img_h_scaler
            
        elif key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            predicted_value = item[col_name]
                            predicted_value['left']*= img_w_scaler
                            predicted_value['right']*= img_w_scaler
                            predicted_value['top']*= img_h_scaler
                            predicted_value['bottom']*= img_h_scaler
    
    return pred

## Adding GST/ Tax TotalAmount
def get_totalGSTAmount(df, prediction):
    try:
        sumTotal = 0.0
        predicted_cgst = prediction.get('CGSTAmount')
        predicted_sgst = prediction.get('SGSTAmount')
        predicted_igst = prediction.get('IGSTAmount')
        
        is_CGST_SGST = df["is_CGST_SGST"].unique()
        is_CGST_SGST = is_CGST_SGST[0]
        print("is_CGST_SGST :",is_CGST_SGST)
        is_IGST = df["is_IGST"].unique()
        is_IGST = is_IGST[0]
        print("is_IGST :",is_IGST)
        if(is_CGST_SGST == 1):
            print("inside cgs 1")
            if (predicted_cgst is not None):
                print("inside cgst not none")
                if (predicted_cgst.get("text") !='') or (predicted_cgst.get("text") != None):
                    print("inside cgst text ")
                    cgstAmount = 0.0
                    try: cgstAmount = float(predicted_cgst.get("text"))
                    except: print("fload conversion exception :",traceback.print_exc())
                    sumTotal = (sumTotal + cgstAmount)
                    print("sumTotal of cgst :",sumTotal)

            if (predicted_sgst is not None):
                print("inside sgst not none")
                if (predicted_sgst.get("text") !='') or (predicted_sgst.get("text") != None):
                    print("inside sgst text ")
                    sgstAmount = 0.0
                    try: sgstAmount = float(predicted_sgst.get("text"))
                    except: print("fload conversion exception :",traceback.print_exc())
                    sumTotal = (sumTotal + sgstAmount)
                    print("sumTotal of Sgst :",sumTotal)
            if (sumTotal > 0):
                return {**prediction,**add_new_field("totalGSTAmount",sumTotal)}
            else:
                return prediction
        else:
            print("is_CGST_SGST is not 1")

        if (int(is_IGST) == 1):
            if (predicted_igst is not None):
                if (predicted_igst.get("text") !='') or (predicted_igst.get("text") != None):
                    igstAmount = 0.0
                    try: igstAmount = float(predicted_igst.get("text"))
                    except: print("fload conversion exception :",traceback.print_exc())
                    sumTotal = (sumTotal + igstAmount)
            return {**prediction,**add_new_field("totalGSTAmount",sumTotal)}
        return prediction
    except:
        print("TotalGSTAmount Exxception :",traceback.print_exc())
        return prediction
        
#*****Code changes by Soumya from 595 to 780 *********#
def validate_gst_amounts(prediction):
    '''
    Case 1: if CGST and SGST both are present 
	GST = SGST + CGST
    Case 2: IGST
    -> TotalGST = IGST
    if sgst cgst  and igst >0 lower confidence
    Case 3: CGST is present but SGST is not
    -> SGST = CGST
    -> Total Tax = SGST + CGST

    '''
    print("New method for GST validation")
    pred = prediction
    # Get CGST SGST & IGST
    predicted_cgst = prediction.get('CGSTAmount')
    predicted_sgst = prediction.get('SGSTAmount')
    predicted_igst = prediction.get('IGSTAmount')
    try:
        CGSTAmount, SGSTAmount, IGSTAmount = None, None, None
        if predicted_cgst:
            if (predicted_cgst.get('text') is None) or (predicted_cgst.get('text') == ''):
                CGSTAmount = None
            else: 
                CGSTAmount = float(predicted_cgst.get('text'))
        if predicted_sgst:
            if (predicted_sgst.get('text') is None) or (predicted_sgst.get('text') == ''):
                SGSTAmount = None
            else: 
                SGSTAmount = float(predicted_sgst.get('text'))
        if predicted_igst:
            if (predicted_igst.get('text') is None) or (predicted_igst.get('text') == ''):
                IGSTAmount = None
            else: 
                IGSTAmount = float(predicted_igst.get('text'))
    except:
        print("Amounts can't be parsed to float")
        return prediction
    print(f"Predicted CGST: {CGSTAmount}, SGST: {SGSTAmount}, IGST: {IGSTAmount}")
    
    SGST_UPDATED, CGST_UPDATED = 0,0
    
    if CGSTAmount and SGSTAmount:
        totalGSTAmount = CGSTAmount + SGSTAmount
    elif (CGSTAmount is not None) and (SGSTAmount is None):
        SGSTAmount = CGSTAmount
        SGST_UPDATED = 1
        totalGSTAmount = SGSTAmount + CGSTAmount
    elif (SGSTAmount is not None) and (CGSTAmount is None):
        CGSTAmount = SGSTAmount
        CGST_UPDATED = 1
        totalGSTAmount = SGSTAmount + CGSTAmount
    elif IGSTAmount:
        totalGSTAmount = IGSTAmount
    else:
        totalGSTAmount = ''
    print("Total GST:", totalGSTAmount)
    for key, val in prediction.items():
        if (key == "CGSTAmount") and CGST_UPDATED:
            pred = {**pred, **add_new_field("CGSTAmount",str(CGSTAmount))}
        if (key == "SGSTAmount") and SGST_UPDATED:
            pred = {**pred, **add_new_field("SGSTAmount",str(SGSTAmount))}
    return {**pred, **add_new_field("totalGSTAmount",totalGSTAmount)}

def reduce_field_confidence(prediction, fields,model_confidence = 0.4,final_confidence_score = 0.4):
    '''Reducing the field confidence'''
    for key, val in prediction.items():
        if key in fields:
            if val is not None:
                # print(val)
                for val_key, val_val in val.items():
                    if 'model_confidence' in val_key:
                        val['model_confidence'] = model_confidence
                        val['final_confidence_score'] = final_confidence_score
    return prediction


def reduce_confidence_itemValue(prediction):
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            if col_name == 'itemValue':
                                item[col_name]['model_confidence'] = 0.40
                                item[col_name]['prediction_probability'] = 0.40

    return prediction

def reduce_subTotal_confidence(flag,prediction):
    if flag == 1:
        prediction = reduce_field_confidence(prediction,['subTotal'])
    elif flag==2:
        prediction = reduce_confidence_itemValue(prediction)
    return prediction


def conf_reduction_amounts(DF,prediction):
    '''
    1. if SGST !=CGST , reduce confidence 
    2. if IGST SGST/CGST exists, reduce confidence '''
    
    print("New method for GST validation")
    pred = prediction
    predicted_cgst = prediction.get('CGSTAmount')
    predicted_sgst = prediction.get('SGSTAmount')
    predicted_igst = prediction.get('IGSTAmount')
    predicted_totalGST = prediction.get('totalGSTAmount')
    try:
        CGSTAmount, SGSTAmount, IGSTAmount = None, None, None
        if predicted_cgst:
            if (predicted_cgst.get('text') is None) or (predicted_cgst.get('text') == ''):
                CGSTAmount = None
            else: 
                CGSTAmount = float(predicted_cgst.get('text'))
        if predicted_totalGST:
            if (predicted_totalGST.get('text') is None) or (predicted_totalGST.get('text') == ''):
                totalGSTAmount = None
            else: 
                totalGSTAmount = float(predicted_totalGST.get('text'))
        if predicted_sgst:
            if (predicted_sgst.get('text') is None) or (predicted_sgst.get('text') == ''):
                SGSTAmount = None
            else: 
                SGSTAmount = float(predicted_sgst.get('text'))
        if predicted_igst:
            if (predicted_igst.get('text') is None) or (predicted_igst.get('text') == ''):
                IGSTAmount = None
            else: 
                IGSTAmount = float(predicted_igst.get('text'))
    except:
        print("Amounts can't be parsed to float")
        return prediction
    print(f"Predicted CGST_: {CGSTAmount}, SGST: {SGSTAmount}, IGST: {IGSTAmount}")
    if 'totalGSTAmount' not in prediction.keys():
        SGSTAmount, CGSTAmount, IGSTAmount = 0,0,0
        prediction = reduce_field_confidence(prediction,['SGSTAmount', 'CGSTAmount', 'IGSTAmount'])
    else:
        calc_subtotal = lineItemSubtotal(prediction)
        prediction,flag = reduction_charges(DF,prediction,calc_subtotal)
        prediction = reduce_subTotal_confidence(flag,prediction)
    if SGSTAmount and CGSTAmount:
        if SGSTAmount != CGSTAmount:
            prediction = reduce_field_confidence(prediction, fields =['SGSTAmount', 'CGSTAmount'])
    elif (SGSTAmount or CGSTAmount) and IGSTAmount:
        prediction = reduce_field_confidence(prediction, fields =['SGSTAmount', 'CGSTAmount', 'IGSTAmount'])
    print('predictiondict', prediction)
    return prediction

def reduction_charges(DF,prediction,calc_subtotal):
    '''GST charges validation '''
    predicted_subtotal = prediction.get('subTotal')
    predicted_totalGST = prediction.get('totalGST')
    predicted_disc = prediction.get('discountAmount')
    predicted_freight = prediction.get('freightAmount')
    predicted_tcs = prediction.get('TCSAmount')
    predicted_insurance = prediction.get('insuranceAmount')
    predicted_totalamt = prediction.get('totalAmount')
    try:
        subTotal, totalGST, discountAmount, freightAmount, TCSAmount, insuranceAmount, totalAmount = None, None, None,None,None, None,None
        if predicted_subtotal:
            if (predicted_subtotal.get('text') is None) or (predicted_subtotal.get('text') == ''):
                subTotal = None
            else: 
                subTotal = float(predicted_subtotal.get('text'))
        if predicted_totalGST:
            if (predicted_totalGST.get('text') is None) or (predicted_totalGST.get('text') == ''):
                totalGST = None
            else: 
                totalGST = float(predicted_totalGST.get('text'))
        if predicted_disc:
            if (predicted_disc.get('text') is None) or (predicted_disc.get('text') == ''):
                discountAmount = 0
            else: 
                discountAmount = float(predicted_disc.get('text'))
        if predicted_freight:
            if (predicted_freight.get('text') is None) or (predicted_freight.get('text') == ''):
                freightAmount = 0
            else: 
                freightAmount = float(predicted_freight.get('text'))
        if predicted_tcs:
            if (predicted_tcs.get('text') is None) or (predicted_tcs.get('text') == ''):
                TCSAmount = 0
            else: 
                TCSAmount = float(predicted_tcs.get('text'))
        if predicted_insurance:
            if (predicted_insurance.get('text') is None) or (predicted_insurance.get('text') == ''):
                insuranceAmount = 0
            else: 
                insuranceAmount = float(predicted_insurance.get('text'))  
        if predicted_totalamt:
            if (predicted_totalamt.get('text') is None) or (predicted_totalamt.get('text') == ''):
                totalAmount = None
            else: 
                totalAmount = float(predicted_totalamt.get('text'))                       
    except:
        print("Amounts can't be parsed to float")
        return prediction
    totalvals = ['subTotal', 'totalGST', 'discountAmount', 'freightAmount', 'TCSAmount', 'insuranceAmount']
    print('predictedkeys ',prediction.keys())
    flag = 0
    counter = 0
    for i in totalvals:
        if i in prediction.keys():
            counter += 1
    if 'totalAmount' in prediction.keys() and counter == 6:
        extracted_val1 = (subTotal-discountAmount) + freightAmount + TCSAmount + insuranceAmount
        calculated_val1 = (calc_subtotal-discountAmount) + freightAmount + TCSAmount + insuranceAmount
        if extracted_val1 == calculated_val1:
            flag = 3
        if calculated_val1 == totalAmount:
            flag = 1
        elif extracted_val1 == totalAmount:
            flag = 2
        if extracted_val1 != totalAmount:
            prediction = reduce_field_confidence(prediction, totalvals.append('totalAmount'))
    elif 'totalAmount' not in prediction.keys() and counter == 6:
        OCRvalvs = DF['text'].to_list()
        if totalAmount not in OCRvalvs:
            prediction = reduce_field_confidence(prediction, totalvals)
    return prediction, flag



def lineItemSubtotal(prediction):
    calc_subTotal = 0
    for key, val in prediction.items():
        if key == "lineItemPrediction":
            if val is not None:
                for page, page_prediction in val.items():
                    for row, row_prediction in page_prediction.items():
                        for item in row_prediction:
                            col_name = list(item.keys())[0]
                            if col_name == 'itemValue':
                                predicted_value = putil.extract_amount(item[col_name]['text'])
                                calc_subTotal += predicted_value

    
    return calc_subTotal

# Applying GST rules based State codes
clean_GSTIN_line_Text = ['/',':','(',')','.',"'",","]

def get_gstin_of(df,GSTINName):
    df = df[(df["is_gstin_format"]==1) & (df["predict_label"]==GSTINName)]
    if df.shape[0] > 0:
        GSTIN_list = list(set([putil.correct_gstin(s) for s in list(df[df["is_gstin_format"]==1]["text"].unique())])) 
        if(len(GSTIN_list)==1):
            nameGSTIN = GSTIN_list[0]
            return nameGSTIN
        elif(len(GSTIN_list) < 1):
            print("VemdorGSTIN Not Extracted")
            nameGSTIN = None
            return nameGSTIN
        elif(len(GSTIN_list) > 1):
            print("More one GSTIN extracted as VemdorGSTIN")
            nameGSTIN = None
            return nameGSTIN
    else:
        nameGSTIN = None
        return nameGSTIN

def apply_GSTIN_rules(df, prediction):
    BillingGSTIN = prediction.get("billingGSTIN")
    ShippingGSTIN = prediction.get("shippingGSTIN")
    VendorGSTIN = prediction.get("vendorGSTIN")
    CGSTAmount = prediction.get("CGSTAmount")
    SGSTAmount = prediction.get("SGSTAmount")
    IGSTAmount = prediction.get("IGSTAmount")
    df["is_CGST_SGST"]= 0
    df["is_IGST"]= 0
    df_copy = df.copy(deepcopy==True)
    #prediction_copy = prediction
    try:
        updated_fields = {}
        gst_list = list(set([putil.correct_gstin(s) for s in list(df[df["is_gstin_format"]==1]["text"].unique())]))
        print("Unique GSTIN in pred.csv",gst_list)
        noOfGSTIN = len(gst_list)
        
        if(VendorGSTIN is None):
            VendorGSTIN = get_gstin_of(df,"vendorGSTIN")
            print("VendorGSTIN :",VendorGSTIN)
        
        if (BillingGSTIN is None):
            BillingGSTIN = get_gstin_of(df,"billingGSTIN")
            print("BillingGSTIN :",BillingGSTIN)
        
        if (ShippingGSTIN is None):
            ShippingGSTIN = get_gstin_of(df,"shippingGSTIN")
            print("ShippingGSTIN :",ShippingGSTIN)
        CGST_SGST = None
        IGST = None
        if noOfGSTIN >= 2:
            print("Two or more GSTIN Prasent")
            if((noOfGSTIN==2)and(gst_list[0][:2] == gst_list[1][:2])):
                CGST_SGST = True
                df["is_CGST_SGST"]= 1
            
            if((noOfGSTIN==2)and(gst_list[0][:2] != gst_list[1][:2])):
                IGST = True
                df["is_IGST"] = 1
                
            if(noOfGSTIN >2):
                if(VendorGSTIN is not None) & (ShippingGSTIN is not None):
                    if(VendorGSTIN[:2] == ShippingGSTIN[:2]):
                        CGST_SGST = True
                        df["is_CGST_SGST"] = 1
                    else:
                        IGST = True
                        df["is_IGST"] = 1
            
        if CGST_SGST == True:
            print("Inside IntraState Trade")
            if (IGSTAmount is not None):
                #prediction = reduce_field_confidence(prediction, fields = 'IGSTAmount')
                prediction.update({"IGSTAmount":None})
                # removig the field as mention in the rule 2. b.1
                # print("Removed IGST fieldValues in Intrastate")

            if (CGSTAmount is not None) and (SGSTAmount is None):
                print("CSGT is Not none and SGST is None")
                if (CGSTAmount.get("text") is not None) and (CGSTAmount.get("text") != ''):
                    updated_fields.update(add_new_field("SGSTAmount", CGSTAmount.get("text")))
                    print("updated_fields :",updated_fields)

            if (SGSTAmount is not None) and (CGSTAmount is None):
                print("SSGT is Not none and CGST is None")
                if (SGSTAmount.get("text") is not None) and (SGSTAmount.get("text") != ''):
                    updated_fields.update(add_new_field("CGSTAmount", SGSTAmount.get("text")))
                    print("updated_fields :",updated_fields)

            if  (CGSTAmount is not None) and (SGSTAmount is not None):
                if (CGSTAmount.get("text") is None) or (CGSTAmount.get("text")==''):
                    if (SGSTAmount.get("text") is not None) and (SGSTAmount.get("text") != ''):
                        prediction["CGSTAmount"]["text"] = SGSTAmount.get("text")
                        prediction["CGSTAmount"]["model_confidence"] = SGSTAmount.get("model_confidence")
                    else:
                        prediction["CGSTAmount"] = None

                if (SGSTAmount.get("text") is None) or (SGSTAmount.get("text")==''):
                    if (CGSTAmount.get("text") is not None) and (CGSTAmount.get("text") != ''):
                        prediction["SGSTAmount"]["text"] = CGSTAmount.get("text")
                        prediction["SGSTAmount"]["model_confidence"] = CGSTAmount.get("model_confidence")
                    else:
                        prediction["SGSTAmount"] = None
            if updated_fields:
                # print("Mapping Updated fields:",updated_fields)
                return df, {**prediction, **updated_fields}
            else: 
                return df, prediction

        if IGST ==True:
            """
            Rule 2 if the saller and buyer state code is different and and 
            model predict the CGST/SGST Reduce the confidence to 0.4
            """
            print("Inside Interstate Trade")
            if CGSTAmount is not None:
                prediction = reduce_field_confidence(prediction, fields = 'CGSTAmount')
                print("Reduced Field Confidence to :",CGSTAmount.get("model_confidence"))
            if SGSTAmount is not None:
                prediction = reduce_field_confidence(prediction, fields = 'SGSTAmount')
                print("Reduced field Confidence:",SGSTAmount.get("model_confidence"))
            print("Exiting from IGST")
            return df, prediction

        return df, prediction
        
    except:
        print(traceback.print_exc())
        return df_copy, prediction 


#reduce confidence of amounts if total amount is less than cess,tcs,discount or the difference is 0.01%
def reduce_amount_fields_confidenace(prediction):
    '''Validating all amount fields and reducing confidance if it is not matching'''
    predicted_subtotal = prediction.get('subTotal')
    predicted_totalGST = prediction.get('totalGSTAmount')
    predicted_disc = prediction.get('discountAmount')
    predicted_freight = prediction.get('freightAmount')
    predicted_tcs = prediction.get('TCSAmount')
    predicted_totalamt = prediction.get('totalAmount')
    predicted_cessamt = prediction.get('CessAmount')

    try:
        new_fields = {}
        subTotal,discountAmount, freightAmount, TCSAmount, totalAmount,CessAmount,totalGSTAmount = None,None, None, None,None,None, None
        if (predicted_subtotal is not None):
            if (predicted_subtotal.get('text') is not None) or (predicted_subtotal.get('text') != ''):
                subTotal = float(predicted_subtotal.get('text'))
        
        if (predicted_totalGST is not None):
            if (predicted_totalGST.get('text') is not None) or (predicted_totalGST.get('text') != ''):
                totalGSTAmount = float(predicted_totalGST.get('text'))
               
        if predicted_disc is not None:
            if (predicted_disc.get('text') is not None) or (predicted_disc.get('text') != ''):
                discountAmount = float(predicted_disc.get('text'))
            
        if predicted_freight is not None:
            if (predicted_freight.get('text') is not None) or (predicted_freight.get('text') != ''):
                freightAmount = float(predicted_freight.get('text'))

        if predicted_tcs is not None:
            if (predicted_tcs.get('text') is not None) or (predicted_tcs.get('text') != ''):
                TCSAmount = float(predicted_tcs.get('text'))
        
        if predicted_totalamt is not None:
            if (predicted_totalamt.get('text') is not None) or (predicted_totalamt.get('text') != ''):
                totalAmount = float(predicted_totalamt.get('text')  )
                print(totalAmount,"totalAmount")

        if predicted_cessamt is not None:
            if (predicted_cessamt.get('text') is not None) or (predicted_cessamt.get('text') != ''):
                CessAmount = float(predicted_cessamt.get('text')  )
        
        totalvals =['totalAmount', "totalGSTAmount",'subTotal','SGSTAmount', 'CGSTAmount', 'IGSTAmount','freightAmount','discountAmount', 'TCSAmount','CessAmount']
        print('predictedkeys ',prediction.keys())
        sumTotal=0
        if subTotal is not None:
            sumTotal=sumTotal+subTotal
        if totalGSTAmount is not None:
            sumTotal=sumTotal+totalGSTAmount
        if CessAmount is not None:
            sumTotal=sumTotal+CessAmount
        if discountAmount is not None:
            sumTotal=sumTotal-discountAmount
        
        if (sumTotal) >0 and (totalAmount is not None):
            diff = (abs(totalAmount - sumTotal))
            # x=(diff/ sumTotal) * 100
            print("rule_5",totalAmount)
            print("sumtotal",sumTotal)
            if (totalAmount != sumTotal) and (diff >2):
                prediction = reduce_field_confidence(prediction, totalvals)
                print("rule5 satisfied")
                return prediction
            else:
                return prediction
        else:
            return prediction

    except:
        print("reduce_amount_fields_confidenace exception:",traceback.print_exc())
        return prediction

#reduce confidence if total amount is less than or equal to taxes
def reduction_confidence_taxes(prediction):
    predicted_totalGST = prediction.get('totalGSTAmount')
    predicted_totalamt = prediction.get('totalAmount')
    try:
        totalGSTAmount, totalAmount = None, None
        if (predicted_totalGST is not None) & (predicted_totalamt is not None):
            if predicted_totalGST is not None:
                if (predicted_totalGST.get('text') is not None) or (predicted_totalGST.get('text') != ''):
                    totalGSTAmount = float(predicted_totalGST.get('text'))
            if predicted_totalamt is not None:
                if (predicted_totalamt.get('text') is not None) or (predicted_totalamt.get('text') != ''):
                    totalAmount = float(predicted_totalamt.get('text')) 
                    print(totalAmount,totalGSTAmount,"total_RULE_4")
            if (totalGSTAmount is not None) & (totalAmount is not None):
                if(totalAmount<=totalGSTAmount):
                    print("rule4_satisfied")
                    prediction = reduce_field_confidence(prediction,['SGSTAmount', 'CGSTAmount', 'IGSTAmount','totalAmount'])
                    return prediction
                else:
                    print("totalGST is less than totalAmount")
                    return prediction
        else:
            return prediction
    except:
        print("reduction_confidence_taxes exception:",traceback.print_exc())
        return prediction
        
##  adding Required fields as empty  if it is not extracted 
def add_required_fields(df,prediction):
    try:
        # Swiggy Mandatory Fields
        # Hard Coded funtion if the client required fields fuction does't add empty fields
        mandatoryFields = ["invoiceDate", "invoiceNumber", "billingGSTIN", "shippingGSTIN",
         "vendorGSTIN", "vendorName","billingName","shippingName", "vendorAddress", "shippingAddress", "billingAddress", 
         "totalAmount", "CGSTAmount", "SGSTAmount", "IGSTAmount", "CessAmount", "freightAmount", 
         "TCSAmount", "subTotal"]
        is_CGST_SGST = df["is_CGST_SGST"].unique()
        is_CGST_SGST = is_CGST_SGST[0] 
        is_IGST = df["is_IGST"].unique()
        print("is_CGST_SGST :",is_CGST_SGST)
        is_IGST = int(is_IGST[0])
        print("is_IGST :",is_IGST)
        probOfField, model_confidence, final_confidence_score = 0,0,0
        adding_empty_fields = {}
        for field in mandatoryFields:
            if (prediction.get(field) is None):
                print("Fields is None",field)
                if field in ["CGSTAmount", "SGSTAmount", "IGSTAmount"]:
                    print("GST Field")
                    if (field == "IGSTAmount") & (is_IGST == 1):
                        adding_empty_fields.update(add_new_field(field, '',probOfField, model_confidence, final_confidence_score))
                    else:
                        continue
                    if (field == "CGSTAmount") & (is_CGST_SGST == 1):
                        adding_empty_fields.update(add_new_field(field, '',probOfField, model_confidence, final_confidence_score))
                    else: 
                        continue
                    if (field == "SGSTAmount") & (is_CGST_SGST == 1):
                        adding_empty_fields.update(add_new_field(field, '',probOfField, model_confidence, final_confidence_score))
                    else:
                        continue 
                else:
                    adding_empty_fields.update(add_new_field(field, '',probOfField, model_confidence, final_confidence_score))
            else:
                continue        
        # print("Added empty field :",adding_empty_fields)

        if adding_empty_fields:
            return {**prediction, **adding_empty_fields}
        else:
            return prediction
    except:
        print("add_required_fields exception :",traceback.print_exc())
        return prediction


def apply_business_rules(DF, prediction, format_):
    DF, prediction = apply_GSTIN_rules(DF, prediction)
    prediction = clean_amounts(prediction)
    prediction = extract_HSNCode(DF, prediction)
    prediction = get_totalGSTAmount(DF,prediction)
    print("after adding total GSTAmount")
    # prediction = validate_gst_amounts(prediction)
    prediction = clean_trailing_punctuations(prediction)
    # prediction = conf_reduction_amounts(DF,prediction)
    # prediction = add_required_fields(DF,prediction)
    # print("prediction after adding empty fields :",prediction)

    # prediction = extract_vendorGSTIN(DF, prediction)
    # try:
    #     prediction = reduce_confidence_LI(prediction)
    # except:
    #     pass
    #
    # if format_ is not None:
    #     prediction = correct_fields(prediction, format_)
    prediction = add_lineitem_dummy_row(prediction, format_)
    return prediction
