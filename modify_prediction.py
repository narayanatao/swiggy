# import pandas as pd
import traceback
import re
import math
from xml.sax.handler import property_interning_dict
import preProcUtilities as putil
from datetime import datetime
from dateutil import parser

uom_val = ["ea","each","piece","pieces","pc","pcs","pce","kg","kgs",
           "tonne","tonnes","tons","box","boxes","set","sets","no","nos",
           "unit","units","bags","bag","pkt","pkts","packet","packets",
           "kilogram","number","pl","pak","pkg","e"]

# In[amount patterns]
ptn1 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn2 = "[0-9]{1,3}[,]{1}[0-9]{3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn3 = "[0-9]{1,3}[,]{1}[0-9]{3}[.]{1}[0-9]{1,4}"
ptn4 = "[0-9]{1,3}[.]{1}[0-9]{1,4}"

ptn5 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn6 = "[0-9]{1,3}[.]{1}[0-9]{3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn7 = "[0-9]{1,3}[.]{1}[0-9]{3}[,]{1}[0-9]{1,4}"
ptn8 = "[0-9]{1,3}[,]{1}[0-9]{1,4}"
ptn9 = "\d{1,12}"

ptns = [ptn1,ptn2,ptn3,ptn4,ptn5,ptn6,ptn7,ptn8,ptn9]
from price_parser import parse_price

def isAmount(s):
    try:
        for ptn in ptns:
            l = re.findall(ptn,s)
            l1 = [g for g in l if len(g) > 0]
            if len(l1) >= 1:
                return True
    except:
        return False
    return False

def extract_amount(text):
    """
    Checks whether passed string is valid amount or not
    Returns: 1 if amount, 0 otherwise
    """
    try:
        text = str(text)
        index_last_dot = text.rfind(".")
        if index_last_dot != -1:
            text = text.replace(".", ",")
            text = list(text)
            text[index_last_dot] = "."
            text = "".join(text)
            if (len(text) - index_last_dot) == 4:
                text = list(text)
                text.append("0")
                text = "".join(text)
                    
        if isAmount(text):
            p = parse_price(text)
            if p.amount is not None:
                if isinstance(p.amount_float, float):
                    return p.amount_float
                else:    
                    return "non parseable"
    except:
        return "non parseable"
    return "non parseable"

# In[Line Item predictions]

@putil.timing
def refine_LI_df(df):
    '''
    Modify prediction dataframe coming from model output
    Return: pandas dataframe
    '''

    df_copy = df.copy(deep = True)
    try:
        filt = df[df["line_row"] > 0]

        # print("filter", filt.shape)
        pred_label = {}
        pred_prob = {}
        if filt.shape[0] > 0:
            for row_ind, row in filt.iterrows():
                text = row["text"]
                # is_alnum = text.isalnum()
                # is_amount = row["is_amount"]
                is_number = row["is_number"]
                token_id = row["token_id"]
                is_amt = text.replace(" ","").replace(",","").replace(".","").isnumeric()

                is_item_desc = row["is_item_desc"]
                is_item_code1 = row["is_item_code1"]
                is_item_val1 = row["is_item_val1"]
                is_qty1 = row["is_qty1"]
                is_unit_price1 = row["is_unit_price1"]
                is_unit_price = row["is_unit_price"]
                is_uom1 = row["is_uom1"]
                is_uom = row["is_uom"]
                is_hsn_key1 = row["is_hsn_key1"]
                is_cgst1 = row["is_cgst1"]
                is_sgst1 = row["is_sgst1"]
                is_igst1 = row["is_igst1"]
                is_disc1 = row["is_disc1"]
                LI_vars = {
                    "is_item_desc":is_item_desc,
                    "is_item_code1":is_item_code1,
                    "is_item_val1":is_item_val1,
                    "is_qty1":is_qty1,
                    "is_unit_price1":is_unit_price1,
                    "is_uom1":is_uom1,
                    "is_hsn_key1":is_hsn_key1,
                    "is_cgst1":is_cgst1,
                    "is_sgst1":is_sgst1,
                    "is_igst1":is_igst1,
                    "is_disc1":is_disc1
                    }

                #ItemDesc
                no_itmdesc = tuple(v for k,v in LI_vars.items() if k != "is_item_desc")
                itmdesc_non_max = max(no_itmdesc)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                itmdesc_cond = is_item_desc > itmdesc_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                itm_desc = (is_item_desc >= 0.70) and itmdesc_cond

                #ItemCode
                no_itmcode1 = tuple(v for k,v in LI_vars.items() if k != "is_item_code1")
                itmcode_non_max = max(no_itmcode1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                itmcode_cond = is_item_code1 > itmcode_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                itm_code = (is_item_code1 >= 0.70) and itmcode_cond

                #ItemValue
                no_itmval1 = tuple(v for k,v in LI_vars.items() if k != "is_item_val1")
                itmval_non_max =  max(no_itmval1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                itmval_cond = is_item_val1 > itmval_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                itm_val = (is_item_val1 >= 0.70) and itmval_cond
                # itm_val = itmval_cond

                #unitPrice
                no_unitprice1 = tuple(v for k,v in LI_vars.items() if k != "is_unit_price1")
                untprice_non_max = max(no_unitprice1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                untprice_cond = ((is_unit_price1 - untprice_non_max) > 0.01)
                untprice_cond = untprice_cond or (is_unit_price > is_uom)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                unit_price = (is_unit_price1 >= 0.80) and untprice_cond
                # unit_price = untprice_cond

                #itemQuantity
                no_qty1 = tuple(v for k,v in LI_vars.items() if k != "is_qty1")
                itmqty_non_max = max(no_qty1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                itmqty_cond = is_qty1 > itmqty_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                item_qty = (is_qty1 >= 0.70) and itmqty_cond

                #itemUOM
                no_uom1 = tuple(v for k,v in LI_vars.items() if k != "is_uom1")
                uom_non_max = max(no_uom1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                uom_cond = is_uom1 > uom_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                item_uom = (is_uom1 >= 0.70) and uom_cond
                # item_uom = uom_cond

                #HSN Code
                no_hsncode1 = tuple(v for k,v in LI_vars.items() if k != "is_hsn_key1")
                HSN_non_max = max(no_hsncode1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                HSN_cond = is_hsn_key1 > HSN_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                hsn_code = (is_hsn_key1 >= 0.70) and HSN_cond

                #CGST Amount
                no_cgst1 = tuple(v for k,v in LI_vars.items() if k != "is_cgst1")
                CGST_non_max = max(no_cgst1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                CGST_cond = is_cgst1 > CGST_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                cgst_amt = (is_cgst1 >= 0.70) and CGST_cond

                #SGST Amount
                no_sgst1 = tuple(v for k,v in LI_vars.items() if k != "is_sgst1")
                SGST_non_max = max(no_sgst1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                SGST_cond = is_sgst1 > SGST_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                sgst_amt = (is_sgst1 >= 0.70) and SGST_cond
                # print(text,sgst_amt,is_sgst1,SGST_non_max)
                # sgst_amt = SGST_cond

                #IGST AMOUNT
                no_igst1 = tuple(v for k,v in LI_vars.items() if k != "is_igst1")
                IGST_non_max = max(no_igst1)
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                IGST_cond = is_igst1 > IGST_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7

                igst_amt = (is_igst1 >= 0.70) and IGST_cond
                # igst_amt = IGST_cond

                #Discount Amount
                no_disc1 = tuple(v for k,v in LI_vars.items() if k != "is_disc1")
                disc_non_max = max(no_disc1)

                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                disc_cond = is_disc1 > disc_non_max
                #Apr 25, 2022 - Cond should be the fuzzy score should be max and above 0.7
                disc_amt = (is_disc1 >= 0.70) and disc_cond

                if itm_desc:
                    # prob_LI_itemDesc = min(prob_LI_itemDesc,0.8)
                    pred_label[token_id] = "LI_itemDescription"
                    pred_prob[token_id] = 1.0
                    continue

                if itm_code:
                    print("Item Code",text)
                    is_itmCode_format = text.replace("-","").replace("/","").replace(".","").isalnum()
                    # prob_LI_itemCode = min(prob_LI_itemCode,0.8)
                    if is_itmCode_format and len(text) >= 4:
                        # print("Item Value replaced",text)
                        pred_label[token_id] = "LI_itemCode"
                        pred_prob[token_id] = 1.0
                        continue

                if itm_val:
                    # print("Item Val",text)
                    if is_amt:
                        # print("Item Value replaced",text)
                        pred_label[token_id] = "LI_itemValue"
                        pred_prob[token_id] = 1.0
                        continue

                if unit_price:
                    # print("unit price",text,predict_label)
                    # print("Unit Price replacement",text,is_amt)
                    if is_amt:
                        pred_label[token_id] = "LI_unitPrice"
                        pred_prob[token_id] = 1.0
                        continue

                if (item_uom or item_qty):
                    # print("UOM",text,predict_label)
                    text = row["text"]
                    l_text = text.lower()
                    l_text = "".join([txt for txt in l_text if txt.isalpha()])
                    is_alpha = l_text.isalpha()
                    is_uom_val = l_text in uom_val
                    # prob_LI_UOM = min(prob_LI_UOM,0.8)
                    # print("is UOM value",text,is_uom_val,is_alpha)
                    if is_alpha and is_uom_val:
                        # print("UOM assigned",text)
                        pred_label[token_id] = "LI_UOM"
                        pred_prob[token_id] = 1.0
                        continue

                if hsn_code:
                    # print("HSN",text,predict_label)
                    text = row["text"]
                    len_text = len(text)
                    hsn_cond = ((len_text == 4) or (len_text == 8) or (len_text == 6)) and is_number
                    # prob_LI_HSNCode = min(prob_LI_HSNCode,0.8)
                    if hsn_cond:
                        pred_label[token_id] = "LI_HSNCode"
                        pred_prob[token_id] = 1.0
                        continue

                if cgst_amt:
                    # print("cgst",text,predict_label,is_amt)
                    # prob_LI_CGSTAmount = min(prob_LI_CGSTAmount,0.8)
                    if is_amt:
                        pred_label[token_id] = "LI_CGSTAmount"
                        pred_prob[token_id] = 1.0
                        continue

                if sgst_amt:
                    # print("sgst",text,predict_label)
                    # prob_LI_SGSTAmount = min(prob_LI_SGSTAmount,0.8)
                    if is_amt:
                        pred_label[token_id] = "LI_SGSTAmount"
                        pred_prob[token_id] = 1.0
                        continue

                if igst_amt:
                    # print("igst",text,predict_label)
                    # prob_LI_IGSTAmount = min(prob_LI_IGSTAmount,0.8)
                    if is_amt:
                        pred_label[token_id] = "LI_IGSTAmount"
                        pred_prob[token_id] = 1.0
                        continue

                if item_qty:
                    # print("item qty",text,predict_label)
                    # prob_LI_itemQuantity = min(prob_LI_itemQuantity,0.8)
                    qty_cond = is_amt or is_number
                    if qty_cond:
                        pred_label[token_id] = "LI_itemQuantity"
                        pred_prob[token_id] = 1.0
                        continue

                if disc_amt:
                    # print("item qty",text,predict_label)
                    # prob_LI_itemQuantity = min(prob_LI_itemQuantity,0.8)
                    disc_cond = is_amt or is_number
                    if disc_cond:
                        pred_label[token_id] = "LI_discountAmount"
                        pred_prob[token_id] = 1.0
                        continue

        df = assignVavluesToDf("predict_label", pred_label, df)
        df = assignVavluesToDf("prediction_probability", pred_prob, df)
        # import numpy as np
        # df["predict_label_new"] = df["token_id"].map(pred_label)
        # df["prediction_probability_new"] = df["token_id"].map(pred_prob)
        # df["predict_label"] = np.where(df["predict_label_new"].isnull(),
        #                                df["predict_label"],
        #                                df["predict_label_new"])
        # df["prediction_probability"] = np.where(df["prediction_probability_new"].isnull(),
        #                                df["prediction_probability"],
        #                                df["prediction_probability_new"])

        return df
    except:
        print("modify dataframe",traceback.print_exc())
        return df_copy


def calculate_lines(df):
    '''
    Recalculate item quantity/ item value/ unit price
    Return: pandas dataframe
    '''

    df_copy = df.copy(deep = True)
    try:
        filt = df[df["line_row"] > 0]
        fields = ['LI_itemQuantity','LI_itemValue','LI_unitPrice']
        if filt.shape[0] > 0:
            line_rows = set(list(df['line_row'].dropna()))
            for line_val in line_rows:
                    TEMP = filt[filt['line_row']== line_val]
                    qty = []
                    price = []
                    val = []

                    # print(TEMP[['text','predict_label', 'prediction_probability']])
                    # print(TEMP.loc[TEMP['predict_label'].isin(fields)][['text',
                    #             'predict_label', 
                    #             'prediction_probability']])
                    for index,row in TEMP.iterrows():
                        if row['predict_label'] == 'LI_unitPrice':
                            price.append(extract_amount(row['text']))
                        elif row['predict_label'] == 'LI_itemQuantity':
                            qty.append(extract_amount(row['text']))
                        elif row['predict_label'] == 'LI_itemValue':
                            val.append(extract_amount(row['text']))

                    qty = [x for x in qty if (x!="non parseable") and (x != 0.0)]
                    price = [x for x in price if (x!="non parseable") and (x != 0.0)]
                    val = [x for x in val if (x!="non parseable") and (x != 0.0)]

                    # print("Table line No {}: \nQty: {}\nprice: {}\nVal: {}".format(line_val,
                    #                     qty, price, val))

                    ocr_values = TEMP['text'].to_list()
                    ocr_map = {extract_amount(i):i for i in ocr_values}
                    ocr_amounts = [extract_amount(i) for i in ocr_values]
                    ocr_amounts = [x for x in ocr_amounts if (x!="non parseable") and (x != 0.0)]


                    if not price:
                        for i in qty:
                            for j in val:
                                calc_price = j/i
                                if any([math.isclose(calc_price, p, rel_tol=0.01) for p in ocr_amounts]):
                                    ocr_price = ocr_map.get(calc_price)
                                    # print(ocr_price)
                                    if ocr_price:
                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['text'] == ocr_price),
                                        ['predict_label']] = "LI_unitPrice"

                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['predict_label'] == 'LI_unitPrice') & 
                                        (df['text'] == ocr_price),
                                        ['prediction_probability']] = 1

                    if not qty:
                        for i in price:
                            for j in val:
                                calc_qty = j/i
                                if any([math.isclose(calc_qty, p, rel_tol=0.01) for p in ocr_amounts]):
                                    ocr_qty = ocr_map.get(calc_qty)
                                    # print(ocr_qty)
                                    if ocr_qty:
                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['text'] == ocr_qty),
                                        ['predict_label']] = "LI_itemQuantity"

                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['predict_label'] == 'LI_itemQuantity') & 
                                        (df['text'] == ocr_qty),
                                        ['prediction_probability']] = 1
                    if not val:
                        for i in qty:
                            for j in price:
                                calc_val = j*i
                                if any([math.isclose(calc_val, p, rel_tol=0.01) for p in ocr_amounts]):
                                    ocr_val = ocr_map.get(calc_val)
                                    # print(ocr_val)
                                    if ocr_val:
                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['text'] == ocr_val),
                                        ['predict_label']] = "LI_itemValue"

                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['predict_label'] == 'LI_itemValue') & 
                                        (df['text'] == ocr_val),
                                        ['prediction_probability']] = 1
                    

        return df
    except:
        print("modify dataframe",traceback.print_exc())
        return df_copy

def revamp_item_amounts(df):
    '''
    Recalculate item quantity/ item value/ unit price
    Return: pandas dataframe
    '''

    df_copy = df.copy(deep = True)
    try:
        filt = df[df["line_row"] > 0]
        if filt.shape[0] > 0:
            line_rows = set(list(df['line_row'].dropna()))
            for line_val in line_rows:
                TEMP = filt[filt['line_row']== line_val]
                qty = []
                price = []
                val = []
                price_map = {}
                qty_map = {}
                val_map = {}

                for index,row in TEMP.iterrows():
                    if row['predict_label'] == 'LI_unitPrice':
                        price.append(extract_amount(row['text']))
                        price_map = {**price_map, **{extract_amount(row['text']):row['text']}}
                    elif row['predict_label'] == 'LI_itemQuantity':
                        qty.append(extract_amount(row['text']))
                        qty_map = {**qty_map, **{extract_amount(row['text']):row['text']}}
                    elif row['predict_label'] == 'LI_itemValue':
                        val.append(extract_amount(row['text']))
                        val_map = {**val_map, **{extract_amount(row['text']):row['text']}}
                qty = [x for x in qty if (x!="non parseable") and (x != 0.0)]
                price = [x for x in price if (x!="non parseable") and (x != 0.0)]
                val = [x for x in val if (x!="non parseable") and (x != 0.0)]

                if len(qty) >= 1:
                    for i in price:
                        if i !=0:
                            for j in val:
                                exp_qty = j/i
                                if any([math.isclose(exp_qty, p, rel_tol=0.01) for p in qty]):
                                    qty_exists = val_map.get(exp_qty)
                                    if qty_exists:
                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['predict_label'] == 'LI_itemQuantity') & 
                                        (df['text'] == qty_exists),
                                        ['prediction_probability']] = 1

                if len(price) >= 1:
                    for i in qty:
                        if i!=0:
                            for j in val:
                                exp_price = j/i
                                if any([math.isclose(exp_price, p, rel_tol=0.01) for p in price]):
                                    price_exists = val_map.get(exp_price)
                                    if price_exists:
                                        df.loc[(df['line_row'] ==line_val) & 
                                        (df['predict_label'] == 'LI_unitPrice') & 
                                        (df['text'] == price_exists),
                                        ['prediction_probability']] = 1

                if len(val) >=1:
                    for i in qty:          
                        for j in price:
                            exp_val = j*i 
                            if any([math.isclose(exp_val, p, rel_tol = 0.01) for p in val]):
                                val_exists = val_map.get(exp_val)
                                if val_exists:
                                    df.loc[(df['line_row'] ==line_val) & 
                                    (df['predict_label'] == 'LI_itemValue') & 
                                    (df['text'] == val_exists),
                                    ['prediction_probability']] = 1


        return df
    except:
        print("modify dataframe",traceback.print_exc())
        return df_copy


# In[Predicting header field using fuzzy logic]:

def getHdrFzFields(cols):
    lft_cols = [col for col in cols
                if (col.lower().startswith("fz_")) and 
                (col.lower().endswith("_left")) and
                ("hdr" not in col.lower())]
    ab_cols = [col for col in cols
                if (col.lower().startswith("fz_")) and 
                (col.lower().endswith("_above")) and
                ("hdr" not in col.lower())]
    return lft_cols,ab_cols

def predictPoNumber(df):
    df_copy = df.copy(deep = True)
    try:

        filt = df[df["predict_label"] == "poNumber"]
        if filt.shape[0] == 0:
            df["po_prediction"] = ((df["fz_lblPoNumber_Left"] >= 0.7) & (df["fz_lblPoNumber_Left_rank"] == 1) |
                                   (df["fz_lblPoNumber_Above"] >= 0.7) & (df["fz_lblPoNumber_Above_rank"] == 1))
            df["po_prediction"] = (df["po_prediction"] == True) & (df["predict_label"] == "Unknown")
            df["po_prediction"] = df["po_prediction"] & (df["is_alpha_wo_punct"] == False)
            df["po_prediction"] = df["po_prediction"] & (df["is_nothing"] == False)
            df["po_prediction"] = df["po_prediction"] & (df["is_date"] == 0)
            df.loc[df["po_prediction"] == True,
                   ["predict_label"]] = ["poNumber"]
            df["max_prob_po_unk"] = df[["fz_lblPoNumber_Left",
                                        "fz_lblPoNumber_Above"]].max(axis = 1)
            # print(df["max_prob_po_unk"])
            df["max_prob_po_unk_comp"] = 1 - df["max_prob_po_unk"]
            df.loc[df["po_prediction"] == True,
                   "prob_Unknown"] = df.loc[df["po_prediction"] == True,
                                               "max_prob_po_unk_comp"]
            df.loc[df["po_prediction"] == True,
                   "prob_poNumber"] = df.loc[df["po_prediction"] == True,
                                               "max_prob_po_unk"]
            df.loc[df["po_prediction"] == True,
                   "prediction_probability"] = df.loc[df["po_prediction"] == True,
                                               "max_prob_po_unk"]
        return df
    except:
        print("predictPoNumber",traceback.print_exc())
        return df_copy



def predictInvNumber(df):
    df_copy = df.copy(deep = True)
    label_val = "invoiceNumber"
    tmp_prediction = "inv_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblInvoiceNumber_Left"
    ab_fz = "fz_lblInvoiceNumber_Above"
    lft_fz_1 = "fz_lblInvoiceNumber1_Left"
    ab_fz_1 = "fz_lblInvoiceNumber1_Above"

    lft_fz_rank = "fz_lblInvoiceNumber_Left_rank"
    lft_fz_rank_1 = "fz_lblInvoiceNumber1_Left_rank"
    ab_fz_rank_1 = "fz_lblInvoiceNumber1_Above_rank"
    ab_fz_rank = "fz_lblInvoiceNumber_Above_rank"
    prob_sub = "prob_invoiceNumber"
    prob_unk = "prob_Unknown"
    pred_prob ="prediction_probability"
    max_prob = "max_prob_inv_unk"
    max_comp = "max_prob_inv_unk_comp"
    
   
    try:
        print(pred_prob)
        #filt = df[df["predict_label"] == label_val]
        #if filt.shape[0] == 0:
        left_fz_cond = (df[lft_fz] >= 0.7) & (df[lft_fz_1] >= 0.7)
        left_rk_cond = (df[lft_fz_rank] == 1) | (df[lft_fz_rank_1] == 1)

        ab_fz_cond = (df[ab_fz] >= 0.7) & (df[ab_fz_1] >= 0.7)
        ab_rk_cond = (df[ab_fz_rank] == 1) | (df[ab_fz_rank_1] == 1)
        df[tmp_prediction] = ((left_fz_cond & left_rk_cond) | (ab_fz_cond & ab_rk_cond))
            #df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_alpha_wo_punct"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_alnum_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == 0)
        # df[tmp_prediction] = df[tmp_prediction] & (df["is_amount"] == 0)
        filt = df[df["predict_label"] == label_val]
        inv_token_model=list(filt["token_id"])
        print("Unique booleans",df[tmp_prediction].unique())
        dff=df[df[tmp_prediction]==True]
        if (dff.shape[0]>0):
            dff[max_prob] = dff[[lft_fz,ab_fz]].max(axis = 1)

            maxt = dff[max_prob].max()
            df_1 = dff[dff[max_prob]== maxt]
           
            print(df_1[max_prob].value_counts(),"MMMMM")
            
            for x in df_1.itertuples():
                if x.predict_label=="Unknown":
                    df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                    df = assignVavluesToDf("prediction_probability",{x.token_id:1}, df)
                    df = assignVavluesToDf(prob_sub,{x.token_id:1},df)
                elif x.predict_label=="invoiceNumber":
                    df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                    df = assignVavluesToDf("prediction_probability",{x.token_id:1}, df)
                    df = assignVavluesToDf(prob_sub,{x.token_id:1},df)
                else:
                    df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                    df = assignVavluesToDf("prediction_probability",{x.token_id:0.4}, df)
                    df = assignVavluesToDf(prob_sub,{x.token_id:0.75},df)

            #if df_1.shape[0] == 0:
                #df[tmp_prediction] = ((df[ab_fz] >= 0.7) & (df[ab_fz_1] >= 0.7) ) 
                # #df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)

                # df[tmp_prediction] = df[tmp_prediction] & (df["is_alpha_wo_punct"] == False)
                # df[tmp_prediction] = df[tmp_prediction] & (df["is_alnum_wo_punct"] == True)
                # df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
                # df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == 0)
                # filt = df[df["predict_label"] == label_val]
                # inv_token_model=list(filt["token_id"])
                # print("Unique booleans",df[tmp_prediction].unique())
                # dff=df[df[tmp_prediction]==True]
                # if (dff.shape[0]>0):
                #     dff[max_prob] = dff[[lft_fz,ab_fz]].max(axis = 1)
                #     print(dff[max_prob].unique(),"Para")
                #     maxt = dff[max_prob].max()
                #     df_1 = dff[dff[max_prob]== maxt]
                   
                #     print(df_1[max_prob].value_counts(),"MMMMM")
                #     for x in df_1.itertuples():
                #         if x.predict_label=="Unknown":
                #             df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                #             df = assignVavluesToDf("prediction_probability",{x.token_id:1}, df)
                #             df = assignVavluesToDf(prob_sub,{x.token_id:1},df)
                #         elif x.predict_label=="invoiceNumber":
                #             pass
                #         else:
                #             df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                #             df = assignVavluesToDf("prediction_probability",{x.token_id:0.4}, df)
                #             df = assignVavluesToDf(prob_sub,{x.token_id:0.75},df)

            
            
                # df[max_comp] = 1 - df[max_prob]
                # df[df[pred_prob] == df[pred_prob].max()]
                # df.loc[df[tmp_prediction] == True,
                #         prob_unk] = df.loc[df[tmp_prediction] == True,
                #                             max_comp]
                # df.loc[df[tmp_prediction] == True,
                #         prob_sub] = df.loc[df[tmp_prediction] == True,
                #                             max_prob]

                # x=df[df[max_prob]==df[max_prob].max()]
                # print(x[max_prob].value_counts())
                # df.loc[df[tmp_prediction] == True,
                #         "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                #                                     max_prob]
                
                # df.loc[df[tmp_prediction] == True,["predict_label"]] = [label_val] 
        return df
    except:
        print("predictInvoiceNumber",traceback.print_exc())
        return df_copy


def predictInvDate(df):
    df_copy = df.copy(deep = True)
    label_val = "invoiceDate"
    tmp_prediction = "invdt_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblInvoicedate_Left"
    ab_fz = "fz_lblInvoicedate_Above"
    lft_fz_rank = "fz_lblInvoicedate_Left_rank"
    ab_fz_rank = "fz_lblInvoicedate_Above_rank"
    prob_sub = "prob_invoiceDate"
    prob_unk = "prob_Unknown"
    max_prob = "max_prob_invdt_unk"
    max_comp = "max_prob_invdt_unk_comp"
    try:
        # filt = df[df["predict_label"] == label_val]
        # if filt.shape[0] == 0:
        df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                                (df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1))
        # df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_alpha_wo_punct"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_alnum_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == 1)
        filt_=df[df[tmp_prediction]==True]
        df.loc[df[tmp_prediction] == True,
                ["predict_label"]] = [label_val] 
              
        df[max_prob] = df[[lft_fz,ab_fz]].max(axis = 1)

        # print(df["max_prob_po_unk"])
        df[max_comp] = 1 - df[max_prob]

        df.loc[df[tmp_prediction] == True,
                prob_unk] = df.loc[df[tmp_prediction] == True,
                                    max_comp]
        df.loc[df[tmp_prediction] == True,
                prob_sub] = df.loc[df[tmp_prediction] == True,
                                    max_prob]
        df.loc[df[tmp_prediction] == True,
                "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                                            max_prob]
        return df
    except:
        print("predictInvoiceDate",traceback.print_exc())
        return df_copy

def predictDiscAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "discountAmount"
    tmp_prediction = "discount_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblDiscountAmount_Left"
    ab_fz = "fz_lblDiscountAmount_Above"
    lft_fz_rank = "fz_lblDiscountAmount_Left_rank"
    ab_fz_rank = "fz_lblDiscountAmount_Above_rank"
    prob_sub = "prob_discountAmount"
    prob_unk = "prob_Unknown"
    max_prob = "max_prob_disc_unk"
    max_comp = "max_prob_disc_unk_comp"
    try:
        filt = df[df["predict_label"] == label_val]
        if filt.shape[0] == 0:
            df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                                   (df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1))
            df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
            # df[tmp_prediction] = df[tmp_prediction] & (df["is_amount"] == 1)
            df.loc[df[tmp_prediction] == True,
                   ["predict_label"]] = [label_val]
            df[max_prob] = df[[lft_fz,ab_fz]].max(axis = 1)
            # print(df["max_prob_po_unk"])
            df[max_comp] = 1 - df[max_prob]
            df.loc[df[tmp_prediction] == True,
                   prob_unk] = df.loc[df[tmp_prediction] == True,
                                      max_comp]
            df.loc[df[tmp_prediction] == True,
                   prob_sub] = df.loc[df[tmp_prediction] == True,
                                      max_prob]
            df.loc[df[tmp_prediction] == True,
                   "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                                               max_prob]
        return df
    except:
        print("predictAmt",traceback.print_exc())
        return df_copy


def predictCGSTAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "CGSTAmount"
    tmp_prediction = "cgst_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblCGSTAmount_Left"
    ab_fz = "fz_lblCGSTAmount_Above"
    lft_fz_rank = "fz_lblCGSTAmount_Left_rank"
    ab_fz_rank = "fz_lblCGSTAmount_Above_rank"
    prob_sub = "prob_CGSTAmount"
    prob_unk = "prob_Unknown"
    max_prob = "max_prob_cgst_unk"
    max_comp = "max_prob_cgst_unk_comp"
    try:
        filt = df[df["predict_label"] == label_val]
        if filt.shape[0] == 0:
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            # df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
            #                        ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)) &
            #                        ((df["fz_lblTotalAmount_Left"] > 0.25) & 
            #                         (df["fz_lblTotalAmount_Left_rank"] == 1)))
            df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                                   ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
            #df[tmp_prediction] = df[tmp_prediction] & (df["PROBABLE_GST_AMOUNT_SLAB"] == 1)
            df.loc[df[tmp_prediction] == True,
                   ["predict_label"]] = [label_val]
            df[max_prob] = df[[lft_fz,ab_fz]].max(axis = 1)
            # print(df["max_prob_po_unk"])
            df[max_comp] = 1 - df[max_prob]
            df.loc[df[tmp_prediction] == True,
                   prob_unk] = df.loc[df[tmp_prediction] == True,
                                      max_comp]
            df.loc[df[tmp_prediction] == True,
                   prob_sub] = df.loc[df[tmp_prediction] == True,
                                      max_prob]
            df.loc[df[tmp_prediction] == True,
                   "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                                               max_prob]
    
        return df
    except:
        print("predictCGSTAmt",traceback.print_exc())
        return df_copy

def predictSGSTAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "SGSTAmount"
    tmp_prediction = "sgst_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblSGSTAmount_Left"
    ab_fz = "fz_lblSGSTAmount_Above"
    lft_fz_rank = "fz_lblSGSTAmount_Left_rank"
    ab_fz_rank = "fz_lblSGSTAmount_Above_rank"
    prob_sub = "prob_SGSTAmount"
    prob_unk = "prob_Unknown"
    max_prob = "max_prob_sgst_unk"
    max_comp = "max_prob_sgst_unk_comp"
    try:
        filt = df[df["predict_label"] == label_val]
        if filt.shape[0] == 0:
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            # df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
            #                        ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)) &
            #                        ((df["fz_lblTotalAmount_Left"] > 0.25) & 
            #                         (df["fz_lblTotalAmount_Left_rank"] == 1)))
            df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                                   ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
            #df[tmp_prediction] = df[tmp_prediction] & (df["PROBABLE_GST_AMOUNT_SLAB"] == 1)
            df.loc[df[tmp_prediction] == True,
                   ["predict_label"]] = [label_val]
            df[max_prob] = df[[lft_fz,ab_fz]].max(axis = 1)
            # print(df["max_prob_po_unk"])
            df[max_comp] = 1 - df[max_prob]
            df.loc[df[tmp_prediction] == True,
                   prob_unk] = df.loc[df[tmp_prediction] == True,
                                      max_comp]
            df.loc[df[tmp_prediction] == True,
                   prob_sub] = df.loc[df[tmp_prediction] == True,
                                      max_prob]
            df.loc[df[tmp_prediction] == True,
                   "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                                               max_prob]
        return df
    except:
        print("predictSGSTAmt",traceback.print_exc())
        return df_copy

def predictIGSTAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "IGSTAmount"
    tmp_prediction = "igst_prediction"
    unk_val = "Unknown"
    lft_fz = "fz_lblIGSTAmount_Left"
    ab_fz = "fz_lblIGSTAmount_Above"
    lft_fz_rank = "fz_lblIGSTAmount_Left_rank"
    ab_fz_rank = "fz_lblIGSTAmount_Above_rank"
    prob_sub = "prob_IGSTAmount"
    prob_unk = "prob_Unknown"
    max_prob = "max_prob_igst_unk"
    max_comp = "max_prob_igst_unk_comp"
    try:
        filt = df[df["predict_label"] == label_val]
        if filt.shape[0] == 0:
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            # df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
            #                        ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)) &
            #                        ((df["fz_lblTotalAmount_Left"] > 0.25) & 
            #                         (df["fz_lblTotalAmount_Left_rank"] == 1)))
            df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                                   ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
            #Apr 25, 2022 - total amount condition removed. We can add a separate or condition, if required
            df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
            df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
            #df[tmp_prediction] = df[tmp_prediction] & (df["PROBABLE_GST_AMOUNT_SLAB"] == 1)
            df.loc[df[tmp_prediction] == True,
                   ["predict_label"]] = [label_val]
            df[max_prob] = df[[lft_fz,ab_fz]].max(axis = 1)
            # print(df["max_prob_po_unk"])
            df[max_comp] = 1 - df[max_prob]
            df.loc[df[tmp_prediction] == True,
                   prob_unk] = df.loc[df[tmp_prediction] == True,
                                      max_comp]
            df.loc[df[tmp_prediction] == True,
                   prob_sub] = df.loc[df[tmp_prediction] == True,
                                      max_prob]
            df.loc[df[tmp_prediction] == True,
                   "prediction_probability"] = df.loc[df[tmp_prediction] == True,
                                               max_prob]
        return df
    except:
        print("predictSGSTAmt",traceback.print_exc())
        return df_copy

def assignVavluesToDf(col_name,col_vals,df,
                      base_col = "token_id"):
    import numpy as np
    new_col = col_name + "_new"
    df[new_col] = df[base_col].map(col_vals)
    df[col_name] = np.where(df[new_col].isnull(),
                            df[col_name],
                            df[new_col])
    return df


def predictTotalSubTotal(df):

    df_copy = df.copy(deep = True)
    label_val = "totalAmount"
    tmp_prediction = "total_prediction"
    tot_lft_fz = "fz_lblTotalAmount_Left"
    tot_ab_fz = "fz_lblTotalAmount_Above"
    sub_lft_fz = "fz_lblSubTotal_Left"
    sub_ab_fz = "fz_lblSubTotal_Above"
    rank_fld = "extract_amount_rank"

    try:
        df[tmp_prediction] = (df[rank_fld] <= 5)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] > 0.0)

        DF = df[df[tmp_prediction] == True]
        if DF.shape[0] >= 1:
            DF = DF.sort_values(["extract_amount_rank"],
                                ascending = True)
            DF["lr_subtot"] = DF["extracted_amount"] * .70
            DF["ur_subtot"] = DF["extracted_amount"] * .99
            DF["lr_tot"] = DF["extracted_amount"] * 1.00
            DF["ur_tot"] = DF["extracted_amount"] * 1.3
            DF["pot_subtotal"] = 0
            DF["pot_total"] = 0
            token_ids = list(DF["token_id"])
            amts = list(DF["extracted_amount"])
            lr_tots = list(DF["lr_tot"])
            ur_tots = list(DF["ur_tot"])
            lr_subtots = list(DF["lr_subtot"])
            ur_subtots = list(DF["ur_subtot"])
            tot_lft_fzs = list(DF[tot_lft_fz])
            tot_ab_fzs = list(DF[tot_ab_fz])
            sub_lft_fzs = list(DF[sub_lft_fz])
            sub_ab_fzs = list(DF[sub_ab_fz])
            from itertools import combinations
            z = zip(token_ids,
                    amts,
                    lr_tots,ur_tots,
                    lr_subtots,ur_subtots,
                    tot_lft_fzs,tot_ab_fzs,
                    sub_lft_fzs,sub_ab_fzs)
            pairs = combinations(z,2)
            token_pairs = []
            for pair in pairs:
                token_id_0 = pair[0][0]
                token_id_1 = pair[1][0]
                amt_0 = pair[0][1]
                amt_1 = pair[1][1]
                lr_tot_0 = pair[0][2]
                lr_tot_1 = pair[1][2]
                ur_tot_0 = pair[0][3]
                ur_tot_1 = pair[1][3]
                lr_sub_0 = pair[0][4]
                lr_sub_1 = pair[1][4]
                ur_sub_0 = pair[0][5]
                ur_sub_1 = pair[1][5]
                tot_lft_fz_0 = pair[0][6]
                tot_lft_fz_1 = pair[1][6]
                tot_ab_fz_0 = pair[0][7]
                tot_ab_fz_1 = pair[1][7]
                sub_lft_fz_0 = pair[0][8]
                sub_lft_fz_1 = pair[1][8]
                sub_ab_fz_0 = pair[0][9]
                sub_ab_fz_1 = pair[1][9]
                # if amt_0


                


            tkn_id = list(DF["token_id"])[0]
            pred_label = {tkn_id:label_val}
            prob = {tkn_id:1.0}
            #Overrides an existing prediction
            df = assignVavluesToDf("predict_label",
                                   pred_label, df)
            df = assignVavluesToDf("prediction_probability",
                                   prob, df)
            df = assignVavluesToDf("prob_totalAmount",prob,df)
            filt_list = list(df[(df["predict_label"]=="totalAmount") & (df["token_id"]!=tkn_id)]["token_id"])
            if len(filt_list)>0:
                prob_1 = {m:0.2 for m in filt_list}
                prob_2 ={m:"Unknown" for m in filt_list}
                df= assignVavluesToDf("prediction_probability",
                                   prob_1, df)
                df = assignVavluesToDf("prob_totalAmount",prob_1,df)
                df = assignVavluesToDf("predict_label",prob_2,df)

        return df
    except:
        print("predictTotalAmt",
              traceback.print_exc())
        return df_copy

def predictCGSTAmt_New(df):
    df_copy = df.copy(deep = True)
    label_val = "CGSTAmount"
    tmp_prediction = "cgst_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblCGSTAmount_Left"
    ab_fz = "fz_lblCGSTAmount_Above"
    is_LI = "line_row"
    left_fz_cols, ab_fz_cols = getHdrFzFields(list(df.columns.values))
    left_fz_cols = [left_fz_col for left_fz_col in left_fz_cols if lft_fz != left_fz_col]
    ab_fz_cols = [ab_fz_col for ab_fz_col in ab_fz_cols if ab_fz != ab_fz_cols]
    prob_sub = "prob_CGSTAmount"
    prob_gst_slab = "PROBABLE_GST_AMOUNT_SLAB"
    try:
        df["left_max_fz"] = df[left_fz_cols].max(axis = 1)
        df["above_max_fz"] = df[ab_fz_cols].max(axis = 1)
      
        df[tmp_prediction] = (((df[lft_fz] >= 0.7) | (df[ab_fz] >= 0.7)))
        df[tmp_prediction] = df[tmp_prediction] & (df[is_LI] == 0) & (df["Percentage"]!=1)


        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] != 0)

        

        df[tmp_prediction] = (df[tmp_prediction] & 
                              ((df[lft_fz] > df["left_max_fz"]) |
                               (df[ab_fz] > df["above_max_fz"])))
       
        
        DF = df[df[tmp_prediction] == True]
        DF=DF[DF.text == DF.text.max()]
        token_ids = list(DF["token_id"])
        if len(token_ids) >= 0:
            pred_lbl = {token_id:label_val for token_id in token_ids}
            pred_prob = {token_id:1.0 for token_id in token_ids}
            df = assignVavluesToDf("predict_label", pred_lbl, df)
            df = assignVavluesToDf("prediction_probability",
                                   pred_prob, df)
            df = assignVavluesToDf(prob_sub,
                                   pred_prob, df)
        



            #To make ML prediction of CGSTAmount as null
            # for token_id in token_ids:
            #     df.loc[(df["token_id"] != token_id) &
            #            (df["CG"])]

        return df
    except:
        print("predictCGSTAmt",traceback.print_exc())
        return df_copy

def predictSGSTAmt_New(df):
    df_copy = df.copy(deep = True)
    label_val = "SGSTAmount"
    tmp_prediction = "sgst_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblSGSTAmount_Left"
    ab_fz = "fz_lblSGSTAmount_Above"
    is_LI = "line_row"
    left_fz_cols, ab_fz_cols = getHdrFzFields(list(df.columns.values))
    left_fz_cols = [left_fz_col for left_fz_col in left_fz_cols if lft_fz != left_fz_col]
    ab_fz_cols = [ab_fz_col for ab_fz_col in ab_fz_cols if ab_fz != ab_fz_cols]
    prob_sub = "prob_SGSTAmount"
    prob_gst_slab = "PROBABLE_GST_AMOUNT_SLAB"
    try:
        df["left_max_fz"] = df[left_fz_cols].max(axis = 1)
        df["above_max_fz"] = df[ab_fz_cols].max(axis = 1)
        df[tmp_prediction] = ((df[lft_fz] >= 0.7) | (df[ab_fz] >= 0.7))
        df[tmp_prediction] = df[tmp_prediction] & (df[is_LI] == 0) & (df["Percentage"]!=1)

        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] != 0)
        df[tmp_prediction] = (df[tmp_prediction] & 
                              ((df[lft_fz] > df["left_max_fz"]) |
                               (df[ab_fz] > df["above_max_fz"])))
        DF = df[df[tmp_prediction] == True]
        DF=DF[DF.text == DF.text.max()]
        token_ids = list(DF["token_id"])
        if len(token_ids) >= 0:
            pred_lbl = {token_id:label_val for token_id in token_ids}
            pred_prob = {token_id:1.0 for token_id in token_ids}
            df = assignVavluesToDf("predict_label", pred_lbl, df)
            df = assignVavluesToDf("prediction_probability",
                                   pred_prob, df)
            df = assignVavluesToDf(prob_sub,
                                   pred_prob, df)
        

            #To make ML prediction of CGSTAmount as null
            # for token_id in token_ids:
            #     df.loc[(df["token_id"] != token_id) &
            #            (df["CG"])]

        return df
    except:
        print("predictSGSTAmt",traceback.print_exc())
        return df_copy

def predictIGSTAmt_New(df):
    df_copy = df.copy(deep = True)
    label_val = "IGSTAmount"
    tmp_prediction = "igst_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblIGSTAmount_Left"
    ab_fz = "fz_lblIGSTAmount_Above"
    is_LI = "line_row"
    left_fz_cols, ab_fz_cols = getHdrFzFields(list(df.columns.values))
    left_fz_cols = [left_fz_col for left_fz_col in left_fz_cols if lft_fz != left_fz_col]
    ab_fz_cols = [ab_fz_col for ab_fz_col in ab_fz_cols if ab_fz != ab_fz_cols]
    prob_sub = "prob_IGSTAmount"
    prob_gst_slab = "PROBABLE_GST_AMOUNT_SLAB"
    try:
        df["left_max_fz"] = df[left_fz_cols].max(axis = 1)
        df["above_max_fz"] = df[ab_fz_cols].max(axis = 1)
        df[tmp_prediction] = ((df[lft_fz] >= 0.7) | (df[ab_fz] >= 0.7))
        df[tmp_prediction] = df[tmp_prediction] & (df[is_LI] == 0) & (df["Percentage"]!=1)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] != 0)
        df[tmp_prediction] = (df[tmp_prediction] & 
                              ((df[lft_fz] > df["left_max_fz"]) |
                               (df[ab_fz] > df["above_max_fz"])))
        DF = df[df[tmp_prediction] == True]
        DF=DF[DF.text == DF.text.max()]
        token_ids = list(DF["token_id"])
        if len(token_ids) >= 0:
            pred_lbl = {token_id:label_val for token_id in token_ids}
            pred_prob = {token_id:1.0 for token_id in token_ids}
            df = assignVavluesToDf("predict_label", pred_lbl, df)
            df = assignVavluesToDf("prediction_probability",
                                   pred_prob, df)
            df = assignVavluesToDf(prob_sub,
                                   pred_prob, df)


            #To make ML prediction of CGSTAmount as null
            # for token_id in token_ids:
            #     df.loc[(df["token_id"] != token_id) &
            #            (df["CG"])]

        return df
    except:
        print("predictCGSTAmt",traceback.print_exc())
        return df_copy

def cleanHdrAmt(df):

    df_copy = df.copy(deep = True)
    try:
        token_ids = df[((df["predict_label"] == "SGSTAmount") | 
                        (df["predict_label"] == "CGSTAmount") |
                        (df["predict_label"] == "IGSTAmount") |
                        (df["predict_label"] == "totalAmount") |
                        (df["predict_label"] == "subTotal") |
                        (df["predict_label"] == "freightAmount") |
                        (df["predict_label"] == "discountAmount")) &
                       (df["line_row"] > 0)]["token_id"] 
             
        LI_token_ids =df[df["line_row"]>0]["token_id"]
        pred_lbl = {token_id:"Unknown" for token_id in token_ids}
        pred_prob = {token_id:1 for token_id in token_ids}
        field_prob = {token_id:0 for token_id in token_ids}
        field_hdr = {token_id:0 for token_id in LI_token_ids}

        df = assignVavluesToDf("predict_label",
                               pred_lbl, df)
        df = assignVavluesToDf("prediction_probability",
                               pred_prob, df)
        df = assignVavluesToDf("prob_Unknown",
                               pred_prob, df)
        df = assignVavluesToDf("prob_SGSTAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_CGSTAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_IGSTAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_totalAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_subTotal",
                               field_prob, df)
        df = assignVavluesToDf("prob_freightAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_discountAmount",
                               field_prob, df)
        df = assignVavluesToDf("prob_SGSTAmount",
                               field_hdr, df)
        df = assignVavluesToDf("prob_CGSTAmount",
                               field_hdr, df)
        df = assignVavluesToDf("prob_IGSTAmount",
                               field_hdr, df)
        df = assignVavluesToDf("prob_totalAmount",
                               field_hdr, df)
        df = assignVavluesToDf("prob_subTotal",
                               field_hdr, df)
        df = assignVavluesToDf("prob_freightAmount",
                               field_hdr, df)
        df = assignVavluesToDf("prob_discountAmount",
                               field_hdr, df)



        return df
    except:
        print("cleanHdrAmt",
              traceback.print_exc())
        return df_copy
        

def predictTotalAmt(df):

    df_copy = df.copy(deep = True)
    label_val = "totalAmount"
    tmp_prediction = "total_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblTotalAmount_Left"
    ab_fz = "fz_lblTotalAmount_Above"

    #May 17 2022 - Added Rank of the Fuzzy score manually
    left_fz_cols, ab_fz_cols = getHdrFzFields(list(df.columns.values))
    left_fz_cols = [left_fz_col for left_fz_col in left_fz_cols if lft_fz != left_fz_col]
    ab_fz_cols = [ab_fz_col for ab_fz_col in ab_fz_cols if ab_fz != ab_fz_cols]
    #May 17 2022 - Added Rank of the Fuzzy score manually

    #May 08 2022 - Changed condition to fuzzy plus first_maxt
    # lft_fz_rank = "fz_lblTotalAmount_Left_rank"
    # ab_fz_rank = "fz_lblTotalAmount_Above_rank"
    lft_fz_rank = "extract_amount_rank"
    ab_fz_rank = "extract_amount_rank"
    #May 08 2022 - Changed condition to fuzzy plus first_maxt
    try:

        #May 17 2022 - Added Rank of the Fuzzy score manually
        df["left_max_fz"] = df[left_fz_cols].max(axis = 1)
        df["above_max_fz"] = df[ab_fz_cols].max(axis = 1)
        #May 17 2022 - Added Rank of the Fuzzy score manually

        #Condition - THe fuzzy score should be the best and it should be an amount
        #May 08 2022 - Changed condition to fuzzy plus first_maxt
        df[tmp_prediction] = ((df[lft_fz] >= 0.85) & (df[lft_fz_rank] <= 3) |
                               ((df[ab_fz] >= 0.85) & (df[ab_fz_rank] <= 3)))
        #May 08 2022 - Changed condition to fuzzy plus first_maxt
        df[tmp_prediction] = df[tmp_prediction] & (df["line_row"] == 0)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        #May 08 2022 - Check if extracted amount is greater than 0
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] > 0.0)
        #May 08 2022 - Check if extracted amount is greater than 0
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == False)


        #df.to_csv("lastbut1.csv")
        #May 17 2022 - Added Rank of the Fuzzy score manually
        df[tmp_prediction] = (df[tmp_prediction] & 
                              ((df[lft_fz] > df["left_max_fz"]) |
                               (df[ab_fz] > df["above_max_fz"])))
        #May 17 2022 - Added Rank of the Fuzzy score manually
    
        DF = df[df[tmp_prediction] == True]
        if DF.shape[0] >= 1:
            #May 17 2022 - Added Rank of the Fuzzy score manually
            max_amount = DF["extracted_amount"].max(axis = 0)
            DF = DF[DF["extracted_amount"] == max_amount]
            #May 17 2022 - Added Rank of the Fuzzy score manually

            tkn_id = list(DF["token_id"])[0]
            pred_label = {tkn_id:label_val}
            prob = {tkn_id:1.0}
            #Overrides an existing prediction
            df = assignVavluesToDf("predict_label",
                                   pred_label, df)
            df = assignVavluesToDf("prediction_probability",
                                   prob, df)
            df = assignVavluesToDf("prob_totalAmount",prob,df)
            filt_list = list(df[(df["predict_label"]=="totalAmount") & (df["token_id"]!=tkn_id)]["token_id"])
            if len(filt_list)>0:
                prob_1 = {m:0.2 for m in filt_list}
                prob_2 ={m:"Unknown" for m in filt_list}
                df= assignVavluesToDf("prediction_probability",
                                   prob_1, df)
                df = assignVavluesToDf("prob_totalAmount",prob_1,df)
                df = assignVavluesToDf("predict_label",prob_2,df)

        return df
    except:
        print("predictTotalAmt",
              traceback.print_exc())
        return df_copy

def predictSubTotalAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "subTotal"
    tmp_prediction = "subtotal_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblSubTotal_Left"
    ab_fz = "fz_lblSubTotal_Above"
    #May 08 2022 - Changed condition to fuzzy plus first_maxt
    lft_fz_rank = "second_max_amount"
    ab_fz_rank = "second_max_amount"
    #May 08 2022 - Changed condition to fuzzy plus first_maxt
    try:
        #May 08 2022 - Changed condition to second_max
        # df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
        #                        ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
        df[tmp_prediction] = ((df[lft_fz_rank] == 1))
        #May 08 2022 - Changed condition to second_max
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        #May 08 2022 - Check if extracted amount is greater than 0
        df[tmp_prediction] = df[tmp_prediction] & (df["extracted_amount"] > 0.0)
        #May 08 2022 - Check if extracted amount is greater than 0
        DF = df[df[tmp_prediction] == True]
        DF=DF[DF.text == DF.text.max()]
   
        if DF.shape[0] >= 1:
            tkn_id = list(DF["token_id"])[0]
            pred_label = {tkn_id:label_val}
            prob = {tkn_id:1.0}
            #Overrides an existing prediction
            df = assignVavluesToDf("predict_label",
                                   pred_label, df)
            df = assignVavluesToDf("prediction_probability",
                                   prob, df)
            df = assignVavluesToDf("prob_subTotal",prob,df)
            filt_list = list(df[(df["predict_label"]=="subTotal") & (df["token_id"]!=tkn_id)]["token_id"])
            if len(filt_list)>0:
                prob_1 = {m:0.2 for m in filt_list}
                prob_2 ={m:"Unknown" for m in filt_list}
                df= assignVavluesToDf("prediction_probability",
                                   prob_1, df)
                df = assignVavluesToDf("prob_subTotal",prob_1,df)
                df = assignVavluesToDf("predict_label",prob_2,df)
        return df
    except:
        print("predictSubTotalAmt",traceback.print_exc())
        return df_copy


# def predictCessAmt(df):
#     df_copy = df.copy(deep = True)
#     label_val = "CessAmount"
#     tmp_prediction = "cess_prediction"
#     # unk_val = "Unknown"
#     lft_fz = "fz_lblCessAmount_Left"
#     ab_fz = "fz_lblCessAmount_Above"
#     lft_fz_rank = "fz_lblCessAmount_Left_rank"
#     ab_fz_rank = "fz_lblCessAmount_Above_rank"
#     try:
#         #Condition - THe fuzzy score should be the best and it should be an amount
#         df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
#                                ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
#         # df[tmp_prediction] = (df[tmp_prediction] == True) #& (df["predict_label"] == unk_val)
#         df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
#         df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
#         DF = df[df[tmp_prediction] == True]
#         if DF.shape[0] == 1:
#             tkn_id = list(DF["token_id"])[0]
#             pred_label = {tkn_id:label_val}
#             prob = {tkn_id:1.0}
#             #Overrides an existing prediction
#             df = assignVavluesToDf("predict_label",
#                                    pred_label, df)
#             df = assignVavluesToDf("prediction_probability",
#                                    prob, df)
#         return df
#     except:
#         print("predictCessAmt",
#               traceback.print_exc())
#         return df_copy

def predictCessAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "CessAmount"
    tmp_prediction = "cess_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblCessAmount_Left"
    ab_fz = "fz_lblCessAmount_Above"
    lft_fz_rank = "fz_lblCessAmount_Left_rank"
    ab_fz_rank = "fz_lblCessAmount_Above_rank"
    lft_fz_1 = "fz_lblCessAmount1_Left"
    ab_fz_1 = "fz_lblCessAmount1_Above"
    lft_fz_rank_1 = "fz_lblCessAmount1_Left_rank"
    ab_fz_rank_1 = "fz_lblCessAmount1_Above_rank"
    prob_sub = "prob_CessAmount"
    prob_unk = "prob_Unknown"
    pred_prob ="prediction_probability"
    max_prob = "max_prob_cess_unk"
    #max_comp = "max_prob_inv_unk_comp"
    
   
    try:
        print(pred_prob)
        #filt = df[df["predict_label"] == label_val]
        #if filt.shape[0] == 0:
        left_fz_cond = (df[lft_fz] >= 0.7) & (df[lft_fz_1] >= 0.7)
        left_rk_cond = (df[lft_fz_rank] == 1) | (df[lft_fz_rank_1] == 1)

        ab_fz_cond = (df[ab_fz] >= 0.7) & (df[ab_fz_1] >= 0.7)
        ab_rk_cond = (df[ab_fz_rank] == 1) | (df[ab_fz_rank_1] == 1)
        df[tmp_prediction] = ((left_fz_cond & left_rk_cond) | (ab_fz_cond & ab_rk_cond))
            #df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == 0)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_amount"] == 1)
        

        filt = df[df["predict_label"] == label_val]
        inv_token_model=list(filt["token_id"])
        print("Unique booleans",df[tmp_prediction].unique())
        dff=df[df[tmp_prediction]==True]
        if (dff.shape[0]>0):
            dff[max_prob] = dff[[lft_fz_1,ab_fz_1]].max(axis = 1)

            maxt = dff[max_prob].max()
            df_1 = dff[dff[max_prob]== maxt]
           
            print(df_1[max_prob].value_counts(),"MMMMM")
            
            for x in df_1.itertuples():
                if x.predict_label=="Unknown":
                    df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                    df = assignVavluesToDf("prediction_probability",{x.token_id:1}, df)
                    df = assignVavluesToDf(prob_sub,{x.token_id:1},df)
                elif x.predict_label=="CessAmount":
                    pass
                else:
                    df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                    df = assignVavluesToDf("prediction_probability",{x.token_id:0.4}, df)
                    df = assignVavluesToDf(prob_sub,{x.token_id:0.75},df)

        return df
    except:
        print("predictCessAmount",traceback.print_exc())
        return df_copy

def predictAddlCessAmt(df):
    df['prob_additionalCessAmount']=0
    df_copy = df.copy(deep = True)
    label_val = "additionalCessAmount"
    tmp_prediction = "AddlCessAmount_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblAddlCessAmount_Left"
    ab_fz = "fz_lblAddlCessAmount_Above"
    lft_fz_rank = "fz_lblAddlCessAmount_Left_rank"
    ab_fz_rank = "fz_lblAddlCessAmount_Above_rank"
    lft_fz_1 = "fz_lblAddlCessAmount1_Left"
    ab_fz_1 = "fz_lblAddlCessAmount1_Above"
    lft_fz_rank_1 = "fz_lblAddlCessAmount1_Left_rank"
    ab_fz_rank_1 = "fz_lblAddlCessAmount1_Above_rank"
    prob_unk = "prob_Unknown"
    pred_prob ="prediction_probability"
    max_prob = "max_prob_AddlCessAmount_unk"
    prob_sub = "prob_additionalCessAmount"
    #max_comp = "max_prob_inv_unk_comp"
    
   
    try:
        print(pred_prob)
        #filt = df[df["predict_label"] == label_val]
        #if filt.shape[0] == 0:
        left_fz_cond = (df[lft_fz] >= 0.7) & (df[lft_fz_1] >= 0.7)
        left_rk_cond = (df[lft_fz_rank] == 1) | (df[lft_fz_rank_1] == 1)

        ab_fz_cond = (df[ab_fz] >= 0.7) & (df[ab_fz_1] >= 0.7)
        ab_rk_cond = (df[ab_fz_rank] == 1) | (df[ab_fz_rank_1] == 1)
        df[tmp_prediction] = ((left_fz_cond & left_rk_cond) | (ab_fz_cond & ab_rk_cond))
            #df[tmp_prediction] = (df[tmp_prediction] == True) & (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_date"] == 0)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_amount"] == 1)
        

        filt = df[df["predict_label"] == label_val]
        print("model predict label count ;",filt.shape)
        print("Unique booleans",df[tmp_prediction].unique())
        dff=df[df[tmp_prediction]==True]
        if (dff.shape[0]>0):
            dff[max_prob] = dff[[lft_fz_1,ab_fz_1]].max(axis = 1)

            maxt = dff[max_prob].max()
            df_1 = dff[dff[max_prob]== maxt]
           
            print(df_1[max_prob].value_counts(),"MMMMM")
            # df_1.to_csv("addcess_test.csv")
            AddcessKeys = ["add cess","addl cess","addcess","addlcess", "additional cess","additionalcess"]
            for x in df_1.itertuples():
                for key_words in AddcessKeys:
                    if key_words.lower() in  str(x.left_processed_ngbr).lower():
                        print("inside keyword match",key_words,x.left_processed_ngbr)
                        if x.predict_label=="Unknown":
                            df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                            df = assignVavluesToDf("prediction_probability",{x.token_id:1}, df)
                            df = assignVavluesToDf(prob_sub,{x.token_id:1},df)

                        elif x.predict_label=="AddlCessAmount":
                            return df
                        else:
                            df = assignVavluesToDf("predict_label",{x.token_id:label_val} ,df)
                            df = assignVavluesToDf("prediction_probability",{x.token_id:0.4}, df)
                            df = assignVavluesToDf(prob_sub,{x.token_id:1},df)
        return df
    except:
        print("predictAddlCessAmount",traceback.print_exc())
        return df_copy



    

def predictTCS(df):
    df_copy = df.copy(deep = True)
    label_val = "TCSAmount"
    tmp_prediction = "TCS_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblTCSAmount_Left"
    ab_fz = "fz_lblTCSAmount_Above"
    lft_fz_rank = "fz_lblTCSAmount_Left_rank"
    ab_fz_rank = "fz_lblTCSAmount_Above_rank"
    try:
        #Condition - THe fuzzy score should be the best and it should be an amount
        df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                               ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
        # df[tmp_prediction] = (df[tmp_prediction] == True) #& (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        DF = df[df[tmp_prediction] == True]
        if DF.shape[0] == 1:
            tkn_id = list(DF["token_id"])[0]
            pred_label = {tkn_id:label_val}
            prob = {tkn_id:1.0}
            #Overrides an existing prediction
            df = assignVavluesToDf("predict_label",
                                   pred_label, df)
            df = assignVavluesToDf("prediction_probability",
                                   prob, df)
        return df
    except:
        print("predictTCSAmount",
              traceback.print_exc())
        return df_copy


def predictfreightAmt(df):
    df_copy = df.copy(deep = True)
    label_val = "FreightAmount"
    tmp_prediction = "freigt_prediction"
    # unk_val = "Unknown"
    lft_fz = "fz_lblFreightAmount_Left"
    ab_fz = "fz_lblFreightAmount_Above"
    lft_fz_rank = "fz_lblFreightAmount_Left_rank"
    ab_fz_rank = "fz_lblFreightAmount_Above_rank"
    try:
        #Condition - THe fuzzy score should be the best and it should be an amount
        df[tmp_prediction] = ((df[lft_fz] >= 0.7) & (df[lft_fz_rank] == 1) |
                               ((df[ab_fz] >= 0.7) & (df[ab_fz_rank] == 1)))
        # df[tmp_prediction] = (df[tmp_prediction] == True) #& (df["predict_label"] == unk_val)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_nothing"] == False)
        df[tmp_prediction] = df[tmp_prediction] & (df["is_num_wo_punct"] == True)
        DF = df[df[tmp_prediction] == True]
        if DF.shape[0] == 1:
            tkn_id = list(DF["token_id"])[0]
            pred_label = {tkn_id:label_val}
            prob = {tkn_id:1.0}
            #Overrides an existing prediction
            df = assignVavluesToDf("predict_label",
                                   pred_label, df)
            df = assignVavluesToDf("prediction_probability",
                                   prob, df)
        return df
    except:
        print("predictfreightAmt",
              traceback.print_exc())
        return df_copy

def eliminate_rate(df):
    df.loc[(df["is_amount"]==1) & (df["original_text"].str.endswith('%')),"Percentage"]=1
    return df

def extract_max_date(df):
    max_date=None
    token_id=None
    df_copy = df.copy(deep = True)
    #print("Before")
    try:
        #df1=df[df["predict_label"]=="invoiceDate"]
        df1=df[df["is_date"]==1]
        if df1[df1["page_num"]==0].shape[0]>1:
            df1=df1[df1["page_num"]==0]
        elif df1[df1["page_num"]==1].shape[0]>1:
            df1=df1[df1["page_num"]==1]
            #print(df1.shape,"Shape")
        for row in df1.itertuples():
            #print(row.text)
            if ("dated" in str(row.above_processed_ngbr).lower() or  "dated" in str(row.left_processed_ngbr).lower()):
                try:
                    formate_date = parser.parse(row.text, dayfirst=True).date().strftime('%d/%m/%Y')
                except:
                    print("format_exception")
                    pass
                format_date=datetime.strptime(formate_date,'%d/%m/%Y')
                #print(format_date)
                if max_date is None:
                    max_date=format_date
                    token_id=row.token_id
                    #print("first_if",max_date)
                if max_date is not None and max_date<=format_date:
                    max_date=format_date
                    token_id=row.token_id
                    #print("second_if",max_date)
            else:
                print("Dated keyword not present",row.text)
        df = assignVavluesToDf("prediction_probability",{token_id:1}, df)
        df = assignVavluesToDf("prob_invoiceDate",{token_id:1}, df)
        #print(max_date,token_id,"M")

        return df
    except:
        print(print("max_date",traceback.print_exc()))
        return df_copy

@putil.timing
def predHdrUsingFuzzy(df):
    df_copy = df.copy(deep = True)

    try:
        df["chars_wo_punct"] = df["text"].str.replace('[^\w\s]','')
        df["is_alpha_wo_punct"] = df["chars_wo_punct"].str.isalpha()
        df["is_alnum_wo_punct"] = df["chars_wo_punct"].str.isalnum()
        df["is_num_wo_punct"] = df["chars_wo_punct"].str.isnumeric()
        df["is_nothing"] = (df["is_alpha_wo_punct"] == False) & (df["is_alnum_wo_punct"] == False) & (df["is_num_wo_punct"] == False)
        df = predictPoNumber(df)    
        
        df = predictInvNumber(df)
    
        
        
        df = predictInvDate(df)
	  

        

        df=extract_max_date(df)
        

        # df.to_csv("After label change.csv")

        # df=lower_conf_date(df)
        df=eliminate_rate(df)
       
        #May 17, 2022 clear all headerAmount predictions on line items
        df = cleanHdrAmt(df)
        #May 17, 2022 clear all headerAmount predictions on line items

        ## geting prediction of tax amount from table column format
        df = predictTotalAmt(df)
        df, taxable, cgst, sgst, igst, cess, total, disc = get_taxes_amounts_from_table_anchor_column(df)

        
        #df.to_csv("total.csv")
        #df = predictSubTotalAmt(df)
        # #Add for Freight, Cess, TCS here
        # df = predictCGSTAmt(df)
        # df = predictSGSTAmt(df)
        # df = predictIGSTAmt(df)
        if cgst is None :
            df = predictCGSTAmt_New(df)    
           
        if sgst is  None:
            df = predictSGSTAmt_New(df)
        if igst is  None:
            df = predictIGSTAmt_New(df)
        # df = predictDiscAmt(df)
        df = predictfreightAmt(df)
        if cess is  None:
            df = predictCessAmt(df)
            # df.to_csv("CessAmount.csv")
        df=predictAddlCessAmt(df)
        # df.to_csv("addl.csv")
        df = predictTCS(df)
        return df
    except:
        print("predHdrUsingFuzzy",
              traceback.print_exc())
        return df_copy


@putil.timing
def findLineValueQtyRate(df):

    from itertools import combinations as cmb
    try:
        num_cols = ["is_item_code1","is_qty1",
                    "is_unit_price1","is_item_val1",
                    "is_uom1","is_hsn_key1",
                    # "is_tax_rate_key1",
                    "is_cgst1","is_sgst1","is_igst1",
                    "is_disc1","extracted_amount"]
        df[num_cols] = df[num_cols].fillna(0.0)
        # df = df[num_cols].astype(float16)
        page_line_grp = df.groupby(["page_num","line_row"])
        # print("Page-line group",page_line_grp.groups)
        result = {}
        for grp_index,grp in enumerate(page_line_grp.groups):
            # print("page-line",grp)
            if grp[1] == 0:
                continue
            cond = df["page_num"] == grp[0]
            cond = cond & (df["line_row"] == grp[1])
            filt = df[cond]
            row_result = []
            # print("page-line",grp,filt.shape[0])
            if filt.shape[0] >= 3:
                cond = filt['extracted_amount'] > 0.0
                cond = cond | (filt["is_number"] == True)
                TEMP = filt[cond]
                TEMP_rows = []
                amts = []
                ids = []
                for index, row in TEMP.iterrows():
                    # amts.append(row["extracted_amount"])
                    # labels.append(row["predict_label"])
                    # ids.append(row["token_id"])
                    # probs.append(row["prediction_probability"])
                    TEMP_row = {}
                    TEMP_row["predict_label"] = row["predict_label"]
                    TEMP_row["prediction_probability"] = row["prediction_probability"]
                    TEMP_row["token_id"] = row["token_id"]

                    TEMP_row["is_item_code1"] = row['is_item_code1']
                    TEMP_row["is_qty1"] = row['is_qty1']
                    TEMP_row["is_unit_price1"] = row['is_unit_price1']
                    TEMP_row["is_unit_price"] = row["is_unit_price"]
                    TEMP_row["is_item_val1"] = row['is_item_val1']
                    TEMP_row["is_uom1"] = row['is_uom1']
                    TEMP_row["is_uom"] = row['is_uom']
                    TEMP_row["is_hsn_key1"] = row['is_hsn_key1']
                    # TEMP_row["is_tax_rate_key1"] = row['is_tax_rate_key1']
                    TEMP_row["is_cgst1"] = row['is_cgst1']
                    TEMP_row["is_sgst1"] = row['is_sgst1']
                    TEMP_row["is_igst1"] = row["is_igst1"]
                    TEMP_row["is_disc1"] = row["is_disc1"]

                    TEMP_row["amount"] = row["extracted_amount"]

                    if row["extracted_amount"] == 0.0:
                        if row["is_number"] == True:
                            try:
                                TEMP_row["amount"] = float(row["text"])
                            except:
                                TEMP_row["amount"] = 0.0
                                pass
                    TEMP_rows.append(TEMP_row)
                    amts.append(TEMP_row["amount"])
                    ids.append(row["token_id"])

                comb_1 = list(cmb(TEMP_rows,2))
                for comb in comb_1:
                    prod = comb[0]["amount"] * comb[1]["amount"]
                    if prod == 0.0:
                        continue
                    oth_amts = []
                    for id_,amt in zip(ids,amts):
                        if (comb[0]["token_id"] != id_) and (comb[1]["token_id"] != id_):
                            oth_amts.append(amt)
                    oth_rows = []
                    for row in TEMP_rows:
                        if (row != comb[0]) and (row != comb[1]):
                            oth_rows.append(row)

                    for am_index1,row_1 in enumerate(oth_rows):
                        amt_1 = row_1["amount"]
                        prod_disc = prod - amt_1
                        for am_index2,row_2 in enumerate(oth_rows):
                            if row_1 == row_2:
                                continue
                            amt_2 = row_2["amount"]
                            if amt_2 == 0.0:
                                continue
                            #found ItemValue without discount
                            # print(prod,prod_disc,amt_2)
                            if math.isclose(prod, amt_2,
                                            rel_tol = 0.001):
                                res = {"pot_qty":comb[0],
                                       "pot_up":comb[1],
                                       "pot_val":row_2,
                                       "pot_disc":None}
                                if res in row_result:
                                    continue
                                row_result.append(res)
                            elif math.isclose(prod_disc, amt_2,
                                              rel_tol = 0.001):
                                #Apr 20, 2022 - we'll consider Qty * UP as ItemValue
                                #Discounts will be extracted separately
                                # res = {"pot_qty":comb[0],
                                #        "pot_up":comb[1],
                                #        "pot_val":row_2,
                                #        "pot_disc":row_1}
                                res = {"pot_qty":comb[0],
                                       "pot_up":comb[1],
                                       "pot_val":row_2,
                                       "pot_disc":None}
                                #Apr 20, 2022 - we'll consider Qty * UP as ItemValue
                                #Discounts will be extracted separately
                                if res in row_result:
                                    continue
                                row_result.append(res)

            # print("Page-Row result",grp[0],grp[1],len(row_result))
            if len(row_result) > 0:
                result[str(grp[0]) + "-" + str(grp[1])] = row_result
        return result
    except:
        print("findLineValueQtyRate",
              traceback.print_exc())
        return {}

@putil.timing
def det_val_qty_up(row_result,df):

    df_copy = df.copy(deep=True)
    try:
        pred_label = {}
        pred_prob = {}
        for key in row_result.keys():
            results = row_result[key]
            page_num = int(key.split("-")[0])
            line_row = int(key.split("-")[1])
            upds = []
            for result in results:
                qty_row = result["pot_qty"]
                up_row = result["pot_up"]
                val_row = result["pot_val"]
                disc_row = result["pot_disc"]
                val_upd = False
                qty_row_qty = qty_row["is_qty1"]
                qty_row_noqty = max([qty_row["is_unit_price1"],
                                     qty_row["is_item_code1"],
                                     qty_row["is_item_val1"],
                                     qty_row["is_uom1"],
                                     qty_row["is_hsn_key1"],
                                     qty_row["is_cgst1"],
                                     qty_row["is_sgst1"],
                                     qty_row["is_igst1"],
                                     qty_row["is_disc1"]])
                qty_row_up = qty_row["is_unit_price1"]
                qty_row_noup = max([qty_row["is_qty1"],
                                    qty_row["is_item_code1"],
                                    qty_row["is_item_val1"],
                                    qty_row["is_uom1"],
                                    qty_row["is_hsn_key1"],
                                    qty_row["is_cgst1"],
                                    qty_row["is_sgst1"],
                                    qty_row["is_igst1"],
                                    qty_row["is_disc1"]])
                up_row_qty = up_row["is_qty1"]
                up_row_noqty = max([up_row["is_unit_price1"],
                                    up_row["is_item_code1"],
                                    up_row["is_item_val1"],
                                    up_row["is_uom1"],
                                    up_row["is_hsn_key1"],
                                    up_row["is_cgst1"],
                                    up_row["is_sgst1"],
                                    up_row["is_igst1"],
                                    up_row["is_disc1"]])
                up_row_up = up_row["is_unit_price1"]
                up_row_noup = max([up_row["is_qty1"],
                                   up_row["is_item_code1"],
                                   up_row["is_item_val1"],
                                   up_row["is_uom1"],
                                   up_row["is_hsn_key1"],
                                   up_row["is_cgst1"],
                                   up_row["is_sgst1"],
                                   up_row["is_igst1"],
                                   up_row["is_disc1"]])
                if disc_row is not None:
                    disc_row_disc = disc_row["is_disc1"]
                    disc_row_nodisc = max([disc_row["is_unit_price1"],
                                           disc_row["is_item_code1"],
                                           disc_row["is_item_val1"],
                                           disc_row["is_uom1"],
                                           disc_row["is_hsn_key1"],
                                           disc_row["is_cgst1"],
                                           disc_row["is_sgst1"],
                                           disc_row["is_igst1"],
                                           disc_row["is_qty1"]])
                # print("Qty and UPs",
                #       up_row_qty,up_row_noqty,qty_row_up,qty_row_noup,
                #       qty_row_qty,qty_row_noqty,up_row_up,up_row_noup,
                #       page_num)
                # print(qty_row)
                # print(up_row)
                if ((up_row_qty > up_row_noqty) and (up_row_qty > 0.7)) and ((qty_row_up >= qty_row_noup) and (qty_row_up > 0.7)):
                    row = qty_row
                    qty_row = up_row
                    up_row = row
                    cond = False
                    if (qty_row_up == qty_row_noup):
                        if (qty_row["is_unit_price"] > qty_row["is_uom"]):
                            cond = True
                    else:
                        cond = True

                    if disc_row is None:
                        val_upd = True
                    else:
                        if disc_row_disc > disc_row_nodisc:
                            val_upd = True
                elif ((qty_row_qty > qty_row_noqty) and (qty_row_qty > 0.7)) and ((up_row_up >= up_row_noup) and (up_row_up > 0.7)):

                    cond = False
                    if (up_row_up == up_row_noup):
                        if (up_row["is_unit_price"] > up_row["is_uom"]):
                            cond = True
                    else:
                        cond = True

                    if cond:
                        if disc_row is None:
                            val_upd = True
                        else:
                            if disc_row_disc > disc_row_nodisc:
                                val_upd = True

                # print("Value upd", val_upd)

                if val_upd:
                    # print(up_row)
                    upds.append({"qty":qty_row,
                                "up":up_row,
                                "val":val_row,
                                "disc":disc_row})
            # print("Length of upds",len(upds),upds)
            if len(upds) >= 1:
                # print("updates",page_num,line_row,upds)
                upd = upds[0]
                qty_row = upd["qty"]
                up_row = upd["up"]
                val_row = upd["val"]
                disc_row = upd["disc"]

                pred_label[qty_row["token_id"]] = "LI_itemQuantity"
                pred_prob[qty_row["token_id"]] = 1.0

                qty_no_cond = ((df["page_num"] == page_num) &
                               (df["line_row"] == line_row) &
                               (df["predict_label"] == "LI_itemQuantity") &
                               (df["token_id"] != qty_row["token_id"]))
                qty_df = df[qty_no_cond]
                if qty_df.shape[0] > 0:
                    tkns = list(df[qty_no_cond].token_id)
                    pred_prob.update({tkn:0.4 for tkn in tkns})


                pred_label[up_row["token_id"]] = "LI_unitPrice"
                pred_prob[up_row["token_id"]] = 1.0

                up_no_cond = ((df["page_num"] == page_num) &
                               (df["line_row"] == line_row) &
                               (df["predict_label"] == "LI_unitPrice") &
                               (df["token_id"] != up_row["token_id"]))
                up_df = df[up_no_cond]
                if up_df.shape[0] > 0:
                    tkns = list(df[up_no_cond].token_id)
                    pred_prob.update({tkn:0.4 for tkn in tkns})


                pred_label[val_row["token_id"]] = "LI_itemValue"
                pred_prob[val_row["token_id"]] = 1.0

                val_no_cond = ((df["page_num"] == page_num) &
                               (df["line_row"] == line_row) &
                               (df["predict_label"] == "LI_itemValue") &
                               (df["token_id"] != val_row["token_id"]))
                val_df = df[val_no_cond]
                if val_df.shape[0] > 0:
                    # print(type(val_df),val_df.token_id,list(val_df.token_id))
                    tkns = list(val_df.token_id)
                    pred_prob.update({tkn:0.4 for tkn in tkns})

                if disc_row is not None:

                    pred_label[disc_row["token_id"]] = "LI_discountAmount"
                    pred_prob[disc_row["token_id"]] = 1.0

                    disc_no_cond = ((df["page_num"] == page_num) &
                                   (df["line_row"] == line_row) &
                                   (df["predict_label"] == "LI_discountAmount") &
                                   (df["token_id"] != disc_row["token_id"]))
                    tkns = list(df[disc_no_cond]["token_id"].values())
                    pred_prob.update({tkn:0.4 for tkn in tkns})

        # print(pred_label,pred_prob)
        df = assignVavluesToDf("predict_label", pred_label, df)
        df = assignVavluesToDf("prediction_probability", pred_prob, df)
        # import numpy as np
        # df["predict_label_new"] = df["token_id"].map(pred_label)
        # df["prediction_probability_new"] = df["token_id"].map(pred_prob)
        # df["predict_label"] = np.where(df["predict_label_new"].isnull(),
        #                                df["predict_label"],
        #                                df["predict_label_new"])
        # df["prediction_probability"] = np.where(df["prediction_probability_new"].isnull(),
        #                                df["prediction_probability"],
        #                                df["prediction_probability_new"])

        return df
    except:
        print("det_val_qty_up",
              traceback.print_exc())
        return df_copy

@putil.timing
def det_gst(DF):

    # import numpy as np
    df_copy = DF.copy(deep = True)

    try:

        num_cols = ["is_item_code1","is_qty1",
                    "is_unit_price1","is_item_val1",
                    "is_uom1","is_hsn_key1","is_tax_rate_key1",
                    "is_cgst1","is_sgst1","is_igst1",
                    "is_disc1","extracted_amount"]
        DF[num_cols] = DF[num_cols].fillna(0.0)
        page_line_grp = DF.groupby(["page_num","line_row"])
        pred_prob = {}
        pred_label = {}
        cols = ["token_id","page_num","line_num","word_num",
                "line_row",
                "extracted_amount","is_number","predict_label",
                "prediction_probability"]
        df = DF[cols]
        for grp_index,grp in enumerate(page_line_grp.groups):
            page_num = int(grp[0])
            line_row = int(grp[1])
            if line_row == 0:
                continue
            cond = df["page_num"] == page_num
            cond = cond & (df["line_row"] == line_row)
            cond = cond & ((df['extracted_amount'] > 0.0) | (df["is_number"] == True))
            TEMP = df[cond]
            itm_val_df = TEMP[(TEMP["predict_label"] == "LI_itemValue") &
                            (TEMP["prediction_probability"] == 1.0)]
            # print("Item Value df",itm_val_df.shape,itm_val_df)
            if itm_val_df.shape[0] == 1:
                itm_val = list(itm_val_df["extracted_amount"])[0]
                cgst_df = TEMP[(TEMP["predict_label"] == "LI_CGSTAmount") &
                                (TEMP["prediction_probability"] == 1.0)]
                sgst_df = TEMP[(TEMP["predict_label"] == "LI_SGSTAmount") &
                                (TEMP["prediction_probability"] == 1.0)]
                igst_df = TEMP[(TEMP["predict_label"] == "LI_IGSTAmount") &
                                (TEMP["prediction_probability"] == 1.0)]
                if cgst_df.shape[0] > 0:
                    cgst_amount = 0.0
                    sel_row = None
                    for row in cgst_df.itertuples():
                        amount = row.extracted_amount
                        if isinstance(amount, float) and isinstance(itm_val, float):
                            max_tax = itm_val * .2
                            min_tax = itm_val * .025
                            if (min_tax <= amount <= max_tax) and (amount > 0.0):
                                if (cgst_amount < amount):
                                    sel_row = row
                                    cgst_amount = amount
                    if sel_row is not None:
                        pred_prob[sel_row.token_id] = 1.0
                        pred_label[sel_row.token_id] = "LI_CGSTAmount"
                        print("cgst row",sel_row.page_num,
                              sel_row.line_row,sel_row.token_id)
                        cgst_no_cond = df["page_num"] == sel_row.page_num
                        cgst_no_cond = cgst_no_cond & (df["line_row"] == sel_row.line_row)
                        cgst_no_cond = cgst_no_cond & (df["token_id"] != sel_row.token_id)
                        cgst_no_cond = cgst_no_cond & (df["predict_label"] == "LI_CGSTAmount")
                        cgsts = df[cgst_no_cond]
                        # print("cgsts",cgsts.shape)
                        tkns = list(cgsts.token_id)
                        pred_prob.update({tkn:0.4 for tkn in tkns})
                        # print("cgsts",cgsts.shape,tkns,pred_prob)

                if sgst_df.shape[0] > 0:
                    sgst_amount = 0.0
                    sel_row = None
                    for row in sgst_df.itertuples():
                        amount = row.extracted_amount
                        if isinstance(amount, float) and isinstance(itm_val, float):
                            max_tax = itm_val * .2
                            min_tax = itm_val * .025
                            if (min_tax <= amount <= max_tax) and (amount > 0.0):
                                if (sgst_amount < amount):
                                    sel_row = row
                                    sgst_amount = amount
                    if sel_row is not None:
                        pred_prob[sel_row.token_id] = 1.0
                        pred_label[sel_row.token_id] = "LI_SGSTAmount"
                        sgst_no_cond = df["page_num"] == sel_row.page_num
                        sgst_no_cond = sgst_no_cond & (df["line_row"] == sel_row.line_row)
                        sgst_no_cond = sgst_no_cond & (df["token_id"] != sel_row.token_id)
                        sgst_no_cond = sgst_no_cond & (df["predict_label"] == "LI_SGSTAmount")
                        sgsts = df[sgst_no_cond]
                        tkns = list(sgsts.token_id)
                        pred_prob.update({tkn:0.4 for tkn in tkns})

                if igst_df.shape[0] > 0:
                    igst_amount = 0.0
                    sel_row = None
                    for row in igst_df.itertuples():
                        amount = row.extracted_amount
                        if isinstance(amount, float) and isinstance(itm_val, float):
                            max_tax = itm_val * .3
                            min_tax = itm_val * .05
                            if (min_tax <= amount <= max_tax) and (amount > 0.0):
                                if (igst_amount < amount):
                                    sel_row = row
                                    igst_amount = amount
                    if sel_row is not None:
                        pred_prob[sel_row.token_id] = 1.0
                        pred_label[sel_row.token_id] = "LI_IGSTAmount"
                        igst_no_cond = df["page_num"] == sel_row.page_num
                        igst_no_cond = igst_no_cond & (df["line_row"] == sel_row.line_row)
                        igst_no_cond = igst_no_cond & (df["token_id"] != sel_row.token_id)
                        igst_no_cond = igst_no_cond & (df["predict_label"] == "LI_IGSTAmount")
                        igsts = df[igst_no_cond]
                        tkns = list(igsts.token_id)
                        pred_prob.update({tkn:0.4 for tkn in tkns})


        # print(pred_label,pred_prob)
        DF = assignVavluesToDf("predict_label", pred_label, DF)
        DF = assignVavluesToDf("prediction_probability", pred_prob, DF)

        # DF["predict_label_new"] = DF["token_id"].map(pred_label)
        # DF["prediction_probability_new"] = DF["token_id"].map(pred_prob)
        # DF["predict_label"] = np.where(DF["predict_label_new"].isnull(),
        #                                DF["predict_label"],
        #                                DF["predict_label_new"])
        # DF["prediction_probability"] = np.where(DF["prediction_probability_new"].isnull(),
        #                                DF["prediction_probability"],
        #                                DF["prediction_probability_new"])

        return DF
    except:
        print("det_gst",traceback.print_exc())
        return df_copy

## Validating and updating Vendor/billing/shipping details
def Update_vendor_billing_shipping_detils_labels(df):
    """
    updating pred_labels only with the unknow labels
    """
    try:
        for idx, row in df.iterrows():
            page_num = row["page_num"]
            token_id = row["token_id"]
            if (row["is_vendorGSTIN"]== 1) & (row["predict_label"] == "Unknown"):
                #print("inside v gsin 2")
                df.loc[(df["token_id"] == token_id),"predict_label"] = "vendorGSTIN"
            if (row["is_billingGSTIN"]== 1) & (row["predict_label"] == "Unknown"):
                #print("inside b gstin",row["is_billingGSTIN"])
                df.loc[(df["token_id"] == token_id),"predict_label"] = "billingGSTIN"
                print("token_ID :",token_id)
                print("lbl upd :",df.loc[[token_id],["predict_label"]])
            if (row["is_shippingGSTIN"]==1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"]==token_id),"predict_label"] = "shippingGSTIN"

            if (row["vendorName"] == 1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"] == token_id),"predict_label"] = "vendorName"
            if (row["vendorAddress"] == 1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"] == token_id),"predict_label"] = "vendorAddress"
            if (row["billingName"] == 1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"]==token_id),"predict_label"] = "billingName"
            if (row["billingAddress"]==1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"]==token_id),"predict_label"] = "billingAddress"
            if (row["shippingName"] == 1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"]==token_id),"predict_label"] = "shippingName"
            if (row["shippingAddress"] == 1) & (row["predict_label"] == "Unknown"):
                df.loc[(df["token_id"]==token_id),"predict_label"] = "shippingAddress"
        return df
    except:
        return df

## Validating and updating Vendor/billing/shipping details
def update_pred_labels_with_prob(df, fieldName,FeatureFieldName):
        
    df_copy = df.copy(deep = True)
    try:
        pred_label = {}
        pred_prob = {} 
        prob_of_field = {}

        token_ids = list(df[df[FeatureFieldName] == 1]["token_id"])
        if len(token_ids) > 0:
            pred_label = {token_id: fieldName for token_id in token_ids}
            print("pred_label upd :",pred_label)
            print("fieldName upd :",fieldName)

            pred_prob = {token_id:1.0 for token_id in token_ids}
            print("pred_prob upd :",pred_prob)
            prob_of_field = {token_id:1.0 for token_id in token_ids}
            print("prob_of_field upd :",prob_of_field)
            df = assignVavluesToDf("predict_label", pred_label, df)
            df = assignVavluesToDf("prediction_probability", pred_prob, df)
            df = assignVavluesToDf("prob_"+ fieldName, prob_of_field, df)
            print("updated Field label:",fieldName) #:",df[df["predict_label"]==fieldName]["predict_label"])
            return df
        else: 
            print("Zero tokens with this feature col",FeatureFieldName)
            return df
    except:
        print("update_pred_labels_with_prob exception :",traceback.print_exc())
        return df_copy

# Updating Model prediction for vendor / billing / shipping details
def update_pred_label_with_prob_in_df(df):
    """
    Adding manual probability for billingName and shippingName as model not trained to predict this.
    """
    df["prob_billingName"] = 0
    df["prob_shippingName"] = 0
    df['prob_shippingGSTIN'] = 0
    df['prob_billingGSTIN'] = 0
    
    #vendor Name
    fieldName = "vendorName"
    FeatureFieldName = "vendorName"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    #vendor GSTIN
    fieldName = "vendorGSTIN"
    FeatureFieldName = "is_vendorGSTIN"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # #vendor Address
    # fieldName = "vendorAddress"
    # FeatureFieldName = "vendorAddress"
    # df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    #billing Name
    fieldName = "billingName"
    FeatureFieldName = "billingName"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # billing GSTIN
    fieldName = "billingGSTIN"
    FeatureFieldName = "is_bilingGSTIN"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # # billing Address
    # fieldName = "billingAddress"
    # FeatureFieldName = "billingAddress"
    # df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # shipping Name
    fieldName = "shippingName"
    FeatureFieldName = "shippingName"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # shipping GSTIN
    fieldName = "shippingGSTIN"
    FeatureFieldName = "is_shippingGSTIN"
    df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    # # shipping Address
    # fieldName = "shippingAddress"
    # FeatureFieldName = "shippingAddress"
    # df = update_pred_labels_with_prob(df, fieldName,FeatureFieldName)

    return df


## clean / extract gstin fromat from text 
def get_gstin_format(DF):
    try:
        df = DF[DF["is_gstin_format"]==1]
        print("gst filter df :",df.shape[0] )
        gstin_list = {}
        if df.shape[0]>0:
            for row in df.itertuples():
                print("row text",row.text)   
                gstin = putil.correct_gstin(row.text)
                if gstin:
                    gstin_list[row.token_id] = gstin
            print("GSTIN extracted list :",gstin_list)
            DF = assignVavluesToDf("text", gstin_list, DF)
            return DF
        else:
            print("GSTINs not found in invoice")
            return DF
    except: 
        print("extrat gstin exception :",traceback.print_exc())
        return DF

def assignVavluesToDf_delet_new_col(col_name,col_vals,df,
                      base_col = "token_id"):
    import numpy as np
    new_col = col_name + "_new"
    df[new_col] = df[base_col].map(col_vals)
    df[col_name] = np.where(df[new_col].isnull(),
                            df[col_name],
                            df[new_col])
    del df[new_col]    
    return df

def extract_list_from_string(string :str):
    import re
    try:
        items_lsit =eval(re.search(r"\[(.*?)\]", string).group(0))
        return items_lsit
    except:
        print("pattern not found")
        return []


@putil.timing
def get_taxes_amounts_from_table_anchor_column(df):
    taxable, cgst, sgst, igst, cess, total, disc = None, None, None, None, None, None, None

    try:
        df["is_digit"] = df["text"].str.replace(".","").str.replace(",","").str.isdigit()
        fitler_df = df[df["is_digit"]==True]
        pred_labels = {}
        pred_prob = {}
        field_prob = {}
        temp = df[df["fz_lblDiscountAmount"]==1]
        print("disc amt label  shape:",temp.shape)
        if temp.shape[0] > 0:
            dis_row = temp.iloc[0].to_dict()
            # print("disc label :",dis_row)
            dis_line_left = dis_row.get('line_left')
            dis_line_right = dis_row.get('line_right')
        else:
            dis_line_left=0

        for row in fitler_df.itertuples():
            # below 3 lines for exact match
            # lst_left_processed_ngbr = extract_list_from_string(string = row.left_processed_ngbr.lower())
            # for item in lst_left_processed_ngbr:
            #     if (item == "total") or (item == "totals") :
            if "total" in row.left_processed_ngbr.lower():
                if ("gross" in row.above_processed_ngbr.lower()) or ("taxable" in row.above_processed_ngbr.lower()):
                    # print("SUBTOTAL_ROW",row.token_id)
                    # print(row.token_id,"2")
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    if taxable == None:
                        taxable = row.top
                        taxable_token = row.token_id
                        print("token_id :",row.token_id,"\tsubTotal :",row.text)
                        pred_labels[row.token_id]= "subTotal"
                        print("2107")
                        pred_prob[row.token_id]= 1
                        field_prob["prob_subTotal"] = {row.token_id:1}
                    else:
                        print("subtotal 1 token already assigned")
                        if taxable < row.top:
                            taxable = row.top
                            print("token_id :",row.token_id,"\tsubTotal :",row.text)
                            pred_labels[row.token_id]= "subTotal"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_subTotal"] = {row.token_id:1}
                            print("Removing old prediction token",pred_labels)
                            if taxable_token in pred_labels.keys():
                                del pred_labels[taxable_token]
                                del pred_prob[taxable_token]
                            taxable_token = row.token_id
                

                # if ("total" in row.above_processed_ngbr.lower()):   
                #     print("\nAbove processed :",row.above_processed_ngbr.lower())
                #     if total == None:
                #         total = row.text
                #         print("token_id :",row.token_id,"\tTotalAmount :",row.text)
                #         pred_labels[row.token_id]= "totalAmount"
                #         pred_prob[row.token_id]= 1
                #         field_prob["prob_totalAmount"] = {row.token_id:1}

                if ("disc" in row.above_processed_ngbr.lower() or "discount" in row.above_processed_ngbr.lower()) and (dis_line_left>0):
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    print("dis_line_left :",dis_line_left,"\tdis_line_right :",dis_line_right)
                    print("dis_left :",row.left,"\tdis_right :",row.right)
                    if (dis_line_left <= row.right) and (row.left <= dis_line_right):
                        print("inside overlap")
                        if disc == None:
                            disc = row.text
                            # disc_token = row.token_id
                            print("token_id :",row.token_id,"\disc :",row.text)
                            pred_labels[row.token_id]= "discountAmount"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_discountAmount"] = {row.token_id:1}

                if (("cgst" in row.above_processed_ngbr.lower() or ("central" in row.above_processed_ngbr.lower() and "tax" in row.above_processed_ngbr.lower())) and row.Percentage!=1):
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    if cgst == None:
                        cgst = row.top
                        cgst_token = row.token_id
                        print("token_id :",row.token_id,"\tCGST :",row.text)
                        pred_labels[row.token_id]= "CGSTAmount"
                        pred_prob[row.token_id]= 1
                        field_prob["prob_CGSTAmount"] = {row.token_id:1}
                    
                    
                    else:
                        if cgst < row.top:
                            cgst = row.top
                            print("token_id :",row.token_id,"\tCGST :",row.text)
                            pred_labels[row.token_id]= "CGSTAmount"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_CGSTAmount"] = {row.token_id:1}
                            print("Removing old prediction token")
                            if cgst_token in pred_labels.keys():
                                del pred_labels[cgst_token]
                                del pred_prob[cgst_token]
                            cgst_token = row.token_id

                if (("sgst" in row.above_processed_ngbr.lower() or ("state" in row.above_processed_ngbr.lower() and "tax" in row.above_processed_ngbr.lower())) and row.Percentage!=1):
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    if sgst == None:
                        sgst = row.top
                        sgst_token = row.token_id
                        print("token_id :",row.token_id,"\tSGST :",row.text)
                        pred_labels[row.token_id]= "SGSTAmount"
                        pred_prob[row.token_id]= 1
                        field_prob["prob_SGSTAmount"] = {row.token_id:1}
                    else:
                        print("sgst 1 token already assigned")
                        if sgst < row.top:
                            sgst = row.top
                            print("token_id :",row.token_id,"\tSGST :",row.text)
                            pred_labels[row.token_id]= "SGSTAmount"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_SGSTAmount"] = {row.token_id:1}
                            print("Removing old prediction token")
                            if sgst_token in pred_labels.keys():
                                del pred_labels[sgst_token]
                                del pred_prob[sgst_token]
                            sgst_token = row.token_id

                if (("igst" in row.above_processed_ngbr.lower() or ("integrated" in row.above_processed_ngbr.lower() and "tax" in row.above_processed_ngbr.lower())) and row.Percentage!=1):
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    if igst == None:
                        igst = row.top
                        igst_token = row.token_id
                        print("token_id :",row.token_id,"\tIGST:",row.text)
                        pred_labels[row.token_id]= "IGSTAmount"
                        pred_prob[row.token_id]= 1
                        field_prob["prob_IGSTAmount"] = {row.token_id:1}
                    else:
                        print("igst 1 token already assigned")
                        if igst < row.top:
                            igst = row.top
                            print("token_id :",row.token_id,"\tIGST :",row.text)
                            pred_labels[row.token_id]= "IGSTAmount"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_IGSTAmount"] = {row.token_id:1}
                            print("Removing old prediction token")
                            if igst_token in pred_labels.keys():
                                del pred_labels[igst_token]
                                del pred_prob[igst_token]
                            igst_token = row.token_id


                if ("cess" in row.above_processed_ngbr.lower()):
                    print("\nAbove processed :",row.above_processed_ngbr.lower())
                    if cess == None:
                        cess = row.top
                        # cess=float(cess)
                        cess_token = row.token_id
                        #print("token_id :",row.token_id,"\tCESS :",row.text)
                        pred_labels[row.token_id]= "CessAmount"
                        pred_prob[row.token_id]= 1
                        field_prob["prob_CessAmount"] = {row.token_id:1}
                    else:
                        cess = row.top
                        print("cess 1 token already assigned",cess)
                        # cess=float(cess)
                        if cess < row.top:
                            cess = row.top
                            print("token_id :",row.token_id,"\tCESS :",row.text)
                            pred_labels[row.token_id]= "CessAmount"
                            pred_prob[row.token_id]= 1
                            field_prob["prob_CessAmount"] = {row.token_id:1}
                            #print("Removing old prediction token")
                            if cess_token in pred_labels.keys():
                                del pred_labels[cess_token]
                                del pred_prob[cess_token]
                            cess_token = row.token_id
                        
        # print("pred_labels",pred_labels)
        # sgst = None
        # cgst = None
        # for key,val in pred_labels.items():
        #     if val == "CGSTAmount":
        #         cgst = val
        #     if val == "SGSTAmount":
        #         sgst = val
        # if ((cgst is None) and (sgst is not None)) or ((sgst is None) and (cgst is not None)):
        #     print("label and prob not updated")
        #     return df, taxable, cgst, sgst, igst, cess, total, disc
        df = assignVavluesToDf_delet_new_col("predict_label", pred_labels, df)
        df = assignVavluesToDf_delet_new_col("prediction_probability", pred_prob, df)
        for key, val in field_prob.items():
            df = assignVavluesToDf_delet_new_col(key, val, df) 
        print("values updated for :","\ttaxable :",taxable, "\tcgst: ",cgst, "\tsgst",sgst, "\tigst",igst, "\tcess",cess, "\ttotal",total, "\tdisc",disc)
        #df.drop("Percentage", axis=1, inplace=True)
        return df, taxable, cgst, sgst, igst, cess, total, disc
    except :
        print(traceback.print_exc())
        return df, taxable, cgst, sgst, igst, cess, total, disc
import copy
def identify_gstin_with_ocr_issues(df):
    copy_df = copy.deepcopy(df)
    try:
        print("before correcting gstin shape:",df[df["is_gstin_format"]==1].shape)
        ocr_issue_identifier_pattern_for_gstin = r'[\d|Z|I|O]{2}[A-Z|1|2|7]{5}[\d|Z|I|O]{4}[A-Z|2|1|7]{1}[A-Z\d]{1}[Z|2|7]{1}[A-Z\d]{1}'
        df.loc[df['text'].str.contains(ocr_issue_identifier_pattern_for_gstin,
                                       regex= True,na=False), 'is_gstin_format'] = 1
        print("after correcting gstin shape:",df[df["is_gstin_format"]==1].shape)
        return df
    except:
        print("identify_gstin_with_ocr_issues",traceback.print_exc())
        return df
                
@putil.timing
def modify_prediction(DF,vendor_masterdata):
    '''
    '''
    print("Modifying model prediction","\nDF shape :",DF.shape)
    
    DF = identify_gstin_with_ocr_issues(DF)
    DF = predHdrUsingFuzzy(DF)
    DF = update_pred_label_with_prob_in_df(DF)
    DF = refine_LI_df(DF)
    # DF = calculate_lines(DF)
    # DF = revamp_item_amounts(DF)
    results = findLineValueQtyRate(DF)
    if results != {}:
        # print("Results",results)
        DF = det_val_qty_up(results, DF)
    DF = det_gst(DF)
    # DF = get_taxes_amounts_from_table_anchor_column(DF) # called inside predfuzzy
    print("Modifying model prediction completed")
    # DF.to_csv("Modified_df1.csv")
    return DF

if __name__ == "__main__":
    import pandas as pd
    df = r"D:\TAPP_GIT\Swiggy_Instamart\Swiggy_pred/doc_1658384547892_8b9b9998a99_pred.csv"
    df=pd.read_csv(df)
    df = modify_prediction(df)
    #df.to_csv(r"CHAI.csv",index = False)


