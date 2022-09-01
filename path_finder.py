import pandas as pd
import os
import re
from difflib import SequenceMatcher
import operator
import numpy as np
import traceback
import json
import requests
from TAPPconfig import getUIServer

# from klein import Klein
# app = Klein()

script_dir = os.path.dirname(__file__)
customFieldPath = os.path.join(script_dir,
                              "Utilities/VENDOR_CUSTOM_FIELD.csv")


# Code needs to be commented: Start
def csv_delimiter_fixer(filename):
    """
    Need to call once
    Manual entry in CSV file changes the format
    """
    with open(filename, 'a+b') as fileobj:
        fileobj.seek(-1, 2)
        if fileobj.read(1) != b"\n":
            fileobj.write(b"\r\n")


csv_delimiter_fixer(customFieldPath)
# Code needs to be commented: End


VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape',
                                    keep_default_na=False)

LOCATION_COORDINATES = {"TOP LEFT": (0.0, 0.5, 0.0, 0.5),
"TOP RIGHT": (0.5, 1.0, 0.0, 0.5),
"BOTTOM LEFT": (0.0, 0.5, 0.5, 1.0),
"BOTTOM RIGHT": (0.5, 1.0, 0.5, 1.0),
"NONE": (0.0, 1.0, 0.0, 1.0),
"Not Applicable": (0.0, 1.0, 0.0, 1.0)}


def form_call_back_url():
    """

    :return:
    """
    #call_back_url = "http://52.172.231.99:8888"
    call_back_url = getUIServer()
    endpoint = "document/get"
    url = call_back_url + "/" + endpoint + "/"
    return url


def get_raw_prediction(document_id):
    """

    :param call_back_url:
    :param document_id:
    :param endpoint:
    :return:
    """
    url = form_call_back_url()
    url = url + document_id
    DF = None

    try:
        print("Getting Document Results:", url)
        result = requests.get(url).json().get('result')
        doc_result = result.get('document')
        raw_prediction = doc_result.get('rawPrediction')
    except Exception as e:
        print(e)
        raw_prediction = None
        pass

    if raw_prediction:
        DF = pd.read_json(path_or_buf=raw_prediction, orient="records")

    return DF


def insert_update_template_csv(list_template):
    """
    Add entry for new MasterData after all the validations have passed
    This code is for CSV insertion
    """
    try:
        global VENDOR_SPECIFIC_FIELD
        VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape',
                                            keep_default_na=False)
        print("Inside create template:")
        print(VENDOR_SPECIFIC_FIELD)

        # Remove the older template if already there
        for template in list_template:
            vendor_name = template["VENDOR_NAME"]
            vendor_id = template["VENDOR_ID"]
            field_name = template["Custom Field Name"]
            print(vendor_name, vendor_id, field_name)
            print("Shape before:", VENDOR_SPECIFIC_FIELD.shape)
            VENDOR_SPECIFIC_FIELD = VENDOR_SPECIFIC_FIELD.loc[~((VENDOR_SPECIFIC_FIELD["VENDOR_NAME"] == vendor_name)
                                                                & (VENDOR_SPECIFIC_FIELD["VENDOR_ID"] == vendor_id)
                                                                & (VENDOR_SPECIFIC_FIELD["Custom Field Name"] == field_name))]
            print("Shape after:", VENDOR_SPECIFIC_FIELD.shape)

        DF_INSERT = pd.DataFrame(list_template)
        print("Template to be added:")
        print(DF_INSERT)
        DF_NEW = pd.concat([VENDOR_SPECIFIC_FIELD, DF_INSERT], ignore_index=True)
        DF_NEW.sort_values(by=["VENDOR_ID", "Custom Field Name"], inplace=True)
        DF_NEW.reset_index(inplace=True, drop=True)
        print("Final Template File:")
        print(DF_NEW)
        DF_NEW.to_csv(customFieldPath, index=False)
        return True, "Successful Created!!"
    except Exception as e:
        print(e)
        traceback.print_exc()
        pass
        return False, "Failure"


def only_spclchar(s):
    if not re.match(r'^[_\W]+$', s):
        return True
    else:
        return False


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def find_similarity_words(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def remove_special_charcters(s):
    """
    """
    a = re.sub(r"[^a-zA-Z ]+", '', s)
    if a == '':
        return s
    else:
        return a


def extract_vendor_specific_extra_fields(DF, vendor_masterdata, TEMP_VENDOR_SPECIFIC_FIELD = None):
    """

    :param DF: pandas DataFrame
    :param vendor_masterdata: dictionary
    :param TEMP_VENDOR_SPECIFIC_FIELD: If not None, skip reading TEMPLATE FILE and use this instead
    Used when called via UI to Validate and Test created Template
    :return: dictionary with key as field name and extraction results as value
    """

    print("____________________")
    print('extract_vendor_specific_extra_fields')
    print(vendor_masterdata)

    vendor_id = vendor_masterdata['VENDOR_ID']
    # Code added to test Template creation
    global VENDOR_SPECIFIC_FIELD
    VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape',
                                        keep_default_na=False)

    if TEMP_VENDOR_SPECIFIC_FIELD is not None:
        VENDOR_SPECIFIC_FIELD = TEMP_VENDOR_SPECIFIC_FIELD

    print(vendor_id)
    print(VENDOR_SPECIFIC_FIELD)
    TEMP = VENDOR_SPECIFIC_FIELD.loc[VENDOR_SPECIFIC_FIELD['VENDOR_ID'] == vendor_id]
    #TEMP.to_csv("../CCCCC.csv")
    #TEMP.dropna(inplace=True)

    print(TEMP)
    dict_fields_to_extract = {}
    if not TEMP.empty:
        for idx, row in TEMP.iterrows():
            field = row['Custom Field Name']
            label = row['Custom Field Label']
            pos = row['Custom Field Label Position']
            shape = row['Custom Field Shape']
            loc = row['Custom Field Location']
            default_val = row['Default Value']
            horizontal_anchor = row['Horizontal Anchor']
            vertical_anchor = row['Vertical Anchor']
            # Code added for location based extraction
            top = row['TOP']
            bottom = row['BOTTOM']
            left = row['LEFT']
            right = row['RIGHT']
            top_delimiter = row['Top Delimiter']
            bottom_delimiter = row['Bottom Delimiter']
            include_top = row['INCLUDE_TOP']
            include_bottom = row['INCLUDE_BOTTOM']
            page_identifier = row['PAGE_IDENTIFIER']
            dict_fields_to_extract[field] = (label, pos, shape, loc, default_val, horizontal_anchor,
                                             vertical_anchor, top, bottom, left, right, top_delimiter, bottom_delimiter
                                             ,include_top ,include_bottom,
                                             page_identifier)


    DF['Left_1'] = DF['W1Lf'].astype(str).str.upper().replace('NAN' ,'')
    DF['Left_2'] = DF['W2Lf'].astype(str).str.upper().replace('NAN' ,'')
    DF['Left_3'] = DF['W3Lf'].astype(str).str.upper().replace('NAN' ,'')
    DF['Left_4'] = DF['W4Lf'].astype(str).str.upper().replace('NAN' ,'')
    DF['Left_5'] = DF['W5Lf'].astype(str).str.upper().replace('NAN' ,'')

    DF['Abv_1'] = DF['W1Ab'].astype(str).str.upper().replace('NAN' ,'')
    DF['Abv_2'] = DF['W2Ab'].astype(str).str.upper().replace('NAN' ,'')
    DF['Abv_3'] = DF['W3Ab'].astype(str).str.upper().replace('NAN' ,'')
    DF['Abv_4'] = DF['W4Ab'].astype(str).str.upper().replace('NAN' ,'')
    DF['Abv_5'] = DF['W5Ab'].astype(str).str.upper().replace('NAN' ,'')

    DF['LINE_TEXT'] = DF['line_text'].astype(str).apply(remove_special_charcters).str.upper()
    DF['LINE_TEXT'] = DF['LINE_TEXT'].str.replace(" " ,"")

    # DF['TEXT_ABOVE'] = DF["Abv_5"] + DF["Abv_4"] + DF["Abv_3"] + DF["Abv_2"] + DF["Abv_1"]
    DF['TEXT_ABOVE'] = DF["Abv_1"] + DF["Abv_2"] + DF["Abv_3"] + DF["Abv_4"] + DF["Abv_5"]
    DF['TEXT_LEFT'] = DF["Left_4"] + DF["Left_3"] + DF["Left_2"] + DF["Left_1"]

    DF['TEXT_ABOVE'] = DF['TEXT_ABOVE'].astype(str).apply(remove_special_charcters).str.upper()
    DF['TEXT_LEFT'] = DF['TEXT_LEFT'].astype(str).apply(remove_special_charcters).str.upper()

    results = {}
    for f, v in dict_fields_to_extract.items():
        print("*********************************************************")
        print("extracting" ,f)
        label = v[0]
        pos = v[1]
        shape = v[2]
        loc = v[3]
        default_val = v[4]
        horizontal_anchor = v[5]
        vertical_anchor = v[6]
        top = v[7]
        bottom = v[8]
        left = v[9]
        right = v[10]
        top_delimiter = v[11]
        bottom_delimiter = v[12]
        include_top = v[13]
        include_bottom = v[14]
        page_identifier = v[15]

        label = label.upper()
        list_labels = label.split()
        list_labels = [remove_special_charcters(x) for x in list_labels]
        string_label = ''.join(list_labels)

        horizontal_anchor = horizontal_anchor.upper()
        list_horizontal_anchor = horizontal_anchor.split()
        list_horizontal_anchor = [remove_special_charcters(x) for x in list_horizontal_anchor]
        horizontal_anchor = ''.join(list_horizontal_anchor)

        vertical_anchor = vertical_anchor.upper()
        list_vertical_anchor = vertical_anchor.split()
        list_vertical_anchor = [remove_special_charcters(x) for x in list_vertical_anchor]
        vertical_anchor = ''.join(list_vertical_anchor)

        # Add filtering based on location
        coordinates = LOCATION_COORDINATES[loc]
        TEMP = DF.loc[(DF['left'] >= coordinates[0]) & (DF['right'] <= coordinates[1]) &
                      (DF['top'] >= coordinates[2]) & (DF['bottom'] <= coordinates[3])]

        # Add filtering based on delimiters
        # Search for delimiters
        top_delimiter = top_delimiter.upper()
        list_top_delimiter = top_delimiter.split()
        list_top_delimiter = [remove_special_charcters(x) for x in list_top_delimiter]
        top_delimiter = ' '.join(list_top_delimiter)

        bottom_delimiter = bottom_delimiter.upper()
        list_bottom_delimiter = bottom_delimiter.split()
        list_bottom_delimiter = [remove_special_charcters(x) for x in list_bottom_delimiter]
        bottom_delimiter = ' '.join(list_bottom_delimiter)

        if (top_delimiter != "NOT APPLICABLE") or (bottom_delimiter != "NOT APPLICABLE"):
            print("Inside Delimiter Case")
            print(top_delimiter, bottom_delimiter)
            top_delimit = 0.0
            bottom_delimit = 1.0
            page_num_min = 0
            page_num_max = TEMP['page_num'].max()

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)

            TEMP_ = TEMP.copy()
            if top_delimiter != "NOT APPLICABLE":
                TEMP_["top_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                                                                        b=top_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['top_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['top_delimiter_score'], ascending=[False],
                                      inplace=True)
                    top_delimit = TEMP_.iloc[0]['line_down']
                    page_num_min = TEMP_.iloc[0]['page_num']

            TEMP_ = TEMP.copy()
            if bottom_delimiter != "NOT APPLICABLE":
                TEMP_["bottom_delimiter_score"] = TEMP_['LINE_TEXT'].apply(find_similarity_words,
                                                                           b=bottom_delimiter)
                TEMP_ = TEMP_.loc[TEMP_['bottom_delimiter_score'] > 0.8]
                if TEMP_.shape[0] > 0:
                    TEMP_.sort_values(['bottom_delimiter_score'], ascending=[False],
                                      inplace=True)
                    bottom_delimit = TEMP_.iloc[0]['line_top']
                    page_num_max = TEMP_.iloc[0]['page_num']

            print(top_delimit, bottom_delimit, page_num_min, page_num_max)
            TEMP = TEMP.loc[(TEMP['page_num'] >= page_num_min)
                            & (TEMP['page_num'] <= page_num_max)
                            & (TEMP['line_top'] >= top_delimit)
                            & (TEMP['line_down'] <= bottom_delimit)]



        ### Find Score based on Left Value
        if pos == 'Left':
            TEMP["surrounding_text_score"] = TEMP['TEXT_LEFT'].apply(find_similarity_words,
                                                                     b=string_label)
            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif pos == 'Above':
            TEMP["surrounding_text_score"] = TEMP['TEXT_ABOVE'].apply(find_similarity_words,
                                                                      b=string_label)
            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
        elif (pos == 'ANCHOR_LEFT_TOP') | (pos == "ANCHOR_LEFT_BOTTOM") | (pos == 'ANCHOR_RIGHT_TOP') | \
                (pos == "ANCHOR_RIGHT_BOTTOM"):
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0

            TEMP_ANCHOR = DF.copy()
            print("Inside ANCHOR Label CASE!!!!!!!", horizontal_anchor, vertical_anchor)
            TEMP_ANCHOR["horizontal_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
                                                                                    b=horizontal_anchor)
            TEMP_ANCHOR["vertical_anchor_score"] = TEMP_ANCHOR['LINE_TEXT'].apply(find_similarity_words,
                                                                                  b=vertical_anchor)


            TEMP_HORIZONTAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['horizontal_anchor_score'] > 0.70)][
                ['LINE_TEXT', 'horizontal_anchor_score', 'line_top', 'line_down', 'line_left', 'line_right']]
            TEMP_HORIZONTAL_ANCHOR.drop_duplicates(inplace=True)
            TEMP_VERTICAL_ANCHOR = TEMP_ANCHOR.loc[(TEMP_ANCHOR['vertical_anchor_score'] > 0.70)][
                ['LINE_TEXT', 'vertical_anchor_score', 'line_left', 'line_right', 'line_top', 'line_down']]
            TEMP_VERTICAL_ANCHOR.drop_duplicates(inplace=True)
            print("CHecking Template")
            print(TEMP_VERTICAL_ANCHOR)
            print(TEMP_HORIZONTAL_ANCHOR)

            if (TEMP_HORIZONTAL_ANCHOR.shape[0] > 0) & (TEMP_VERTICAL_ANCHOR.shape[0] > 0):
                TEMP_HORIZONTAL_ANCHOR.sort_values(['horizontal_anchor_score'], ascending=[False], inplace=True)

                TEMP_VERTICAL_ANCHOR.sort_values(['vertical_anchor_score'], ascending=[False], inplace=True)
                print(TEMP_VERTICAL_ANCHOR)
                print(TEMP_HORIZONTAL_ANCHOR)

                print(TEMP.shape)
                TEMP_ = TEMP.copy()
                dummyDF = pd.DataFrame(columns=list(TEMP.columns))
                for idx_, row_ in TEMP_HORIZONTAL_ANCHOR.iterrows():
                    top_boundary = float(row_['line_top'])
                    bottom_boundary = float(row_['line_down'])
                    horizontal_anchor_score = float(row_['horizontal_anchor_score'])
                    for idx__, row__ in TEMP_VERTICAL_ANCHOR.iterrows():
                        left_boundary = float(row__['line_left'])
                        right_boundary = float(row__['line_right'])
                        vertical_anchor_score = float(row__['vertical_anchor_score'])
                        TEMP = TEMP_.copy()
                        if pos == "ANCHOR_LEFT_TOP":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                             & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_LEFT_BOTTOM":
                            horizontal_cut = float(row_['line_right'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                             & (TEMP['left'] >= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_TOP":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_down'])
                            TEMP = (TEMP.loc[(TEMP['top'] >= vertical_cut)
                                             & (TEMP['right'] <= horizontal_cut)])
                        elif pos == "ANCHOR_RIGHT_BOTTOM":
                            horizontal_cut = float(row_['line_left'])
                            vertical_cut = float(row__['line_top'])
                            TEMP = (TEMP.loc[(TEMP['bottom'] <= vertical_cut)
                                             & (TEMP['right'] <= horizontal_cut)])

                        TEMP = (TEMP.loc[(top_boundary <= TEMP['bottom'])
                                         & (TEMP['top'] <= bottom_boundary)
                                         & (left_boundary <= TEMP['right'])
                                         & (TEMP['left'] <= right_boundary)])

                        if TEMP.shape[0] > 0:
                            TEMP['anchor_left'] = left_boundary
                            TEMP['anchor_right'] = right_boundary
                            TEMP['anchor_top'] = top_boundary
                            TEMP['anchor_bottom'] = bottom_boundary

                            TEMP['min_left'] = TEMP[['left', 'anchor_left']].max(axis=1)
                            TEMP['min_right'] = TEMP[['right', 'anchor_right']].min(axis=1)
                            TEMP['min_top'] = TEMP[['top', 'anchor_top']].max(axis=1)
                            TEMP['min_bottom'] = TEMP[['bottom', 'anchor_bottom']].min(axis=1)

                            TEMP['overlap_horizontal'] = TEMP['min_right'] - TEMP['min_left']
                            TEMP['overlap_vertical'] = TEMP['min_bottom'] - TEMP['min_top']

                            TEMP['overlap_score_horizontal'] = (TEMP['overlap_horizontal']
                                                                / TEMP['width'])
                            TEMP['overlap_score_vertical'] = (TEMP['overlap_vertical']
                                                              / TEMP['height'])

                            TEMP['overlap_score_vertical'] = TEMP['overlap_score_vertical'] * horizontal_anchor_score
                            TEMP['overlap_score_horizontal'] = TEMP['overlap_score_horizontal'] * vertical_anchor_score

                            TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                            TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                            if TEMP.shape[0] > 0:
                                TEMP['surrounding_text_score'] = TEMP[['overlap_score_horizontal'
                                    , 'overlap_score_vertical']].mean(axis=1)
                                dummyDF = pd.concat([dummyDF, TEMP], axis=0)
                if dummyDF.shape[0] > 0:
                    TEMP = dummyDF.sort_values(['surrounding_text_score'], ascending=[False])
                    TEMP = TEMP.reset_index()
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
        elif pos == 'LOCATION':
            # Code added for Location based extraction
            print("Inside Location Based Extraction")
            try:
                top = float(top) * 0.95
                bottom = float(bottom) * 1.05
                left = float(left) * 0.95
                right = float(right) * 1.05

                TEMP = (TEMP.loc[(top <= TEMP['bottom'])
                                 & (TEMP['top'] <= bottom)
                                 & (left <= TEMP['right'])
                                 & (TEMP['left'] <= right)])
                if TEMP.shape[0] > 0:
                    TEMP['top_boundary'] = top
                    TEMP['bottom_boundary'] = bottom
                    TEMP['left_boundary'] = left
                    TEMP['right_boundary'] = right

                    TEMP['min_top'] = TEMP[['top', 'top_boundary']].max(axis=1)
                    TEMP['min_bottom'] = TEMP[['bottom', 'bottom_boundary']].min(axis=1)
                    TEMP['min_left'] = TEMP[['left', 'left_boundary']].max(axis=1)
                    TEMP['min_right'] = TEMP[['right', 'right_boundary']].min(axis=1)

                    TEMP['overlap_area'] = ((TEMP['min_right'] - TEMP['min_left']) *
                                            (TEMP['min_bottom'] - TEMP['min_top']))

                    TEMP['area'] = ((TEMP['right'] - TEMP['left']) *
                                    (TEMP['bottom'] - TEMP['top']))

                    TEMP["wordshape_score"] = TEMP['wordshape'].apply(find_similarity_words, b=shape)
                    TEMP = TEMP.loc[TEMP['wordshape_score'] > 0.60]
                    if TEMP.shape[0] > 0:
                        TEMP['surrounding_text_score'] = TEMP['overlap_area'] / TEMP['area']
                    else:
                        TEMP["surrounding_text_score"] = 0.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
            except:
                TEMP["surrounding_text_score"] = 0.0
                TEMP["wordshape_score"] = 0.0
        elif pos == 'MULTI_TOKEN':
            # Code added for Location based extraction
            print("Multi Token Extraction")
            try:
                proceed_for_extraction = True
                page_num = None
                top_b = 0
                bottom_b = 1
                left_b = 0
                right_b = 1

                page_identifier = str(page_identifier)
                page_identifier = page_identifier.strip()

                location_provided = False
                try:
                    page_num = int(page_identifier) - 1
                    top_b = float(str(top).strip())
                    bottom_b = float(str(bottom).strip())
                    left_b = float(str(left).strip())
                    right_b = float(str(right).strip())
                    location_provided = True
                except:
                    pass

                skip_top = False
                skip_bottom = False
                skip_left = False
                skip_right = False
                if isfloat(str(top).strip()):
                    top_b = float(str(top).strip())
                    skip_top = True

                if isfloat(str(bottom).strip()):
                    bottom_b = float(str(bottom).strip())
                    skip_bottom = True

                if isfloat(str(left).strip()):
                    left_b = float(str(left).strip())
                    skip_left = True

                if isfloat(str(right).strip()):
                    right_b = float(str(right).strip())
                    skip_right = True

                print(location_provided, "^^&&")
                page_identifier = page_identifier.upper()
                print("Page Identifier:", page_identifier)
                if (not location_provided) & (page_identifier != "NOT APPLICABLE"):
                    try:
                        page_num = int(page_identifier) - 1
                        TEMP = TEMP.loc[TEMP['page_num'] == page_num]
                    except:
                        page_identifier_list = re.split(r'[^\w]', str(page_identifier))
                        print("Page Identifier List:", page_identifier_list)
                        unique_pages = list(TEMP['page_num'].unique())
                        print("Unique Pages:", unique_pages)
                        page_identifier_match = {}
                        for p in unique_pages:
                            l = list(TEMP.loc[TEMP['page_num'] == p]['text'])
                            l = [str(x) for x in l]
                            s = " ".join(l)
                            s = s.upper()
                            matches = [x for x in page_identifier_list if x in s]
                            page_identifier_match[p] = len(matches) / len(page_identifier_list)
                        print("Page Identifier Match Scores:", page_identifier_match)
                        predicted_page = max(page_identifier_match.items(), key=operator.itemgetter(1))[0]
                        max_score = page_identifier_match[predicted_page]
                        if max_score < 0.6:
                            proceed_for_extraction = False
                        else:
                            TEMP = TEMP.loc[TEMP['page_num'] == predicted_page]
                            page_num = predicted_page

                print(top, bottom, left, right)
                top_marker = remove_special_charcters(str(top)).upper()
                bottom_marker = remove_special_charcters(str(bottom)).upper()
                left_marker = remove_special_charcters(str(left)).upper()
                right_marker = remove_special_charcters(str(right)).upper()

                top_marker = top_marker.replace(" ", "")
                bottom_marker = bottom_marker.replace(" ", "")
                left_marker = left_marker.replace(" ", "")
                right_marker = right_marker.replace(" ", "")
                print(top_marker, bottom_marker, left_marker, right_marker)

                if (not location_provided) & (not skip_top) & (top_marker != "PAGEBOUNDARY"):
                    TEMP["top_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                                                                       b=top_marker)
                    TEMP["top_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(top_marker), "top_marker_score_exact"] = 1
                    TEMP["top_marker_score"] = TEMP[["top_marker_score", "top_marker_score_exact"]].max(axis=1)

                    TEMP.sort_values(['top_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print("top_marker:", top_marker)
                    print(TEMP[["page_num", "LINE_TEXT", "top_marker_score", "line_down"]])
                    d_ = dict(TEMP.iloc[0])
                    if d_["top_marker_score"] > 0.6:
                        if include_top == "YES":
                            top_b = d_["line_top"] * 0.99
                        else:
                            top_b = d_["line_down"] * 0.99
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_bottom) & (bottom_marker != "PAGEBOUNDARY"):
                    TEMP["bottom_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                                                                          b=bottom_marker)
                    TEMP["bottom_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(bottom_marker), "bottom_marker_score_exact"] = 1
                    TEMP["bottom_marker_score"] = TEMP[["bottom_marker_score", "bottom_marker_score_exact"]].max(axis=1)
                    TEMP.sort_values(['bottom_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print("bottom_marker:", bottom_marker)
                    print(TEMP[["page_num", "LINE_TEXT", "bottom_marker_score", "line_top"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["bottom_marker_score"] > 0.6) & (
                            (page_num is None) | ((page_num is not None) & (page_num == d_["page_num"]))):
                        if include_bottom == "YES":
                            bottom_b = d_["line_down"] * 1.01
                        else:
                            bottom_b = d_["line_top"] * 1.01
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_left) & (left_marker != "PAGEBOUNDARY"):
                    TEMP["left_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                                                                        b=left_marker)
                    TEMP["left_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(left_marker), "left_marker_score_exact"] = 1
                    TEMP["left_marker_score"] = TEMP[["left_marker_score", "left_marker_score_exact"]].max(axis=1)
                    print("left_marker:", left_marker)
                    # print(TEMP[["page_num", "LINE_TEXT", "left_marker_score", 'left_marker_score_exact',"line_right"]].drop_duplicates().head(30),"left***")

                    TEMP.sort_values(['left_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print(TEMP[["page_num", "LINE_TEXT", "left_marker_score", "line_right"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["left_marker_score"] > 0.6) & (page_num == d_["page_num"]):
                        len_ = len(str(left).split())
                        TEMP_ = TEMP.loc[(DF["page_num"] == d_["page_num"]) &
                                         (DF["line_num"] == d_["line_num"])]
                        TEMP_.sort_values(["page_num", "line_num", "word_num"], inplace=True)

                        list_texts = list(TEMP_["text"])
                        list_boundary = list(TEMP_["right"])

                        list_text_search = [''.join(list_texts[i: i + len_]) for i in range(len(list_texts) - len_ + 1)]
                        list_text_search = [remove_special_charcters(x).upper() for x in list_text_search]
                        list_score = [find_similarity_words(s, left_marker) for s in list_text_search]
                        max_score_index = list_score.index(max(list_score))

                        list_boundary_search = [list_boundary[i: i + len_][-1] for i in
                                                range(len(list_boundary) - len_ + 1)]
                        left_b = list_boundary_search[max_score_index] * 0.99
                    else:
                        proceed_for_extraction = False
                if (not location_provided) & (not skip_right) & (right_marker != "PAGEBOUNDARY"):
                    TEMP["right_marker_score"] = TEMP['LINE_TEXT'].apply(find_similarity_words,
                                                                         b=right_marker)
                    TEMP["right_marker_score_exact"] = 0
                    TEMP.loc[TEMP['LINE_TEXT'].str.contains(right_marker), "right_marker_score_exact"] = 1
                    TEMP["right_marker_score"] = TEMP[["right_marker_score", "right_marker_score_exact"]].max(axis=1)
                    print("right_marker:", right_marker)
                    TEMP.sort_values(['right_marker_score', 'page_num'], ascending=[False, True], inplace=True)
                    print(TEMP[["page_num", "LINE_TEXT", "right_marker_score", "line_left"]])
                    d_ = dict(TEMP.iloc[0])
                    if (d_["right_marker_score"] > 0.6) & (
                            (page_num is None) | ((page_num is not None) & (page_num == d_["page_num"]))):
                        print("yes")
                        print(right, "rrr")
                        len_ = len(str(right).split())
                        print(len_, "LR")
                        TEMP_ = TEMP.loc[(DF["page_num"] == d_["page_num"]) &
                                         (DF["line_num"] == d_["line_num"])]
                        TEMP_.sort_values(["page_num", "line_num", "word_num"], inplace=True)

                        list_texts = list(TEMP_["text"])
                        list_boundary = list(TEMP_["left"])

                        list_text_search = [''.join(list_texts[i: i + len_]) for i in range(len(list_texts) - len_ + 1)]
                        list_text_search = [remove_special_charcters(x).upper() for x in list_text_search]
                        list_score = [find_similarity_words(s, right_marker) for s in list_text_search]

                        max_score_index = list_score.index(max(list_score))

                        list_boundary_search = [list_boundary[i: i + len_][0] for i in
                                                range(len(list_boundary) - len_ + 1)]

                        # right_b = d_["line_left"] * 0.99
                        right_b = list_boundary_search[max_score_index] * 1.01
                        page_num = d_["page_num"]
                    else:
                        proceed_for_extraction = False

                if proceed_for_extraction:
                    print(top_b, bottom_b, left_b, right_b)
                    print("pageno", page_num)
                    print("topb", top_b)
                    print("bob", bottom_b)
                    print("lb", left_b)
                    print("rb", right_b)


                    TEMP = (TEMP.loc[(TEMP["page_num"] == page_num)
                                     & (top_b < TEMP['top'])
                                     & (TEMP['bottom'] < bottom_b)
                                     & (left_b < TEMP['left'])
                                     & (TEMP['right'] < right_b)])
                    print("temp", TEMP[["line_text"]])

                    TEMP['isalphanum'] = TEMP['text'].apply(only_spclchar)
                    TEMP = TEMP.loc[TEMP['isalphanum']]
                    if TEMP.shape[0] > 0:
                        TEMP.sort_values(["page_num", "line_num", "word_num"],
                                         ascending=[True, True, True], inplace=True)

                        TEMP = TEMP.groupby(['page_num', 'line_num']).agg({'text': lambda x: "%s" % ' '.join(x),
                                                                           'word_num': 'count',
                                                                           'left': 'min',
                                                                           'right': 'max',
                                                                           'top': 'min',
                                                                           'bottom': 'max',
                                                                           'image_height': 'first',
                                                                           'image_widht': 'first'}).reset_index()

                        TEMP = TEMP.groupby(['page_num']).agg({'text': lambda x: "%s" % '\n'.join(x),
                                                               'line_num': 'count',
                                                               'word_num': 'count',
                                                               'left': 'min',
                                                               'right': 'max',
                                                               'top': 'min',
                                                               'bottom': 'max',
                                                               'image_height': 'first',
                                                               'image_widht': 'first'}).reset_index()

                        TEMP["surrounding_text_score"] = 1.0
                        TEMP["wordshape_score"] = 1.0
                        print(TEMP)
                    else:
                        TEMP["surrounding_text_score"] = 0.0
                        TEMP["wordshape_score"] = 0.0
                else:
                    TEMP["surrounding_text_score"] = 0.0
                    TEMP["wordshape_score"] = 0.0
            except Exception as e:
                print(e)
                TEMP["surrounding_text_score"] = 0.0
                TEMP["wordshape_score"] = 0.0
        else:
            TEMP["surrounding_text_score"] = 0.0
            TEMP["wordshape_score"] = 0.0
        TEMP = TEMP.loc[TEMP["wordshape_score"] >= 0.5]
        TEMP['field_score'] = (TEMP['surrounding_text_score'] * 0.7 + TEMP['wordshape_score'] * 0.3)
        print(TEMP.sort_values(['field_score'], ascending=[False], inplace=False)[['text',
                                                                                   'wordshape_score',
                                                                                   'surrounding_text_score',
                                                                                   'field_score']])

        TEMP = TEMP.loc[TEMP['field_score'] > 0.55]
        TEMP.sort_values(['field_score'], ascending=[False], inplace=True)
        print(TEMP[['text',
                    'wordshape_score', 'surrounding_text_score', 'field_score']])
        print(TEMP[['text',
                    'wordshape_score', 'surrounding_text_score', 'field_score']])

        final_candidates_ = {}
        if not TEMP.empty:
            conf_var = np.random.uniform(0.6, 0.8)
            extracted_value = TEMP.iloc[0]['text']
            final_candidates_['line_num'] = TEMP.iloc[0]['line_num']
            final_candidates_["prob_" + f] = conf_var
            final_candidates_['text'] = extracted_value
            final_candidates_["Label_Present"] = True
            final_candidates_['word_num'] = TEMP.iloc[0]['word_num']

            final_candidates_['left'] = TEMP.iloc[0]['left']
            final_candidates_['right'] = TEMP.iloc[0]['right']
            final_candidates_['conf'] = conf_var
            final_candidates_['top'] = TEMP.iloc[0]['top']
            final_candidates_['bottom'] = TEMP.iloc[0]['bottom']

            final_candidates_['page_num'] = TEMP.iloc[0]['page_num']
            final_candidates_['image_height'] = TEMP.iloc[0]['image_height']
            final_candidates_['image_widht'] = TEMP.iloc[0]['image_widht']

            final_candidates_["label_confidence"] = None
            final_candidates_["wordshape"] = None
            final_candidates_["wordshape_confidence"] = None
            final_candidates_["Odds"] = None
            final_candidates_['model_confidence'] = 1

            final_candidates_['final_confidence_score'] = conf_var
            final_candidates_['vendor_masterdata_present'] = True
            final_candidates_['extracted_from_masterdata'] = False
        else:
            # Default Value is getting populated
            print("Inside Not Extracted:", default_val)
            extracted_value = default_val
            final_candidates_['line_num'] = 0
            final_candidates_["prob_" + f] = 1
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
        results[f] = final_candidates_
    return results


def form_row_CSV(dict_template):
    """

    :param dict_template:
    :return:
    """
    global VENDOR_SPECIFIC_FIELD
    VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape',
                                        keep_default_na=False)
    list_cols = list(VENDOR_SPECIFIC_FIELD.columns)

    for col in list_cols:
        if not col in dict_template:
            dict_template[col] = "Not Applicable"

    dict_template = {key: dict_template[key] for key in list_cols if key in dict_template}
    return dict_template


def transform_template_UI_CSV(dict_template):
    """
    Transforms CSV template format into UI format
    :param dict_template:
    :return:
    """
    print("************************************")
    #print(dict_template)

    result_dict = {"VENDOR_NAME": dict_template["vendor_name"],
                   "VENDOR_ID": dict_template["vendor_id"],
                   "Custom Field Name": dict_template["field_name"],
                   "Default Value": dict_template["default_value"]}

    if dict_template["template_type"] == "static":
        print("Inside Static Template Case")
        result_dict["Custom Field Label"] = "STATIC"
        result_dict["Custom Field Label Position"] = "STATIC"
        result_dict["Default Value"] = dict_template["default_value"]
    elif dict_template["template_type"] == "anchor_label":
        print("Inside Anchor Label Case")
        result_dict["Custom Field Label"] = str("ANCHOR_" + dict_template["horizontal_anchor_location"]
                                                + "_" + dict_template["vertical_anchor_location"]).upper()
        result_dict["Custom Field Label Position"] = str("ANCHOR_" + dict_template["horizontal_anchor_location"]
                                                + "_" + dict_template["vertical_anchor_location"]).upper()
        result_dict["Horizontal Anchor"] = dict_template["horizontal_anchor"]
        result_dict["Vertical Anchor"] = dict_template["vertical_anchor"]
        result_dict["Custom Field Shape"] = dict_template["field_shape"]
        # Code added for delimiter
        result_dict["Top Delimiter"] = dict_template["top_delimiter"]
        result_dict["Bottom Delimiter"] = dict_template["bottom_delimiter"]
    elif dict_template["template_type"] == "bounding_box_identifier":
        print("Inside Bounding Box Identifier Case")
        print(dict_template)
        result_dict["Custom Field Label"] = "MULTI_TOKEN"
        result_dict["Custom Field Label Position"] = "MULTI_TOKEN"
        result_dict["TOP"] = dict_template["top"]
        result_dict["INCLUDE_TOP"] = dict_template["include_top"]
        result_dict["BOTTOM"] = dict_template["bottom"]
        result_dict["INCLUDE_BOTTOM"] = dict_template["include_bottom"]
        result_dict["LEFT"] = dict_template["left"]
        result_dict["RIGHT"] = dict_template["right"]
        result_dict["PAGE_IDENTIFIER"] = dict_template["page_identifier"]
        # Code added for delimiter
        result_dict["Top Delimiter"] = dict_template["top_delimiter"]
        result_dict["Bottom Delimiter"] = dict_template["bottom_delimiter"]
    else:
        print("Inside Single Label Case")
        result_dict["Custom Field Label"] = dict_template["label"]
        result_dict["Custom Field Label Position"] = dict_template["label_position"]
        result_dict["Custom Field Shape"] = dict_template["field_shape"]
        result_dict["Custom Field Location"] = dict_template["field_location"]
        # Code added for delimiter
        result_dict["Top Delimiter"] = dict_template["top_delimiter"]
        result_dict["Bottom Delimiter"] = dict_template["bottom_delimiter"]

    return result_dict


def transform_template_CSV_UI(dict_template):
    """
    Transforms CSV template format into UI format
    :param dict_template:
    :return:
    """
    print("************************************")
    #print(dict_template)

    result_dict = {"vendor_name": dict_template["VENDOR_NAME"],
                   "vendor_id": dict_template["VENDOR_ID"],
                   "field_name": dict_template["Custom Field Name"],
                   "default_value": dict_template["Default Value"]}

    if dict_template["Custom Field Label"] == "STATIC":
        print("Inside Static Template Case")
        result_dict["field_type"] = "MULTI_TOKEN"
        result_dict["template_type"] = "static"
        result_dict["field_value"] = dict_template["Default Value"]
    elif dict_template["Custom Field Label"] in ["ANCHOR_LEFT_TOP", "ANCHOR_LEFT_BOTTOM",
                                                 "ANCHOR_RIGHT_TOP", "ANCHOR_RIGHT_BOTTOM"]:
        print("Inside Anchor Label Case")
        result_dict["field_type"] = "SINGLE_TOKEN"
        result_dict["template_type"] = "anchor_label"
        result_dict["horizontal_anchor"] = dict_template["Horizontal Anchor"]
        result_dict["vertical_anchor"] = dict_template["Vertical Anchor"]
        result_dict["field_shape"] = dict_template["Custom Field Shape"]
        result_dict["horizontal_anchor_location"] = str(dict_template["Custom Field Label"]).split('_')[1]
        result_dict["vertical_anchor_location"] = str(dict_template["Custom Field Label"]).split('_')[2]
    elif dict_template["Custom Field Label"] == "MULTI_TOKEN":
        print("Inside Bounding Box Identifier Case")
        result_dict["field_type"] = "MULTI_TOKEN"
        result_dict["template_type"] = "bounding_box_identifier"
        result_dict["top"] = dict_template["TOP"]
        result_dict["include_top"] = dict_template["INCLUDE_TOP"]
        result_dict["bottom"] = dict_template["BOTTOM"]
        result_dict["include_bottom"] = dict_template["INCLUDE_BOTTOM"]
        result_dict["left"] = dict_template["LEFT"]
        result_dict["right"] = dict_template["RIGHT"]
        result_dict["page_identifier"] = dict_template["PAGE_IDENTIFIER"]
    else:
        print("Inside Single Label Case")
        result_dict["field_type"] = "SINGLE_TOKEN"
        result_dict["template_type"] = "single_label"
        result_dict["label"] = dict_template["Custom Field Label"]
        result_dict["label_position"] = dict_template["Custom Field Label Position"]
        result_dict["field_shape"] = dict_template["Custom Field Shape"]
        result_dict["field_location"] = dict_template["Custom Field Location"]

    return result_dict


def get_templates_CSV(vendor_id, list_field = None):
    """
    Get active template based on vendor_id and field_name
    field_name is optional
    :return:
    """
    global VENDOR_SPECIFIC_FIELD
    VENDOR_SPECIFIC_FIELD = pd.read_csv(customFieldPath, encoding='unicode_escape',
                                        keep_default_na=False)
    TEMPLATE = VENDOR_SPECIFIC_FIELD.loc[VENDOR_SPECIFIC_FIELD["VENDOR_ID"] == vendor_id]
    if list_field is not None:
        TEMPLATE = TEMPLATE.loc[TEMPLATE["Custom Field Name"].isin(list_field)]

    list_templates = []
    for idx, row in TEMPLATE.iterrows():
        list_templates.append(dict(row))

    return list_templates


def get_extraction_for_template(DF, template):
    """
    Input is CSV format template (Single Template) in dictionary form
    :param DF:
    :param template:
    :return:
    """
    try:
        extracted_value = extract_vendor_specific_extra_fields(DF, {"VENDOR_ID": template["VENDOR_ID"],
                                                                    "VENDOR_NAME": template["VENDOR_NAME"]},
                                                               pd.DataFrame([template]))
        return extracted_value
    except Exception as e:
        print("ERROR IN EXTRACTION!!!!")
        print(e)
        traceback.print_exc()
        return {}


# @app.route('/path_finder/test_templates', methods=['POST'])
def test_templates(request):
    """

    :return:
    """
    response_object = {}
    try:
        print("Request received!!!")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = request
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        print(content)
        vendor_id = content["vendor_id"]
        vendor_name = content["vendor_name"]
        list_document_id = content["list_document_id"]
        list_template = content["list_template"]

        print(list_document_id)
        print(list_template)

        list_extracted_value = []
        for document_id in list_document_id:
            print("Processing document:", document_id)
            DF = get_raw_prediction(document_id)
            list_extracted_fields = []
            for template in list_template:
                field_name = template["field_name"]
                template_csv = form_row_CSV(transform_template_UI_CSV(template))
                print(template_csv)
                extracted_value = get_extraction_for_template(DF, template_csv)[field_name]
                extracted_value["field_name"] = field_name
                list_extracted_fields.append(extracted_value)
            list_extracted_value.append({"document_id": document_id,
                                         "list_extracted_fields": list_extracted_fields})

        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object["extracted_value"] = list_extracted_value
        response_object['message'] = "Success"
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object["extracted_value"] = []
        response_object['message'] = "Failure"
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/path_finder/validate_template', methods=['POST'])
def validate_template(request):
    """

    :return:
    """
    response_object = {}
    try:
        print("Request received!!!")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = request
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        print(content)
        vendor_id = content["vendor_id"]
        vendor_name = content["vendor_name"]
        document_id = content["document_id"]
        template = content["template"]
        print(type(template))

        print("Request Parameters:", vendor_id, vendor_name, template, document_id)

        DF = get_raw_prediction(document_id)

        template_csv = form_row_CSV(transform_template_UI_CSV(template))

        print(template_csv)

        extracted_value = get_extraction_for_template(DF, template_csv)


        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object['extracted_value'] = extracted_value
        response_object['message'] = "Success"
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['extracted_value'] = {}
        response_object['message'] = "Failure"
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/path_finder/create_templates', methods=['POST'])
def insert_template(request):
    """

    :return:
    """
    response_object = {}
    try:
        print("Request received!!!")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = request
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        print(content)
        list_template = content["list_template"]
        print(type(list_template))

        print("Request Parameters:", list_template)

        list_template_CSV = []
        for l in list_template:
            dict_template = transform_template_UI_CSV(l)
            dict_row = form_row_CSV(dict_template)
            print("XXXXX:", dict_row)
            list_template_CSV.append(dict_row)

        status, message = insert_update_template_csv(list_template_CSV)

        if status:
            response_object['status'] = "Success"
            response_object['message'] = "Successfully Created!!!"
        else:
            response_object['status'] = "Failure"
            response_object['message'] = "Something went wrong!!!"
        response_object['responseCode'] = 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/path_finder/get_templates', methods=['POST'])
def get_templates(request):
    """

    :return:
    """
    response_object = {}
    try:
        print("Request received!!!")
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        # rawContent = request.content.read()
        # encodedContent = rawContent.decode("utf-8")
        # content = json.loads(encodedContent)
        content = request
        #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
        print(content)
        vendor_id = content["vendor_id"]
        vendor_name = content["vendor_name"]
        list_fields = content["list_fields"]
        document_id = content["document_id"]

        print("Request Parameters:", vendor_id, vendor_name, list_fields, document_id)

        DF = get_raw_prediction(document_id)

        if list_fields == "None":
            list_templates = get_templates_CSV(vendor_id)
        else:
            list_templates = get_templates_CSV(vendor_id, list(list_fields))

        print(list_templates)
        list_template_UI = []
        for template in list_templates:
            d_ = transform_template_CSV_UI(template)
            print("XXXXX", d_)
            extracted_value = get_extraction_for_template(DF, template)
            print(extracted_value)
            d_["extracted_value"] = extracted_value[d_["field_name"]]
            list_template_UI.append(d_)

        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object['templates'] = list_template_UI
        response_object['message'] = "Success"
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['templates'] = []
        response_object['message'] = "Failure"
        pass

    #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    # request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    response = json.dumps(response_object,
                          indent=4,
                          sort_keys=False,
                          default=str)
    print(type(response))
    return response


def main():
    """

    :return:
    """
    # vendor_masterdata = {'VENDOR_ID': 'TAO_4', 'CLIENT': 'TAO',
    #                      'VENDOR_NAME': 'ATLAS FOR MEN',
    #                      'IDENTIFIER_TEXT': 'ATLAS FOR MEN',
    #                      'DOCUMENT_TEXT': 'ATLAS FOR MEN', 'MATCH_SCORE': 1}
    #
    # DF = pd.read_csv("../AFM.csv")
    #
    # dict_UI = {'vendor_name': 'ATLAS FORMEN ', 'vendor_id': 'TAO_4', 'field_name': '_______Season', 'default_value': ' ', 'template_type': 'bounding_box_identifier', 'top': 'CATEGORY', 'include_top': 'NO', 'bottom': 'SIZES', 'include_bottom': 'YES', 'left': 'SEASON', 'right': 'VERSION', 'page_identifier': 'Not Applicable'}
    # #dict_UI = {'vendor_name': 'ATLAS FORMEN ', 'vendor_id': 'TAO_4', 'field_name': 'ItemCode', 'default_value': ' ', 'template_type': 'single_label', 'label': 'STYLE #', 'label_position': 'Left', 'field_shape': 'xxddd', 'field_location': 'NONE'}
    #
    list_dict_UI = [{'vendor_name': 'ATLAS FORMEN ', 'vendor_id': 'TAO_4', 'field_name': '_______Season', 'default_value': ' ', 'template_type': 'bounding_box_identifier', 'top': 'CATEGORY', 'include_top': 'NO', 'bottom': 'SIZES', 'include_bottom': 'YES', 'left': 'SEASON', 'right': 'VERSION', 'page_identifier': 'Not Applicable'},
                    {'vendor_name': 'ATLAS FORMEN ', 'vendor_id': 'TAO_4', 'field_name': 'ItemCode',
                     'default_value': ' ', 'template_type': 'bounding_box_identifier', 'top': 'CATEGORY',
                     'include_top': 'NO', 'bottom': 'SIZES', 'include_bottom': 'YES', 'left': 'SEASON',
                     'right': 'VERSION', 'page_identifier': 'RANDOM PAGE'}]

    list_dict_CSV = []
    for l in list_dict_UI:
        dict_template = transform_template_UI_CSV(l)
        dict_row = form_row_CSV(dict_template)
        print("XXXXX:", dict_row)
        list_dict_CSV.append(dict_row)

    insert_update_template_csv(list_dict_CSV)

    return

    # dict_template = transform_template_UI_CSV(dict_UI)
    # print(dict_template)
    # dic_row = form_row_CSV(dict_template)
    # print(dic_row)
    # TEMP = pd.DataFrame([dic_row])
    # TEMP.to_csv("../AAA.csv")
    # extracted_res = extract_vendor_specific_extra_fields(DF, vendor_masterdata, TEMP)
    # print(extracted_res)
    # print(insert_template_csv(dic_row))
    # return

    l = get_templates_CSV("TAO_1")
    for a in l:
        print(transform_template_CSV_UI(a))
    return



    # print(extract_vendor_specific_extra_fields(DF, vendor_masterdata))
    return


# if __name__ == "__main__":
#     #main()
#     app.run("0.0.0.0", 3333)