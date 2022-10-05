from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import traceback
import time
import json
import os
import pandas as pd
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")
'''
Authenticate
Authenticates your credentials and creates a client.
'''

subscription_key = "b907c0a273ed45eaaa1ac319ae59ff4f"
endpoint_url = "https://computer-vision-ocr-tao.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint_url, CognitiveServicesCredentials(subscription_key))

def get_OCR_values(file_path):
    """
    Function reads the ocr.json and creates a CSV file
    """

    print("get_OCR_values:", file_path)
    f = open(file_path, "r")
    o = f.read()
    f.close()
    j = json.loads(o)
    rows = []
    # resultList = j["analyzeResult"]["readResults"]
    resultList = j["analyze_result"]["read_results"]
    for resultobj in resultList:
        pageNo = resultobj["page"]
        lines = resultobj["lines"]
        width = resultobj["width"]
        height = resultobj["height"]
        lineNo = 0
        for line in lines:
            value = line["text"]
            # bb = line["boundingBox"]
            bb = line["bounding_box"]
            left = min(bb[0], bb[6]) / width
            right = max(bb[2], bb[4]) / width
            top = min(bb[1],bb[3]) / height
            down = max(bb[5],bb[7]) / height

            words = line["words"]
            wordNo = 0
            for word in words:
                row = {}
                wordValue = word["text"]
                wordConfidence = word["confidence"]
                # wordBB = word["boundingBox"]
                wordBB = word["bounding_box"]
                wordLeft = min(wordBB[0], wordBB[6]) / width
                wordRight = max(wordBB[2], wordBB[4]) / width
                wordTop = min(wordBB[1],wordBB[3]) / height
                wordDown = max(wordBB[5],wordBB[7]) / height
                row["OriginalFile"] = file_path[:-9]
                row["page_num"] = pageNo
                row["line_num"] = lineNo
                row["line_text"] = value
                row["line_left"] = round(left,3)
                row["line_right"] = round(right,3)
                row["line_top"] = round(top,3)
                row["line_down"] = round(down,3)

                row["line_height"] = round(down - top,3)
                row["line_width"] = round(right - left,3)

                row["word_num"] = wordNo
                row["text"] = wordValue
                row["conf"] = wordConfidence
                row["left"] = round(wordLeft,3)
                row["right"] = round(wordRight,3)
                row["top"] = round(wordTop,3)
                row["bottom"] = round(wordDown,3)
                row["height"] = round(wordDown - wordTop,3)
                row["width"] = round(wordRight - wordLeft,3)

                rows.append(row)
                wordNo = wordNo + 1

            lineNo = lineNo + 1

    DF = pd.DataFrame(rows)
    DF.to_csv("temp.csv")
    return DF


def run_OCR(img_path):
    try:
        print("Calling OCR for:", img_path)
        filename, file_extension = os.path.splitext(img_path)
        ocr_out_path = filename + ".json"

        read_image = open(img_path, "rb")
        # Call API with URL and raw response (allows you to get the operation location)
        read_response = computervision_client.read_in_stream(read_image, raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]
        # print("Generated Operation ID:", operation_id)
        # Call the "GET" API and wait for the retrieval of the results
        n_tries = 10
        n_try = 0
        wait_sec = 1

        while n_try < n_tries:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            time.sleep(wait_sec)
            n_try += 1

        if read_result.status == OperationStatusCodes.succeeded:
            ocr_json = read_result.as_dict()
            with open(ocr_out_path, 'w') as f:
                json.dump(ocr_json, f)
            print("OCR Done!!!")
            resultList = ocr_json["analyze_result"]["read_results"]
            angle_list = [page["angle"] for page in resultList]
            angle = float(angle_list[0]) if len(angle_list) > 0 else 0.0
            print("Identified angle:", angle)
            return ocr_out_path
        else:
            print("OCR Failed!!!!")
            return ocr_out_path
    except Exception as e:
        print("Inside Exception:", e)
        traceback.print_exc()
        pass
    print("##############################")


    return None


def find_date(s):
    """

    :param s:
    :return:
    """
    match = re.search(r"\d{2}/\d{2}/\d{4}", s)
    if match is not None:
        return match.group()
    else:
        return ""


def find_PAN(s):
    """

    :param s:
    :return:
    """
    l = re.findall(r"[A-Z]{5}[0-9]{4}[A-Z]", " "+s+" ")
    if len(l) > 0:
        return l[0]
    else:
        return ""

def find_Passpot_NO(s):
    """

    :param s:
    :return:
    """
    l = re.findall(r"[A-Z]{1}[0-9]{7}", " "+s+" ")
    if len(l) > 0:
        return l[0]
    else:
        return ""


def find_12_digit_num(s):
    """

    :param s:
    :return:
    """
    l = re.findall(r"\D(\d{12})\D", " "+s+" ")
    if len(l) > 0:
        return l[0]
    else:
        return ""


def extract_number(DF, num_type):
    """

    :param DF:
    :return:
    """
    DF['text'] = DF['text'].astype(str)

    lines = DF.sort_values(['page_num', 'line_num', 'word_num'],
                           ascending=[True, True, True]).groupby(['page_num', 'line_num']).agg(
        {'text': lambda x: "%s" % ' '.join(x),
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first'}).reset_index()

    lines['extracted_value'] = ""
    lines['text_without_spaces'] = lines['text'].str.replace(' ', '')

    if num_type == "AADHAR":
        lines['extracted_value'] = lines['text_without_spaces'].apply(find_12_digit_num)
    elif num_type == "PAN":
        lines['extracted_value'] = lines['text_without_spaces'].apply(find_PAN)
    elif num_type == "PASSPORT":
        lines['extracted_value'] = lines['text_without_spaces'].apply(find_Passpot_NO)
    elif num_type == "DATE":
        lines['extracted_value'] = lines['text_without_spaces'].apply(find_date)
    else:
        lines['extracted_value'] = ""
    lines = lines.loc[(lines['extracted_value'] != "")]

    set_extracted_value = set(lines['extracted_value'])

    if len(set_extracted_value) == 1:
        extracted_value = set_extracted_value.pop()
        lines = lines.loc[lines["extracted_value"] == extracted_value]
        return dict(lines.iloc[0][["extracted_value", "page_num", "conf", "left",
                                   "right", "top", "bottom"]])
    else:
        return None


def find_similarity(b, a):
    """

    :param a:
    :param b:
    :return:
    """

    if a in b:
        return 1.0
    return SequenceMatcher(None, str(a), str(b)).ratio()


def extract_name_from_PAN(DF):
    """

    :param DF:
    :return:
    """
    # Name is just the line above DOB
    DF['text'] = DF['text'].astype(str)
    DF['line_text'] = DF['line_text'].astype(str)
    D = DF[["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
            "line_down", "conf"]]
    lines = D.drop_duplicates(["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
                               "line_down"], keep='last')

    lines["match"] = lines["line_text"].apply(find_similarity, a = "INCOME TAX DEPARTMENT")

    l = lines.loc[lines["match"] >= 0.2]
    if l.empty:
        return None

    line_num_dob = dict(l.sort_values(['match', 'page_num', 'line_num'],
                                     ascending=[False, True, True]).iloc[0])

    lines_name = lines.loc[(lines['page_num'] == line_num_dob['page_num']) &
                          (lines['line_num'] > line_num_dob['line_num']) &
                           (lines['line_left'] <= line_num_dob['line_right']) &
                           (lines['line_right'] >= line_num_dob['line_left'])]
    lines_name.sort_values(['page_num', 'line_num'], ascending=[True, True],
                           inplace=True)

    if lines_name.shape[0] > 0:
        d = dict(lines_name.iloc[0][['line_text', 'page_num', "line_left",
                                     "line_right", "line_top",
                                     "line_down", "conf"]])
        d_ = {"extracted_value": d['line_text'],
              "page_num": d["page_num"],
              "conf": d["conf"],
              "left": d["line_left"],
              "right": d['line_right'],
              "top": d['line_top'],
              "bottom": d['line_down']}
        return d_

    return None


def extract_name_from_aadhar(DF):
    """

    :param DF:
    :return:
    """
    # Name is just the line above DOB
    DF['text'] = DF['text'].astype(str)
    DF['line_text'] = DF['line_text'].astype(str)
    D = DF[["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
            "line_down", "conf"]]
    lines = D.drop_duplicates(["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
            "line_down"], keep='last')

    lines["match"] = lines["line_text"].apply(find_similarity, a = "DOB")

    l = lines.loc[lines["match"] >= 0.2]

    if l.empty:
        return None

    line_num_dob = dict(l.sort_values(['match', 'page_num', 'line_num'],
                                     ascending=[False, True, True]).iloc[0])

    lines_name = lines.loc[(lines['page_num'] == line_num_dob['page_num']) &
                          (lines['line_num'] < line_num_dob['line_num']) &
                           (lines['line_left'] <= line_num_dob['line_right']) &
                           (lines['line_right'] >= line_num_dob['line_left'])]
    lines_name.sort_values(['page_num', 'line_num'], ascending=[True, False],
                           inplace=True)

    if lines_name.shape[0] > 0:
        d = dict(lines_name.iloc[0][['line_text', 'page_num',  "line_left",
                                     "line_right", "line_top",
                                     "line_down", "conf"]])
        d_ = {"extracted_value": d['line_text'],
              "page_num": d["page_num"],
              "conf": d["conf"],
              "left": d["line_left"],
              "right": d['line_right'],
              "top": d['line_top'],
              "bottom": d['line_down']}
        return d_

    return None

def extract_name_from_Passport(DF):
    """

    :param DF:
    :return:
    """
    print("extract_name_from_Passport")
    # Name is just the line above DOB
    DF['text'] = DF['text'].astype(str)
    DF['line_text'] = DF['line_text'].astype(str)
    D = DF[["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
            "line_down", "conf"]]
    lines = D.drop_duplicates(["page_num", "line_num", "line_text", "line_left", "line_right", "line_top",
                               "line_down"], keep='last')

    lines["match_given_name"] = lines["line_text"].apply(find_similarity, a = "Given Name")
    lines["match_sur_name"] = lines["line_text"].apply(find_similarity, a="Surname")
    lines["match_nationality"] = lines["line_text"].apply(find_similarity, a="Nationality")

    l_given_name = lines.loc[lines["match_given_name"] >= 0.4]
    l_sur_name = lines.loc[lines["match_sur_name"] >= 0.4]
    l_nationality = lines.loc[lines["match_nationality"] >= 0.4]

    l_given_name.sort_values(['match_given_name', 'page_num', 'line_num'],
                             ascending=[False, True, True],
                             inplace=True)

    l_sur_name.sort_values(['match_sur_name', 'page_num', 'line_num'],
                             ascending=[False, True, True],
                             inplace=True)

    l_nationality.sort_values(['match_nationality', 'page_num', 'line_num'],
                           ascending=[False, True, True],
                           inplace=True)

    # print(l_sur_name[['page_num', 'line_num', 'line_text']])
    # print(l_given_name[['page_num', 'line_num', 'line_text']])
    # print(l_nationality[['page_num', 'line_num', 'line_text']])
    given_name = None
    sur_name = None
    nationality = None
    if not l_given_name.empty:
        dict_ = dict(l_given_name.iloc[0])
        given_name = (dict_['page_num'], dict_['line_num'], dict_['line_top'],
                      dict_['line_down'])

    if not l_sur_name.empty:
        dict_ = dict(l_sur_name.iloc[0])
        sur_name = (dict_['page_num'], dict_['line_num'], dict_['line_top'],
                      dict_['line_down'])

    if not l_nationality.empty:
        dict_ = dict(l_nationality.iloc[0])
        nationality = (dict_['page_num'], dict_['line_num'], dict_['line_top'],
                      dict_['line_down'])

    # Extract SurName
    # print(sur_name)
    # print(given_name)
    # print(nationality)
    conf = 1.0
    page_num = 0
    left_ = 1.0
    right_ = 0.0
    top_ = 1.0
    bottom_ = 0.0

    extracted_surname = None
    if (sur_name is not None) & (given_name is not None):
        if sur_name[0] == given_name[0]:
            page_ = sur_name[0]
            start_line = sur_name[1] + 1
            end_line = given_name[1] - 1
            top = sur_name[3] * 0.98
            bottom = given_name[2] * 1.02
            l = lines.loc[(lines['page_num'] == page_) &
                          (lines['line_num'] >= start_line) &
                          (lines['line_num'] <= end_line) &
                          (lines['line_top'] >= top) &
                          (lines['line_down'] <= bottom)]
            if not l.empty:
                l.sort_values(['line_num'], ascending=[True], inplace=True)
                page_num = page_
                conf = min(conf, l['conf'].mean())
                left_ = min(left_, l['line_left'].min())
                right_ = max(right_, l['line_right'].max())
                top_ = min(top_, l['line_top'].min())
                bottom_ = max(bottom_, l['line_down'].max())
                extracted_surname = " ".join(list(l['line_text'])).strip(' \t\n\r')

    # Extract Given Name
    extracted_given_name = None
    if (given_name is not None) & (nationality is not None):
        if given_name[0] == nationality[0]:
            page_ = given_name[0]
            start_line = given_name[1] + 1
            end_line = nationality[1] - 1
            top = given_name[3] * 0.98
            bottom = nationality[2] * 1.02
            l = lines.loc[(lines['page_num'] == page_) &
                          (lines['line_num'] >= start_line) &
                          (lines['line_num'] <= end_line) &
                          (lines['line_top'] >= top) &
                          (lines['line_down'] <= bottom)]
            if not l.empty:
                l.sort_values(['line_num'], ascending=[True], inplace=True)
                page_num = page_
                conf = min(conf, l['conf'].mean())
                left_ = min(left_, l['line_left'].min())
                right_ = max(right_, l['line_right'].max())
                top_ = min(top_, l['line_top'].min())
                bottom_ = max(bottom_, l['line_down'].max())
                extracted_given_name = " ".join(list(l['line_text'])).strip(' \t\n\r')

    extracted_value = ""
    if extracted_given_name is not None:
        extracted_value += extracted_given_name
    if extracted_surname is not None:
        extracted_value += " "
        extracted_value += extracted_surname

    if extracted_value != "":
        return {"extracted_value": extracted_value,
                "page_num": page_num,
                "conf": conf,
                "left": left_,
                "right": right_,
                "top": top_,
                "bottom": bottom_}
    else:
        return None

def extract_DOB_Passport(DF):
    """

    :param DF:
    :return:
    """
    DF['text'] = DF['text'].astype(str)

    lines = DF.sort_values(['page_num', 'line_num', 'word_num'],
                           ascending=[True, True, True]).groupby(['page_num', 'line_num']).agg(
        {'text': lambda x: "%s" % ' '.join(x),
         'word_num': 'count',
         'left': 'min',
         'right': 'max',
         'top': 'min',
         'bottom': 'max',
         'conf': 'mean',
         'height': 'first',
         'width': 'first'}).reset_index()

    lines['extracted_value'] = ""
    lines['text_without_spaces'] = lines['text'].str.replace(' ', '')

    lines['extracted_value'] = lines['text_without_spaces'].apply(find_date)
    lines = lines.loc[(lines['extracted_value'] != "")]
    if lines.shape[0] > 0:
        lines["extracted_date"] = pd.to_datetime(lines['extracted_value'],
                                                 format="%d/%m/%Y")
        lines.sort_values(['extracted_date'], ascending=[True], inplace=True)
        lines['extracted_date'] = lines['extracted_date'].dt.strftime("%d/%m/%Y")
        return dict(lines.iloc[0][["extracted_value", "page_num", "conf", "left",
                                   "right", "top", "bottom"]])

    return None


def extract_data(DF):
    """

    :param img_path:
    :return:
    """
    # ocr_out_path = run_OCR(img_path)
    # DF = get_OCR_values(ocr_out_path)

    extracted_aadhar = extract_number(DF, "AADHAR")
    extracted_PAN = extract_number(DF, "PAN")
    extracted_Passport_NO = extract_number(DF, "PASSPORT")

    extracted_data = {"doc_type": "UNKNOWN",
                      "doc_number": None,
                      "DOB": None,
                      "NAME": None}

    if (extracted_aadhar is not None) & (extracted_PAN is None) & (extracted_Passport_NO is None):
        extracted_data["doc_type"] = "AADHAR"
        extracted_data["doc_number"] = extracted_aadhar
    elif (extracted_aadhar is None) & (extracted_PAN is not None) & (extracted_Passport_NO is None):
        extracted_data["doc_type"] = "PAN"
        extracted_data["doc_number"] = extracted_PAN
    elif (extracted_aadhar is None) & (extracted_PAN is None) & (extracted_Passport_NO is not None):
        extracted_data["doc_type"] = "PASSPORT"
        extracted_data["doc_number"] = extracted_Passport_NO

    if extracted_data["doc_type"] == "PASSPORT":
        extracted_data["DOB"] = extract_DOB_Passport(DF)
    else:
        extracted_data["DOB"] = extract_number(DF, "DATE")

    print(extracted_data)
    if extracted_data["doc_type"] == "AADHAR":
        extracted_data["NAME"] = extract_name_from_aadhar(DF)
    elif extracted_data["doc_type"] == "PAN":
        extracted_data["NAME"] = extract_name_from_PAN(DF)
    elif extracted_data["doc_type"] == "PASSPORT":
        extracted_data["NAME"] = extract_name_from_Passport(DF)

    print(extracted_data)
    return extracted_data


def main():
    """

    :return:
    """
    img_paths = ["Aadhar_Samples/1.jpg",
                 "Aadhar_Samples/2.jpg",
                 "Aadhar_Samples/3.jpg",
                 "Aadhar_Samples/4.jpg",
                 "Aadhar_Samples/5.jpg",
                 "Aadhar_Samples/6.jpg",
                 "Aadhar_Samples/7.jpg",
                 "Aadhar_Samples/8.jpg",
                 "Aadhar_Samples/9.jpg",
                 "Aadhar_Samples/10.jpg",
                 "Aadhar_Samples/11.jpg",
                 "PAN_Samples/1.jpg",
                 "PAN_Samples/2.jpg",
                 "PAN_Samples/3.jpg",
                 "PAN_Samples/4.jpg",
                 "PAN_Samples/5.jpg",
                 "PAN_Samples/6.jpg",
                 "PAN_Samples/7.jpg",
                 "PAN_Samples/8.jpg",
                 "PAN_Samples/9.jpg",
                 "PAN_Samples/10.jpg",
                 "Passport_Samples/front_1.jpg",
                 "Passport_Samples/front_2.jpg",
                 "Passport_Samples/front_3.jpg",
                 "Passport_Samples/front_4.jpg",
                 "Passport_Samples/front_5.jpg"]

    # img_paths = ["Passport_Samples/front_1.jpg"]
    dict_df = []
    for i in img_paths:
        try:
            extraction = extract_data(i)
            extraction["img_path"] = i
            dict_df.append(extraction)
        except Exception as e:
            print(e)
            pass
    DD = pd.DataFrame(dict_df)
    DD.to_csv("output_.csv")

if __name__ == "__main__":
    main()