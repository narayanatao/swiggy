import pandas as pd
import os
from csv import DictWriter
import re
import preProcUtilities as putil
import json
import traceback
import operator
import requests
from TAPPconfig import getUIServer

# from klein import Klein
# app = Klein()

script_dir = os.path.dirname(__file__)
master_file_csv_path = os.path.join(script_dir,
                              "Utilities/VENDOR_ADDRESS_MASTERDATA.csv")

client = "SWIGGY"
CONST = putil.getPostProcessConstantLabels()
CONST['vendor_master_data_cutoff_score'] = 0.6
print("MasterData CutOff Score:", CONST['vendor_master_data_cutoff_score'])

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

csv_delimiter_fixer(master_file_csv_path)
# Code needs to be commented: End


def form_call_back_url():
    """

    :return:
    """
    # call_back_url = "http://52.172.231.99:8888"
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


def get_vendor(DF):
    """

    :return:
    """
    if DF is None:
        return None, None, None

    s = form_document_text(DF)
    VENDOR_MASTERDATA = read_masterdata_csv()
    dict_score = {}
    for idx, row in VENDOR_MASTERDATA.iterrows():
        f = row['VENDOR_ID']
        identifier_text = str(row['IDENTIFIER_TEXT'])
        score = calculate_score(identifier_text, s)
        dict_score[f] = score

    if len(dict_score) == 0:
        return None, None, None
    predicted_format = max(dict_score.items(), key=operator.itemgetter(1))[0]
    max_score = dict_score[predicted_format]
    if max_score > CONST['vendor_master_data_cutoff_score']:
        dict_vendor_data = VENDOR_MASTERDATA.loc[VENDOR_MASTERDATA['VENDOR_ID']
                                                 == predicted_format].iloc[0].to_dict()
        return predicted_format, max_score, dict_vendor_data
    else:
        return None, None, None


# @app.route('/format_identifier/refresh_format', methods=['POST'])
def refresh_format(request):
    """

    :param request:
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
        list_document_id = content["list_document_id"]
        list_response = []
        for doc in list_document_id:
            try:
                DF = get_raw_prediction(doc)
                format_, vendor_masterdata_score, vendor_masterdata = get_vendor(DF)
                print(format_, vendor_masterdata_score, vendor_masterdata)
                if format_ is not None:
                    del vendor_masterdata["DOCUMENT_TEXT"]
                    vendor_masterdata['MATCH_SCORE'] = vendor_masterdata_score
                    vendor_masterdata["document_id"] = doc
                    list_response.append({doc: vendor_masterdata})
                else:
                    list_response.append({doc: {"document_id": doc}})
            except Exception as e:
                print(e)
                list_response.append({doc: {"document_id": doc}})
                pass
        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object['refreshed_result'] = list_response
    except Exception as e:
        print(e)
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['refreshed_result'] = None
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/format_identifier/get_suggestion', methods=['POST'])
def get_suggested_masterdata(request):
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
        document_id = content["document_id"]

        # bytes_data = request.args[b"PRED_FILE"][0]
        # with open("temp.csv", 'wb') as f:
        #     f.write(bytes_data)
        # DF = pd.read_csv('temp.csv', index_col=0)

        DF = get_raw_prediction(document_id)

        format_, vendor_masterdata_score, vendor_masterdata = get_vendor(DF)
        print(format_, vendor_masterdata_score, vendor_masterdata)

        if format_ is not None:
            response_object['status'] = "Success"
            response_object['responseCode'] = 200
            del vendor_masterdata["DOCUMENT_TEXT"]
            vendor_masterdata['MATCH_SCORE'] = vendor_masterdata_score
            response_object['master_data'] = vendor_masterdata
        else:
            response_object['status'] = "Success"
            response_object['responseCode'] = 200
            response_object['master_data'] = {}
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['master_data'] = None
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


def read_masterdata_csv():
    """
    Read MasterData from CSV file
    CSV
    """
    VENDOR_MASTERDATA = pd.read_csv(master_file_csv_path, encoding='unicode_escape')
    VENDOR_MASTERDATA = VENDOR_MASTERDATA.loc[VENDOR_MASTERDATA["CLIENT"] == client]
    return VENDOR_MASTERDATA


def form_document_text(DF):
    """
    Form upper-case string stream of document text
    """
    l = list(DF['text'])
    l = [str(x) for x in l]
    s = " ".join(l)
    s = s.upper()
    return s


def generate_id(list_ids):
    """
    Generates next ID for insertion
    ID: client + "_" + <next number in the sequence>
    """
    id_prefix = client + "_"
    if len(list_ids) == 0:
        return id_prefix + str(1)

    # list_ids = [int(text.removeprefix(id_prefix)) for text in list_ids]
    list_ids = [int(text[len(id_prefix):]) for text in list_ids]
    generated_id = id_prefix + str(max(list_ids)+1)
    return generated_id


def form_masterdata_for_insertion(VENDOR_MASTERDATA, vendor_name, identifier_text, DF):
    """
    Format:
    {'VENDOR_ID': 'KGS_1',
    'CLIENT': 'KGS',
    'VENDOR_SPECIFIC': 0,
    'VENDOR_NAME': 'La Redoute',
    'IDENTIFIER_TEXT': 'La Redoute La Redoute La Redoute La Redoute',
    'DOCUMENT_TEXT': <document text>}
    """

    document_text = form_document_text(DF)
    l = list(VENDOR_MASTERDATA['VENDOR_ID'])
    vendor_id = generate_id(l)
    identifier_text = identifier_text.upper()
    match_score = calculate_score(identifier_text, document_text)
    dict_masterdata_insert = {'VENDOR_ID': vendor_id,
                              'CLIENT': client,
                              'VENDOR_NAME': vendor_name,
                              'IDENTIFIER_TEXT': identifier_text,
                              'DOCUMENT_TEXT': document_text,
                              'MATCH_SCORE': match_score}

    return dict_masterdata_insert


def insert_masterdata_csv(dict_md):
    """
    Add entry for new MasterData after all the validations have passed
    This code is for CSV insertion
    """
    try:
        csv_delimiter_fixer(master_file_csv_path)
        with open(master_file_csv_path, 'a', newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=list(dict_md.keys()))
            dictwriter_object.writerow(dict_md)
            f_object.close()

        return True, "Successful Created!!"
    except Exception as e:
        print(e)
        traceback.print_exc()
        pass
        return False, "Failure"


def calculate_score(identifier_text, document_text):
    """
    Calculate match score between identifier_text and document_text
    """
    l_ = re.split(r'[^\w]', str(identifier_text))
    l_ = [x.upper() for x in l_ if ((x != '') and (x != 'nan'))]
    l_ = [x for x in l_ if (len(x) > 1)]
    if len(l_) == 0:
        return 0.0
    matches = [x for x in l_ if x in document_text]
    score = len(matches)/len(l_)
    return round(score, 2)


def calculate_score_list(identifier_text, VENDOR_MASTERDATA):
    """

    :return:
    """
    dict_scores = {}
    for idx, row in VENDOR_MASTERDATA.iterrows():
        vendor_id = row["VENDOR_ID"]
        document_text = row["DOCUMENT_TEXT"]
        score = calculate_score(identifier_text, document_text)
        dict_scores[vendor_id] = score

    return dict_scores


def validate_masterdata(VENDOR_MASTERDATA, identifier_text, DF):
    """

    :return:
    """
    document_text = form_document_text(DF)

    self_score = calculate_score(identifier_text, document_text)
    dict_scores = calculate_score_list(identifier_text, VENDOR_MASTERDATA)
    max_score = 0.0
    if len(dict_scores) > 0:
        max_score = max(dict_scores.values())

    print("MasterData Cutoff Score:", CONST['vendor_master_data_cutoff_score'])
    print("Self Score:", self_score)
    print("Other Scores:", dict_scores)
    print("Max Score:", max_score)


    if self_score <= CONST['vendor_master_data_cutoff_score']:
        return False, self_score, "Identifier Text doesn't seem to match with Document Text!!"
        # if self_score == 0.0:
        #     return False, self_score, "Identifier Text seems to be too generic (matches with other Formats).Please enter unique words!!"
        # else:
        #     return False, self_score, "Identifier Text doesn't seem to match with Document Text!!"

    if max_score >= self_score:
        return False, 0.0, "Identifier Text seems to be too generic (matches with other Formats).Please enter unique words!!"

    return True, self_score, "Proceed and Save!!"


def main():
    """
    Script needs input folder path to iterate over
    :return:
    """

    print(get_raw_prediction("doc_1646222031617_99b988ab98b"))
    return
    # VENDOR_MASTERDATA = read_masterdata_csv()
    DF = pd.read_csv("../AMERIMARK.csv")
    # identifier_text = "La Redoute La Redoute La Redoute La Redoute"
    # identifier_text = "ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS ATLAS"
    # identifier_text = "ANTHONY RICHARDS ANTHONY RICHARDS ANTHONY RICHARDS ANTHONY RICHARDS ANTHONY RICHARDS ANTHONY RICHARDS"
    # identifier_text = "size size size size size size"
    #
    # VENDOR_MASTERDATA = read_masterdata_csv()
    #
    # status, msg = validate_masterdata(VENDOR_MASTERDATA, identifier_text, DF)
    #
    # if status:
    #     preprocess_vendor = "DUMMY"
    #     vendor_name = "DUMMY"
    #     dict_md = form_masterdata_for_insertion(VENDOR_MASTERDATA, preprocess_vendor,
    #         0, vendor_name, identifier_text, DF)
    #     print(list(dict_md.keys()))
    #     insert_masterdata_csv(dict_md)
    #     print("Successfully created MasterData!!!")
    # else:
    #     print(msg)
    print(get_vendor(DF))
    print("Exit Main!!!!!")


# @app.route('/format_identifier/create', methods=['POST'])
def insert_masterdata_wrapper(request):
    """

    :param request:
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
        identifier_text = content["identifier_text"]
        vendor_name = content["vendor_name"]
        document_id = content["document_id"]

        # identifier_text = request.args[b"IDENTIFIER_TEXT"][0].decode("utf-8")
        # print("Identifier text:", identifier_text)
        #
        # vendor_name = request.args[b"VENDOR_NAME"][0].decode("utf-8")
        # print("Vendor Name:", vendor_name)
        #
        # bytes_data = request.args[b"PRED_FILE"][0]
        # with open("temp.csv", 'wb') as f:
        #     f.write(bytes_data)
        # DF = pd.read_csv('temp.csv', index_col=0)

        DF = get_raw_prediction(document_id)

        VENDOR_MASTERDATA = read_masterdata_csv()
        dict_md = form_masterdata_for_insertion(VENDOR_MASTERDATA, vendor_name,
                                                identifier_text, DF)
        print("Inserting:", dict_md)
        status, msg = insert_masterdata_csv(dict_md)
        del dict_md["DOCUMENT_TEXT"]
        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object['message'] = msg
        response_object['master_data'] = dict_md
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['message'] = "Failure"
        response_object['master_data'] = None
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/format_identifier/validate', methods=['POST'])
def validate_masterdata_wrapper(request):
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
        identifier_text = content["identifier_text"]
        vendor_name = content["vendor_name"]
        document_id = content["document_id"]


        # identifier_text = request.args[b"IDENTIFIER_TEXT"][0].decode("utf-8")
        # print("Identifier text:", identifier_text)
        #
        # vendor_name = request.args[b"VENDOR_NAME"][0].decode("utf-8")
        # print("Vendor Name:", vendor_name)
        #
        # bytes_data = request.args[b"PRED_FILE"][0]
        # with open("temp.csv", 'wb') as f:
        #     f.write(bytes_data)
        # DF = pd.read_csv('temp.csv', index_col=0)

        DF = get_raw_prediction(document_id)

        VENDOR_MASTERDATA = read_masterdata_csv()

        status, score, msg = validate_masterdata(VENDOR_MASTERDATA, identifier_text, DF)
        if status:
            response_object['validate_result'] = "VALID"
        else:
            response_object['validate_result'] = "INVALID"
        response_object['status'] = "Success"
        response_object['responseCode'] = 200
        response_object['score'] = score
        response_object['message'] = msg
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['validate_result'] = "INVALID"
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        response_object['score'] = 0.0
        response_object['message'] = "Failure"
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# if __name__ == "__main__":
#     #main()
#     app.run("0.0.0.0", 2222)


