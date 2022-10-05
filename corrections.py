import pandas as pd
import requests
import json
import traceback
from TAPPconfig import getUIServer

# from klein import Klein
# app = Klein()


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


def form_oce_lines(DF):
    """

    """
    cols = ["page_num", "line_num", "line_text", "line_left", "line_top", "line_right", "line_down", "image_height",
            "image_widht"]

    TEMP = DF.sort_values(["page_num", "line_num"], ascending=[True, True]).drop_duplicates(cols)[cols]

    TEMP["line_top"] = (TEMP["line_top"] * TEMP["image_height"]).astype(int)
    TEMP["line_down"] = (TEMP["line_down"] * TEMP["image_height"]).astype(int)

    TEMP["line_left"] = (TEMP["line_left"] * TEMP["image_widht"]).astype(int)
    TEMP["line_right"] = (TEMP["line_right"] * TEMP["image_widht"]).astype(int)

    TEMP["ID"] = TEMP["page_num"].astype(str) + "_" + TEMP["line_num"].astype(str)

    del TEMP["image_height"]
    del TEMP["image_widht"]

    return TEMP.to_dict('records')


# @app.route('/corrections/getOCRLinesTemp', methods=['POST'])
def get_ocr_lines_temp(request):
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

        DF = get_raw_prediction(document_id)

        if DF is None:
            raise Exception("No record found for document!!")

        doc_lines = form_oce_lines(DF)

        doc_lines_updated = []
        for d in doc_lines:
            d["fieldValue"] = d["line_text"]
            d["editField"] = d["line_text"]
            d["boundingBox"] = {"left": d["line_left"],
                                "top": d["line_top"],
                                "right": d["line_right"],
                                "down": d["line_down"]}
            del d["line_left"]
            del d["line_top"]
            del d["line_right"]
            del d["line_down"]
            doc_lines_updated.append(d)

        response_object["documentId"] = document_id
        response_object["documentLines"] = doc_lines_updated

        response_object['status'] = "Success"
        response_object['responseCode'] = 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        pass

    request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    response = json.dumps(response_object, indent=4, sort_keys=False, default=str)
    return response


# @app.route('/corrections/getOCRLines', methods=['POST'])
def get_ocr_lines(request):
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
        print(content)
        document_id = content["document_id"]

        DF = get_raw_prediction(document_id)

        if DF is None:
            raise Exception("No record found for document!!")

        doc_lines = form_oce_lines(DF)

        response_object["documentId"] = document_id
        response_object["documentLines"] = doc_lines

        response_object['status'] = "Success"
        response_object['responseCode'] = 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response_object['status'] = "Failure"
        response_object['responseCode'] = 404
        pass

    #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    # request.responseHeaders.addRawHeader(b"content-type", b"application/json")
    #Changed by Hari Jul 20 - After making the postExtractionServicea flask app
    response = json.dumps(response_object,
                          indent=4,
                          sort_keys=False,
                          default=str)
    return response


# if __name__ == "__main__":
#     #main()
#     app.run("0.0.0.0", 4444)