import requests

import json
import pandas as pd

## Upload the same list of files
# Here: https://tapp2data.blob.core.windows.net/test/
masterdata_path = r"VENDOR_ADDRESS_MASTERDATA.csv"

#url = "http://13.71.23.200:8888/vendor/add"
url = "http://13.71.23.200:7766/vendor/add"

headers = {
  'Content-Type': 'application/json'
}

payload = {
            "ver": "1.0",
            "ts": 1571813791276,
            "params": {
                "msgid": ""
            },
            "request": {
            "vendorId":"",
            "name" : "",
            "address" : "",
            "logo" : "",
            "currency" : "",
            "firstInvoiceDate" : "",
            "lastInvoiceDate" : "",
            "lastInvoiceSubmittedOn" : "",
            "lastInvoiceProcessedOn" : "",
            "avgValuePerQuarter" : "",
            "avgInvoicesPerQuarter" : "",
            "xmlMapping" : {}
            }
        }

def addVendorToUI(VendorInfo):

    payload['request']['vendorId'] = VendorInfo[0]
    payload['request']['name'] = VendorInfo[1]
    payload['request']['address'] = VendorInfo[2]
    payload['request']['currency'] = VendorInfo[3]
    payload['request']['logo'] = VendorInfo[4]

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    print(VendorInfo[0]," -> ",response)

DF = pd.read_csv(masterdata_path,encoding='unicode_escape')

for index, row in DF.iterrows():
    try:
        vendorDetails = []
        vendorDetails.append(row['VENDOR_ID'])
        vendorDetails.append(row['vendorName'])
        vendorDetails.append(row['vendorAddress'])
        vendorDetails.append(row['currency'])
        vendorDetails.append(row['Logo'])
        addVendorToUI(vendorDetails)
    except Exception as e:
        print(vendorDetails, e)
