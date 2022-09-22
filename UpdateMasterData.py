# -*- coding: utf-8 -*-
import csv
from inspect import ArgSpec
import os
import sys
import copy
import pandas as pd
from pip import main
import numpy as np
import traceback
from azure.storage.blob import BlobServiceClient
from time import strftime
import warnings
warnings.filterwarnings("ignore")
import TAPPconfig as cfg
rootFolderPath = cfg.getRootFolderPath()

def download_MasterData_from_blob(file_name,dwonload_folder):
    try:
        account_name = 'submasterstorage'
        connect_str = "DefaultEndpointsProtocol=https;AccountName=submasterstorage;AccountKey=q061i+V0b9DGdgf+YWef1R110gZbKblmTKhWVN5XmeZm4rfl7ATOrPHm40SfxQQHq/Sfc3+Q+NMBQcqow/uYjQ==;EndpointSuffix=core.windows.net"
        ocr_pred_container = "swiggy-attachments"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        source_container_client = blob_service_client.get_container_client(ocr_pred_container)
        source_blobs = source_container_client.list_blobs()

        for b in source_blobs:
            if b["name"].endswith("xlsx"):
                blob_name = os.path.basename(b['name'])
                # print("fileName :",blob_name )
                if blob_name == file_name:
                    # print("Blob name matched with given file Name.")
                    download_path =  os.path.join(dwonload_folder,file_name)
                    blob_client = blob_service_client.get_blob_client(container=ocr_pred_container,blob=file_name)
                    with open(download_path, "wb") as download_file:
                        try:
                            download_file.write(blob_client.download_blob().readall())
                            print("Download successfull at :",download_path)
                            # print("Blob serch done!")
                            return True

                        except:
                            print("failed to download:",download_path)
                            # print("Blob serch done!")
                            return True
        print("Blob not found.")
        return False
    except:
        print("Blob dwonload Exception :",traceback.print_exc())
        return False
 
def generate_id(list_ids):
    """
    Generates next ID for insertion
    ID: client + "_" + <next number in the sequence>
    """
    client = "SWIGGY"
    id_prefix = client + "_"
    if len(list_ids) == 0:
        return id_prefix + str(1)

    # list_ids = [int(text.removeprefix(id_prefix)) for text in list_ids]
    list_ids = [int(text[len(id_prefix):]) for text in list_ids]
    generated_id = id_prefix + str(max(list_ids)+1)
    return generated_id


def update_existing_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_BUYRES_MASTERDATA):
    CURRENT_VENDOR_MASTER_COPY = copy.deepcopy(CURRENT_VENDOR_MASTER)
    try:
        NEW_VENDOR_MASTER = NEW_VENDOR_MASTER[(~NEW_VENDOR_MASTER["VENDOR_GSTIN"].duplicated()) | NEW_VENDOR_MASTER["VENDOR_GSTIN"].isna()]
        # print("Client_shared DF shape after removing GSTIN duplicates :",NEW_VENDOR_MASTER.shape) 
        NEW_VENDOR_MASTER = NEW_VENDOR_MASTER.replace(np.nan, "", regex=True)
        #NEW_VENDOR_MASTER["VENDOR_GSTIN"] = NEW_VENDOR_MASTER["VENDOR_GSTIN"].replace(np.nan,'')
        #print("Client_shared DF shape after removing GSTIN duplicates :",NEW_VENDOR_MASTER.tail()) 
        NEW_VENDOR_MASTER["VENDOR_NAME"] = NEW_VENDOR_MASTER["VENDOR_NAME"].str.upper()
        non_GSTIN =  NEW_VENDOR_MASTER[NEW_VENDOR_MASTER["VENDOR_GSTIN"]=='']
        # print("non_GSTIN data shape :",non_GSTIN.shape)
        non_GSTIN.drop_duplicates(keep=False,inplace=True)
        # print("non_GSTIN data shape after rm duplicates :",non_GSTIN.shape)
        NEW_VENDOR_MASTER = NEW_VENDOR_MASTER[NEW_VENDOR_MASTER["VENDOR_GSTIN"]!='']
        NEW_VENDOR_MASTER = pd.concat([NEW_VENDOR_MASTER,non_GSTIN])    
        print("Current Vendor Master Data shape :",CURRENT_VENDOR_MASTER.shape)       
        print("New Vendor Master Data shape :",NEW_VENDOR_MASTER.shape)       
        update_count = 0
        # updating/ Refreshing Vendor Names
        for idx,row in CURRENT_VENDOR_MASTER_COPY.iterrows():
            for i,r in NEW_VENDOR_MASTER.iterrows():
                if row["VENDOR_GSTIN"] != '':
                    if row["VENDOR_GSTIN"] ==r["VENDOR_GSTIN"]:
                        if row["VENDOR_NAME"] != r["VENDOR_NAME"]:
                            # print("b4r updating :",CURRENT_VENDOR_MASTER["VENDOR_NAME"][idx])
                            CURRENT_VENDOR_MASTER.at[idx,"VENDOR_NAME"] = r["VENDOR_NAME"]
                            #CURRENT_VENDOR_MASTER.loc[(CURRENT_VENDOR_MASTER[idx,"VENDOR_NAME"])] = r["VENDOR_NAME"]
                            # print("after updating :",CURRENT_VENDOR_MASTER["VENDOR_NAME"][idx])
                            update_count = update_count +1
        print("No of records updated :",update_count)
        print("Adding new records to current vendor master")       
        ENTITY_GSTINS = set(CURRENT_BUYRES_MASTERDATA["GSTIN"].to_list())
        VENDOR_GSTINS = set(CURRENT_VENDOR_MASTER["VENDOR_GSTIN"].to_list())
        V_NO_GSTIN = CURRENT_VENDOR_MASTER[CURRENT_VENDOR_MASTER["VENDOR_GSTIN"]=='']
        VENDOR_NAMES = set(V_NO_GSTIN["VENDOR_NAME"].to_list())
        #print("CLIENT_DF shape :",CLIENT_DF.shape)
        add_count = 0
        for _, row in NEW_VENDOR_MASTER.iterrows():
            vendor_name = str(row["VENDOR_NAME"])
            gst_num = str(row["VENDOR_GSTIN"])
            if (gst_num!="") and ((gst_num not in VENDOR_GSTINS) and (gst_num not in ENTITY_GSTINS)):
                # print("this record is not present in our existing masterdata",gst_num)
                New_Vendor_ID = generate_id(list(CURRENT_VENDOR_MASTER["VENDOR_ID"]))
                row = {"VENDOR_ID": New_Vendor_ID,
                       "CLIENT":"SWIGGY", 
                       "VENDOR_GSTIN": gst_num, 
                       "VENDOR_NAME":vendor_name, 
                       "IDENTIFIER_TEXT":"", 
                       "DOCUMENT_TEXT":"", 
                       "MATCH_SCORE": ""
                      }
                # print("VENDOR_MASTERDATA shape b4r adding new record :",CURRENT_VENDOR_MASTER.shape)
                CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.append(row,ignore_index = True)
                # print("VENDOR_MASTERDATA shape after adding new record :",CURRENT_VENDOR_MASTER.shape)
                add_count = add_count + 1
            if (gst_num ==""):
                # print("gst_num :",gst_num)
                for name in VENDOR_NAMES:
                    if name != vendor_name:
                        # print("name :",name ,"\tvendor_name :",vendor_name)
                        New_Vendor_ID = generate_id(list(CURRENT_VENDOR_MASTER["VENDOR_ID"]))
                        row = {"VENDOR_ID": New_Vendor_ID,
                               "CLIENT":"SWIGGY", 
                               "VENDOR_GSTIN": "", 
                               "VENDOR_NAME":vendor_name, 
                               "IDENTIFIER_TEXT":"", 
                               "DOCUMENT_TEXT":"", 
                               "MATCH_SCORE": ""
                              }
                        # print("VENDOR_MASTERDATA shape b4r adding new Name :",CURRENT_VENDOR_MASTER.shape)
                        CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.append(row,ignore_index = True)
                        # print("VENDOR_MASTERDATA shape after adding new Name :",CURRENT_VENDOR_MASTER.shape)
                        add_count = add_count + 1
        print("No of new records added :",add_count)
        print("Finding and adding new records Done!")
        return CURRENT_VENDOR_MASTER 
    except:
        print("Master data update exception :",traceback.print_exc())
        return CURRENT_VENDOR_MASTER_COPY

def adding_new_record_into_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_BUYRES_MASTERDATA):

    VENDOR_MASTERDATA_COPY = copy.deepcopy(CURRENT_VENDOR_MASTER)
    try:
        ENTITY_GSTINS = set(CURRENT_BUYRES_MASTERDATA["GSTIN"].to_list())
        VENDOR_GSTINS = set(CURRENT_VENDOR_MASTER["VENDOR_GSTIN"].to_list())
        V_NO_GSTIN = CURRENT_VENDOR_MASTER[CURRENT_VENDOR_MASTER["VENDOR_GSTIN"]=='']
        VENDOR_NAMES = set(V_NO_GSTIN["VENDOR_NAME"].to_list())
        #print("CLIENT_DF shape :",CLIENT_DF.shape)
        add_count = 0
        for _, row in NEW_VENDOR_MASTER.iterrows():
            vendor_name = str(row["VENDOR_NAME"])
            gst_num = str(row["VENDOR_GSTIN"])
            if (gst_num!="") and ((gst_num not in VENDOR_GSTINS) and (gst_num not in ENTITY_GSTINS)):
                # print("this record is not present in our existing masterdata",gst_num)
                New_Vendor_ID = generate_id(list(CURRENT_VENDOR_MASTER["VENDOR_ID"]))
                row = {"VENDOR_ID": New_Vendor_ID,
                       "CLIENT":"SWIGGY", 
                       "VENDOR_GSTIN": gst_num, 
                       "VENDOR_NAME":vendor_name, 
                       "IDENTIFIER_TEXT":"", 
                       "DOCUMENT_TEXT":"", 
                       "MATCH_SCORE": ""
                      }
                # print("VENDOR_MASTERDATA shape b4r adding new record :",CURRENT_VENDOR_MASTER.shape)
                CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.append(row,ignore_index = True)
                # print("VENDOR_MASTERDATA shape after adding new record :",CURRENT_VENDOR_MASTER.shape)
                add_count = add_count + 1
            if (gst_num ==""):
                # print("gst_num :",gst_num)
                for name in VENDOR_NAMES:
                    if name != vendor_name:
                        # print("name :",name ,"\tvendor_name :",vendor_name)
                        New_Vendor_ID = generate_id(list(CURRENT_VENDOR_MASTER["VENDOR_ID"]))
                        row = {"VENDOR_ID": New_Vendor_ID,
                               "CLIENT":"SWIGGY", 
                               "VENDOR_GSTIN": "", 
                               "VENDOR_NAME":vendor_name, 
                               "IDENTIFIER_TEXT":"", 
                               "DOCUMENT_TEXT":"", 
                               "MATCH_SCORE": ""
                              }
                        # print("VENDOR_MASTERDATA shape b4r adding new Name :",CURRENT_VENDOR_MASTER.shape)
                        CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.append(row,ignore_index = True)
                        # print("VENDOR_MASTERDATA shape after adding new Name :",CURRENT_VENDOR_MASTER.shape)
                        add_count = add_count + 1
        print("No of new records added :",add_count)
        print("Finding and adding new records Done!")
        return CURRENT_VENDOR_MASTER
    except:
        print("Finding and adding new records exception :",traceback.print_exc())
        return VENDOR_MASTERDATA_COPY

#### updating Entity Master Data
def update_existing_enity_master(CURRENT_ENTITY_MASTER,NEW_ENTITY_MASTER):
    CURRENT_ENTITY_MASTER_COPY = copy.deepcopy(CURRENT_ENTITY_MASTER)
    try:
        print("CURRENT_ENTITY_MASTER shape :",CURRENT_ENTITY_MASTER.shape) 

        print("NEW_ENTITY_MASTER shape :",NEW_ENTITY_MASTER.shape) 

        NEW_ENTITY_MASTER = NEW_ENTITY_MASTER[(~NEW_ENTITY_MASTER["GSTN"].duplicated()) | NEW_ENTITY_MASTER["GSTN"].isna()]
        print("NEW_ENTITY_MASTER shape after removing GSTIN duplicates :",NEW_ENTITY_MASTER.shape) 
        NEW_ENTITY_MASTER = NEW_ENTITY_MASTER.replace(np.nan, "", regex=True)
        NEW_ENTITY_MASTER["Entity "] = NEW_ENTITY_MASTER["Entity "].str.upper()
        NON_GSTIN_ENTITY =  NEW_ENTITY_MASTER[NEW_ENTITY_MASTER["GSTN"]=='']
        print("NON_GSTIN_ENTITY data shape :",NON_GSTIN_ENTITY.shape)
        NON_GSTIN_ENTITY.drop_duplicates(keep=False,inplace=True)
        print("NON_GSTIN_ENTITY data shape after removing Entity Name duplicates :",NON_GSTIN_ENTITY.shape)
        NEW_ENTITY_MASTER = NEW_ENTITY_MASTER[NEW_ENTITY_MASTER["GSTN"]!='']
        NEW_ENTITY_MASTER = pd.concat([NEW_ENTITY_MASTER,NON_GSTIN_ENTITY])           
        update_count = 0
        # updating/ Refreshing Vendor Names
        for idx,row in CURRENT_ENTITY_MASTER_COPY.iterrows():
            for i,r in NEW_ENTITY_MASTER.iterrows():
                if row["GSTIN"] != '':
                    if row["GSTIN"] ==r["GSTN"]:
                        if row["NAME"] != r["Entity "]:
                            #print("b4r updating :",CURRENT_ENTITY_MASTER["NAME"][idx])
                            CURRENT_ENTITY_MASTER.at[idx,"NAME"] = r["Entity "]
                            #print("after updating :",CURRENT_ENTITY_MASTER["NAME"][idx])
                            update_count = update_count + 1
        print("No of Records updated :",update_count)
        return CURRENT_ENTITY_MASTER
    except:
        print("Master data update exception :",traceback.print_exc())
        return CURRENT_ENTITY_MASTER_COPY

## Adding new records into intity master data
def adding_new_record_into_entity_master(CURRENT_ENTITY_MASTER,NEW_ENTITY_MASTER,CURRENT_VENDOR_MASTER):
    
    CURRENT_ENTITY_MASTER_COPY = copy.deepcopy(CURRENT_ENTITY_MASTER)
    print("CURRENT_ENTITY_MASTER_COPY :",CURRENT_ENTITY_MASTER_COPY.shape)
    try:
        ENTITY_GSTINS = set(NEW_ENTITY_MASTER["GSTN"].to_list())
        VENDOR_GSTINS = set(CURRENT_VENDOR_MASTER["VENDOR_GSTIN"].to_list())
        NON_GSTIN_ENTITY = NEW_ENTITY_MASTER[NEW_ENTITY_MASTER["GSTN"]=='']
        NON_GSTIN_VENDOR = CURRENT_VENDOR_MASTER[CURRENT_VENDOR_MASTER["VENDOR_GSTIN"]=='']
        ENTITY_NAMES = set(NON_GSTIN_ENTITY["Entity "].to_list())
        VENDOR_NAMES = set(NON_GSTIN_VENDOR["VENDOR_NAME"].to_list())
        print("NEW_ENTITY_MASTER shape :",NEW_ENTITY_MASTER.shape)
        new_record_count = 0
        for _, row in NEW_ENTITY_MASTER.iterrows():
            vendor_name = str(row["Entity "])
            gst_num = str(row["GSTN"])
            # print("gst_num :",gst_num)
            if (gst_num!="") and ((gst_num not in VENDOR_GSTINS) and (gst_num not in ENTITY_GSTINS)):
                print("this record is not present in our existing masterdata",gst_num)
                New_Vendor_ID = generate_id(list(CURRENT_ENTITY_MASTER["VENDOR_ID"]))
                row = {"VENDOR_ID": New_Vendor_ID,
                       "CLIENT":"SWIGGY", 
                       "VENDOR_GSTIN": gst_num, 
                       "VENDOR_NAME":vendor_name, 
                       "IDENTIFIER_TEXT":"", 
                       "DOCUMENT_TEXT":"", 
                       "MATCH_SCORE": ""
                      }
                print("VENDOR_MASTERDATA shape b4r adding new record :",CURRENT_ENTITY_MASTER.shape)
                CURRENT_ENTITY_MASTER = CURRENT_ENTITY_MASTER.append(row,ignore_index = True)
                print("VENDOR_MASTERDATA shape after adding new record :",CURRENT_ENTITY_MASTER.shape)
                new_record_count = new_record_count + 1

            if (gst_num ==""):
                for name in VENDOR_NAMES:
                    if name != vendor_name:
                        print("name :",name ,"\tvendor_name :",vendor_name)
                        New_Vendor_ID = generate_id(list(CURRENT_ENTITY_MASTER["VENDOR_ID"]))
                        row = {"VENDOR_ID": New_Vendor_ID,
                               "CLIENT":"SWIGGY", 
                               "VENDOR_GSTIN": "", 
                               "VENDOR_NAME":vendor_name, 
                               "IDENTIFIER_TEXT":"", 
                               "DOCUMENT_TEXT":"", 
                               "MATCH_SCORE": ""
                              }
                        #print("VENDOR_MASTERDATA shape b4r adding new Name :",CURRENT_ENTITY_MASTER.shape)
                        CURRENT_ENTITY_MASTER = CURRENT_ENTITY_MASTER.append(row,ignore_index = True)
                        #print("VENDOR_MASTERDATA shape after adding new Name :",CURRENT_ENTITY_MASTER.shape)
                        new_record_count = new_record_count + 1
        print("No of Records added : ",new_record_count)
        print("Finding and adding new records Done!")
        return CURRENT_ENTITY_MASTER
    except:
        print("Finding and adding new records exception :",traceback.print_exc())
        return CURRENT_ENTITY_MASTER_COPY

### Update all master data
   
def Update_Master_Data():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file_path = os.path.join(rootFolderPath, "update_master_data"+ strftime("%d_%m_%Y") + ".log")
    # log_file_path = os.path.join("update_master_data"+ strftime("%d_%m_%Y") + ".log")
    log_file = open(log_file_path,"w")
    sys.stdout = log_file
    sys.stderr = log_file
    print("Start Time :",strftime("%d_%m_%Y"))
    vendorMasterDataPath = cfg.getVendorMasterData()
    buyerMasterDataPath = cfg.getBuyerMasterData()
    try:
        args = sys.argv
        print("args :",len(args))
        if len(args)==1:
            Vendor_Master = r"MH_Vendor_Master.xlsx"  
            Entitiy_Master = r"MH_Entity_Master.xlsx"
            dwonload_folder = r"./Utilities"
            print("Dwonloading new vendor master data")
            Dwonload_Vendor_MasterData =  download_MasterData_from_blob(Vendor_Master,dwonload_folder)
            print("Dwonloading new entity master data")
            Dwonload_Entity_MasterData =  download_MasterData_from_blob(Entitiy_Master,dwonload_folder)


            CURRENT_VENDOR_MASTER = pd.read_csv(vendorMasterDataPath) #r"./Utilities/VENDOR_ADDRESS_MASTERDATA.csv")
            NEW_VENDOR_MASTER = pd.read_excel(os.path.join(dwonload_folder,Vendor_Master))
            # Entity master data table rows starts from 3rd row and column starts from second 
            NEW_ENTITY_MASTER = pd.read_excel(r"./Utilities/MH_Entity_Master.xlsx",skiprows=3,usecols = "B:H")
            # NEW_VENDOR_MASTER = NEW_VENDOR_MASTER[["Contact Name","GST Identification Number (GSTIN)"]]
            print("NEW_VENDOR_MASTER",NEW_VENDOR_MASTER.columns)

            #NEW_VENDOR_MASTER = NEW_VENDOR_MASTER.rename(columns={"Contact Name":"VENDOR_NAME","GST Identification Number (GSTIN)":"VENDOR_GSTIN"})
            #NEW_ENTITY_MASTER = pd.read_excel(r"./Utilities/MH-Entity_Master.xlsx")
            print("NEW_ENTITY_MASTER cols",NEW_ENTITY_MASTER.columns)

            CURRENT_ENTITY_MASTER = pd.read_csv(buyerMasterDataPath) #r"./Utilities/BUYER_ADDRESS_MASTERDATA.csv")

            # Removing Replacing NaN values to ""
            CURRENT_ENTITY_MASTER = CURRENT_ENTITY_MASTER.replace(np.nan, "", regex=True)
            CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.replace(np.nan, "", regex=True)
            NEW_ENTITY_MASTER = NEW_ENTITY_MASTER.replace(np.nan, "", regex=True)
            NEW_VENDOR_MASTER = NEW_VENDOR_MASTER.replace(np.nan, "", regex=True)

            if Dwonload_Vendor_MasterData:
                
                # Reading Client shred vendor data and deleting duplicates
                try:
                    print("Updating existing vendor master records")
                    CURRENT_VENDOR_MASTER = update_existing_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_ENTITY_MASTER)
                    # print("Adding new records into vendor master")            
                    # CURRENT_VENDOR_MASTER = adding_new_record_into_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_ENTITY_MASTER)
                    identifier = []
                    score = []
                    for row in CURRENT_VENDOR_MASTER.itertuples():
                        identifier_text = str(row.VENDOR_NAME) + " " + str(row.VENDOR_GSTIN)
                        while(len(identifier_text.split(" ")) < 10):
                            identifier_text = str(identifier_text) + " " + str(identifier_text)
                        #identifier_text = " ".join(identifier_text)
                        identifier.append(identifier_text)
                        score.append(1)
                    CURRENT_VENDOR_MASTER["IDENTIFIER_TEXT"]  = identifier
                    CURRENT_VENDOR_MASTER["DOCUMENT_TEXT"]  = identifier
                    CURRENT_VENDOR_MASTER["MATCH_SCORE"]  = score
                    CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    CURRENT_VENDOR_MASTER.to_csv(r"./Utilities/SWIGGY_VENDOR_ADDRESS_MASTERDATA.csv",index = False)
                    print("Vendor master updates Done!")
                except:
                    print("Update vendor master exception :",traceback.print_exc())
                    pass

            if Dwonload_Entity_MasterData:
                
                # Reading Client shred vendor data and deleting duplicates
                try:
                    print("Updating existing entity master records")
                    CURRENT_ENTITY_MASTER = update_existing_enity_master(CURRENT_ENTITY_MASTER,NEW_ENTITY_MASTER)
                    print("Adding new records into entity master")            
                    CURRENT_ENTITY_MASTER = adding_new_record_into_entity_master(CURRENT_ENTITY_MASTER,NEW_ENTITY_MASTER,CURRENT_VENDOR_MASTER)
                    CURRENT_ENTITY_MASTER = CURRENT_ENTITY_MASTER.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    CURRENT_ENTITY_MASTER.to_csv(r"./Utilities/SWIGGY_BUYER_ADDRESS_MASTERDATA.csv")
                    print("Entity master updates Done!")
                except:
                    print("Update entity master exception :",traceback.print_exc())
                    pass
            print("Excecution Done!")
            if log_file is not None:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_file.close()

            return

        elif len(args)==2:
            print("Running arguments :",args[1])
            external_path = args[1]
            CURRENT_VENDOR_MASTER = pd.read_csv(vendorMasterDataPath) #r"./Utilities/VENDOR_ADDRESS_MASTERDATA.csv")
            NEW_VENDOR_MASTER = pd.read_excel(external_path)
            CURRENT_ENTITY_MASTER = pd.read_csv(buyerMasterDataPath) #r"./Utilities/BUYER_ADDRESS_MASTERDATA.csv")

            # Removing Replacing NaN values to ""
            CURRENT_ENTITY_MASTER = CURRENT_ENTITY_MASTER.replace(np.nan, "", regex=True)
            CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.replace(np.nan, "", regex=True)
            NEW_VENDOR_MASTER = NEW_VENDOR_MASTER.replace(np.nan, "", regex=True)

            # Reading Client shred vendor data and deleting duplicates
            try:
                print("Updating existing vendor master records")
                CURRENT_VENDOR_MASTER = update_existing_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_ENTITY_MASTER)
                # print("Adding new records into vendor master")            
                # CURRENT_VENDOR_MASTER = adding_new_record_into_vendor_master(CURRENT_VENDOR_MASTER,NEW_VENDOR_MASTER,CURRENT_ENTITY_MASTER)
                identifier = []
                score = []
                for row in CURRENT_VENDOR_MASTER.itertuples():
                    identifier_text = str(row.VENDOR_NAME) + " " + str(row.VENDOR_GSTIN)
                    while(len(identifier_text.split(" ")) < 10):
                        identifier_text = str(identifier_text) + " " + str(identifier_text)
                    #identifier_text = " ".join(identifier_text)
                    identifier.append(identifier_text)
                    score.append(1)
                CURRENT_VENDOR_MASTER["IDENTIFIER_TEXT"]  = identifier
                CURRENT_VENDOR_MASTER["DOCUMENT_TEXT"]  = identifier
                CURRENT_VENDOR_MASTER["MATCH_SCORE"]  = score
                CURRENT_VENDOR_MASTER = CURRENT_VENDOR_MASTER.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                CURRENT_VENDOR_MASTER.to_csv(r"./Utilities/SWIGGY_VENDOR_ADDRESS_MASTERDATA.csv",index = False)
                print("Vendor master updates Done!")
                return
            except:
                print("Update vendor master exception :",traceback.print_exc())
                return
        else: 
            print("invalid arguments")
            return

    except :
        print("Update Master Data exception :",traceback.print_exc())
        if log_file is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()
        return
if __name__ == "__main__":
    Update_Master_Data()
