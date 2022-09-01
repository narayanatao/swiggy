# -*- coding: utf-8 -*-

import json
import os

#Load configuration
with open("config.json") as config_file:
    configdata = json.load(config_file)

config_file.close()

#CHANGE STARTS HERE
# If .env file is present, the values in .env file should override the config.json values.
# This must be done only for the overlapping keys
envdata = {}
if os.path.exists(".env"):
    with open(".env")  as env_file:
        envdata = json.load(env_file)

config_keys = configdata.keys()
for k,v in envdata.items():
    if k in config_keys:
        configdata[k] = v
# If .env file is present, the values in .env file should override the config.json values.
# This must be done only for the overlapping keys
#CHANGE ENDS HERE

def getConfigData():
    return configdata

def getBlobStoreProvider():
    blobStoreProvider = configdata["BLOB_STORAGE_PROVIDER"]
    return blobStoreProvider

def blobStoreDetails():
    accountName = configdata["BLOB_ACCOUNT_NAME"]
    accountKey = configdata["BLOB_ACCOUNT_KEY1"]
    return accountName, accountKey

def getFolder(name):
    folder = configdata[name]
    os.makedirs(folder, exist_ok = True)
    return folder

def getDocMapFolder():
    return getFolder("DOCUMENT_TEMPLATE_MAPPING_FOLDER")

def getTiffFolder():
    return getFolder("TIFF_CONVERTED_FOLDER")

def getPngFolder():
    return getFolder("PNG_CONVERTED_FOLDER")

def getPreProcAsyncFolder():
    return getFolder("ASYNC_PROC_START_FOLDER")

def getCompareTemplateFolder():
    return getFolder("COMPARE_TEMPLATES_FOLDER")

def getSplitFolder():
    return getFolder("SPLIT_FOLDER")

def getDownloadFolder():
    return getFolder("DOWNLOAD_FOLDER")

def getPreProcAsyncInprogFolder():
    return getFolder("ASYNC_PROC_INPROG_FOLDER")

def getAbbyyResultXmlToJsonMap():
    return getFolder("ABBYY_RESULTXML_JSON_MAP")

def getExtractionTempFolder():
    return getFolder("EXTRACTION_TEMP")

def getDeskewFileSuffix():
    return configdata["DESKEW_FILE_SFX"]

def getCroppedFileSuffix():
    return configdata["CROPPED_FILE_SFX"]

def getMultiPageFileSuffix():
    return configdata["MULTIPAGE_FILE_SFX"]

def getEnhancedFileSuffix():
    return configdata["ENHANCED_FILE_SFX"]

def getPreprocessedFileSuffix():
    return configdata["PREPROCESSED_FIL_SFX"]

def getGhostPath():
    return configdata["GHOST_SCRIPT_EXE"]

def getGhostTiffDvc():
    return configdata["GHOST_TIFF_DEVICE"]

def getGhostPause():
    return configdata["GHOST_TIFF_PAUSE"]

def getGhostTiffDownScale():
    return configdata["GHOST_DOWN_SCALE"]

def getGhostTiffDownScaleFactor():
    return configdata["GHOST_DOWN_SCALE_FACTOR"]

def getGhostQuit():
    return configdata["GHOST_QUIT"]

def getGhostCommandNoPause():
    return configdata["GHOST_COMMAND_NOPAUSE"]

def getGhostCommandDsf():
    return configdata["GHOST_COMMAND_DSF"]

def getGhostCommandQuit():
    return configdata["GHOST_COMMAND_QUIT"]

def getGhostCommandOut():
    return configdata["GHOST_COMMAND_OUT"]

def getGhostCommandForce():
    return configdata["GHOST_COMMAND_FORCE"]

def getUIServer():
    return configdata["UI_SERVER"]

def getDocUpdURL():
    return configdata["DOCUMENT_UPDATE_URL"]

def getDocResUpdURL():
    return configdata["DOCUMENT_RESULT_UPDATE_URL"]

def getDocResCrtURL():
    return configdata["DOCUMENT_RESULT_CREATE_URL"]

def getDocGetURL():
    return configdata["DOCUMENT_GET_URL"]

def getDocResGetURL():
    return configdata["DOCUMENT_RESULT_GET_URL"]

def getVendorGetURL():
    return configdata["VENDOR_GET_URL"]

def getAbbyyHotFolder():
    return configdata["ABBYY_HOT_FOLDER"]

def getAbbyyExportFolder():
    return configdata["ABBYY_EXPORT_FOLDER"]

def getAbbyyUnknownFolder():
    return configdata["ABBYY_UNKNOWN_FOLDER"]

def getCropBorder():
    return configdata["CROP_BORDER"]

def getImgSimFlannMtchDist():
    return configdata["IMAGE_SIM_FLANN_MATCH_DIST"]

def getImgSimFlannMtchThresh():
    return configdata["IMAGE_SIM_FLANN_MATCH_THRESH"]

def getExtractionEngine():
    return configdata["EXTRACTION_ENGINE"]

def getDocumentType():
    return configdata["DOCUMENT_TYPE"]

def getStorageType():
    return configdata["STORAGE_TYPE"]

def getBlobStoreImport():
    return configdata["BLOB_FOLDER_IMPORT"]

def getBlobStorePreproc():
    return configdata["BLOB_FOLDER_PREPROC"]

def getBlobStoreExport():
    return configdata["BLOB_FOLDER_EXPORT"]

def getBlobStoreCompareTemplate():
    return configdata["BLOB_FOLDER_COMPARE_TEMPLATE"]

def getBlobStoreTempResMap():
    return configdata["BLOB_FOLDER_TEMPLATE_RESULT_MAP"]

def getFolderImport():
    return configdata["FOLDER_IMPORT"]

def getFolderPreproc():
    return configdata["FOLDER_PREPROC"]

def getFolderExport():
    return configdata["FOLDER_EXPORT"]

def getFolderCompareTemplate():
    return configdata["FOLDER_COMPARE_TEMPLATE"]

def getFolderTempResMap():
    return configdata["FOLDER_RESULT_MAP"]

def getFolderSampleInvoice():
    return configdata["FOLDER_SAMPLE_INVOICE"]

def getFolderExtraction():
    return configdata["FOLDER_ABBYY_RESULT"]

def getRootFolderPath():
    return configdata["ROOT_FOLDER"]

def getSystemUser():
    return configdata["SYSTEM_USER"]

def getTappVersion():
    return configdata["TAPP_VERSION"]

def getDocUpdApi():
    return configdata["DOCUMENT_UPDATE_API_ID"]

def getDocResAddApi():
    return configdata["DOCUMENT_RESULT_ADD_API_ID"]

def getStatusPreprocInProg():
    return configdata["STATUS_PREPROC_INPROGRESS"]

def getStatusReadyExtract():
    return configdata["STATUS_READY_FOR_EXTRACTION"]

def getStatusExtractInProg():
    return configdata["STATUS_EXTRACTION_INPROGRESS"]

def getStatusExtractDone():
    return configdata["STATUS_EXTRACTION_DONE"]

def getStatusReview():
    return configdata["STATUS_REVIEW"]

def getStatusReviewCompleted():
    return configdata["STATUS_REVIEW_COMPLETED"]

def getStatusProcessed():
    return configdata["STATUS_PROCESSED"]

def getStatusFailed():
    return configdata["STATUS_FAILED"]

def getParamStatusSuccess():
    return configdata["PARAM_STATUS_SUCCESS"]

def getParamStatusFailed():
    return configdata["PARAM_STATUS_FAILED"]

def getErrcodeError():
    return configdata["ERRORCODE_ERROR"]

def getErrcodeExtResUpdFail():
    return configdata["ERRORCODE_EXT_RESULT_UPDATE_FAILED"]

def getErrcodeExtTpl404():
    return configdata["ERRORCODE_EXT_TEMPLATE_NOT_FOUND"]

def getErrcodePreprocDocNotAcc():
    return configdata["ERRORCODE_PREPROC_DOC_NOT_ACCEPTED"]

def getErrcodePreprocFail():
    return configdata["ERRORCODE_PREPROC_FAILED"]

def getErrcodePreprocDocTpl404():
    return configdata["ERRORCODE_PREPROC_DOC_TPL_404"]

def getErrcodePreprocImgEnhFail():
    return configdata["ERRORCODE_PREPROC_IMG_ENH_FAILED"]

def getErrmsgExtractInitFail():
    return configdata["ERRORMSG_EXT_INITIATION_FAILED"]

def getErrmsgExtractResNotUpd():
    return configdata["ERRORMSG_EXT_RESULTS_NOT_UPDATED"]

def getErrmsgExtractTpl404():
    return configdata["ERRORMSG_EXT_NO_TEMPLATE_FOUND"]

def getErrmsgPreprocFileNotDownload():
    return configdata["ERRORMSG_PREPROC_FILE_NOT_DOWNLOADED"]

def getErrmsgPreprocNotValidInv():
    return configdata["ERRORMSG_PREPROC_NOT_VALID_INVOICE"]

def getErrmsgPreprocTiffConv():
    return configdata["ERRORMSG_PREPROC_TIFF_FILE_CONVERT"]

def getErrmsgPreprocFail():
    return configdata["ERRORMSG_PREPROC_FAILED"]

def getErrmsgPreprocNomatchInv():
    return configdata["ERRORMSG_PREPROC_NOMATCH_INVOICE"]

def getErrmsgPreprocImgEnhFail():
    return configdata["ERRORMSG_PREPROC_FAILED_IMG_ENHANCEMENT"]

def getStatusmsgPreprocCompleted():
    return configdata["STATUSMSG_PREPROC_COMPLETED"]

def getStatusmsgPreprocAccepted():
    return configdata["STATUSMSG_PREPROC_ACCEPTED"]

def getStatusmsgExtractSuccess():
    return configdata["STATUSMSG_EXTRACTION_SUCCESS"]

def getStatusmsgExtractInitated():
    return configdata["STATUSMSG_EXTRACTION_INITIATED"]

def getTappMlProbThreshold():
    return configdata["TAPP_ML_PROB_THRESHOLD"]

def getMultiPageTiffCompress():
    return configdata["MULTIPAGE_TIFF_COMPRESSION"]

def getMimeTiff():
    return configdata["MIMETYPE_TIFF"]

def getMimeTif():
    return configdata["MIMETYPE_TIF"]

def getMimePng():
    return configdata["MIMETYPE_PNG"]

def getMimePdf():
    return configdata["MIMETYPE_PDF"]

def getMimeJson():
    return configdata["MIMETYPE_JSON"]

def getMimeXml():
    return configdata["MIMETYPE_XML"]

def getExtnTiff():
    return configdata["EXTN_TIFF"]

def getExtnTif():
    return configdata["EXTN_TIF"]

def getExtnPdf():
    return configdata["EXTN_PDF"]

def getExtnTxt():
    return configdata["EXTN_TXT"]

def getExtnJson():
    return configdata["EXTN_JSON"]

def getTemp():
    return configdata["TEMP_FOLDER"]

def getNoPageToCompare():
    return configdata["NO_PAGES_TO_COMPARE"]

def getStagePreproc():
    return configdata["STAGE_PREPROC"]

def getStageExtract():
    return configdata["STAGE_EXTRACT"]

def getConstCallbackUrl():
    return configdata["CONST_CALLBACKURL"]

def getXmlRetryCount():
    return configdata["XML_RETRY_COUNT"]

def getExtractionPort():
    return configdata["EXTRACTION_PORT"]

def getExtractionApiIP():
    return configdata["EXTRACTION_API_IP"]

def getExtractionApiPort():
    return configdata["EXTRACTION_API_PORT"]

def getExtractionApiAddr():
    return configdata["EXTRACTION_API_ADDR"]

def getPreprocPort():
    return configdata["PREPROC_PORT"]

def getPreprocSync():
    return configdata["PREPROC_SYNC"]

def getPreprocParallel():
    return configdata["PREPROC_PARALLEL"]

def getVendorListAPI():
    return configdata["GET_VENDOR_API"]

def getCropRequired():
    return configdata["CROP_REQUIRED"]

def getDeskewingRequired():
    return configdata["DESKEWING_REQUIRED"]

def getModelPath():
    return configdata["MODEL_PATH"]

def getFacebookModelPath():
    return configdata["FACEBOOK_MODEL_PATH"]

def getVectorDimensionalLimit():
    return configdata["VECTOR_DIMENSIONAL_LIMIT"]

def getXMLMappingFile():
    return configdata["XML_MAPPING_GLOBAL"]

def getFormRecognizerEndpoint():
    return configdata["FORM_RECOGNIZER_ENDPOINT"]

def getFormRecognizerAPIKey():
    return configdata["FORM_RECOGNIZER_API_KEY"]

def getFormRecognizerPostURL():
    return configdata["FORM_RECOGNIZER_POST_URL"]

def getOCROutFolder():
    return getFolder("FOLDER_OCR_PROCESSED")

def getFeatureOutFolder():
    return getFolder("FOLDER_FEATURE_PROCESSED")

def getLablePickleFile():
    return configdata["LABEL_PICKLE_FILE"]

def getBlobConnectionString():
    return configdata["BLOB_CONNECTION_STRING"]

def getBlobContainerName():
    return configdata["BLOB_CONTAINER_NAME"]

def getBlobOCRFolder():
    return configdata["BLOB_FOLDER_OCR"]

def getBlobFeatureFolder():
    return configdata["BLOB_FOLDER_FEATURE"]

def getFeatureTypes():
    return configdata["FEATURE_TYPES_CSV"]

def getLabelMapping():
    return configdata["LABEL_MAPPING_CSV"]

def getClassWeights():
    return configdata["CLASS_WEIGHTS_CSV"]

def getModelJoblib():
    return configdata["MODEL_JOBLIB"]

def getNoOfModelCols():
    return configdata["NO_MODEL_COLUMNS"]

def getLabelKeywords():
    return configdata["LABEL_KEYWORDS"]

def getFeatureTxtFile():
    return configdata["MODEL_FEATURES_TEXT"]

def getPostProcessConstantLabels():
    return configdata["POST_PROCESS_CONSTANT_LABELS"]

def getPostProcessFieldLabels():
    return configdata["POST_PROCESS_FIELD_LABELS"]

def getPostProcessScoring():
    return configdata["POST_PROCESS_SCORING"]

def getSTPConfiguration():
    return configdata["STP_CONFIGURATION"]

def getPostProcessDtTypeCheck():
    return configdata["POST_PROCESS_DATATYPE_CHECK"]

def getStopWordFilePath():
    return configdata["STOP_WORD_FILE"]

def getlabelKeywordsNonTokenized():
    return configdata['LABEL_KEYWORDS_NEW']

def getPostProcessSTPScore():
    return configdata['POST_PROCESS_STP_SCORE']

def getRequiredVendorFields():
    return configdata["VENDOR_FIELDS_LIST"]

def getVendorMatchThresh():
    return configdata["VENDOR_IDENTIFICATION_THRESH"]

def getErrmsgMultiInv():
    return configdata["ERRORMSG_MULTIINVOICE_FAILED"]

def getMasterDataFilePath():
    return configdata["VENDOR_MASTERDATA_CSV"]

def getValidationRulesPath():
    return configdata["VENDOR_RULES_JSON"]

def getCleanedKeywordsPath():
    return configdata["CLEANED_KEYWORDS"]

def getKeywordsAliasPath():
    return configdata["LABEL_KEYWORDS_ALIAS"]

def getDocGetResultRPA_URL():
    return configdata["DOCUMENT_GET_RESULT_RPA_URL"]

def getErrcodeResultDownloadFail():
    return configdata["ERRORCODE_RESULT_DOWNLOAD_FAILED"]

def getErrmsgResultDownloadFail():
    return configdata["ERRORMSG_RESULT_DOWNLOAD_FAILED"]

def getStatusmsgResultDownloadCompleted():
    return configdata["STATUSMSG_RESULT_DOWNLOAD_COMPLETED"]

def getStatusmsgResultDownloadFailed():
    return configdata["STATUSMSG_RESULT_DOWNLOAD_FAILED"]

def getBlobClientFolder():
    return configdata["BLOB_CLIENT_FOLDER"]

def getXLSDownloadColumnsPath():
    return configdata["XLS_DOWNLOAD_COLUMNS_PATH"]

def getExtractionSubmitAPI():
    return configdata["EXTRACTION_SUBMIT_API"]

def getExtractionGetAPI():
    return configdata["EXTRACTION_GET_API"]

def getBlobStorageAPI():
    return configdata["BLOB_STORAGE_API"]

def getSubscriberId():
    return configdata["SUBSCRIBER_ID"]

def getPrivateEncryptKey():
    return configdata["PRIVATE_ENCRYPT_KEY"]

def getTaskBroker():
    return configdata["TASK_BROKER"]

def getTaskName():
    return configdata["TASK_NAME"]

def getUIRootFolder():
    return configdata["UI_ROOT_FOLDER"]

def getDownloadResultsApi():
    return configdata["DOWNLOAD_RESULTS_API"]

def getPreprocServer():
    return configdata["PREPROC_SERVER"]

def getKafkaTopic():
    return configdata["KAFKA_EXTRACTION_TOPIC"]

def getKafkaTopicPartitions():
    return configdata["KAFKA_TOPIC_PARTITIONS"]

def getKafkaServer():
    return configdata["KAFKA_SERVER"]

def getKafkaConsumerGroup():
    return configdata["KAFKA_CONSUMER_GROUP"]

def getAppName():
    return configdata["APP_NAME"]

def getDocoumentResult():
    return configdata["DOCUMENT_GET_RESULT_API"]

def getDocumentFind():
    return configdata["DOCUMENT_FIND_API"]
    
def getEventGridAPI():
    return configdata["EVENT_GRID_POLLING_API"]

def getEventGridKey():
    return configdata["EVENT_GRID_POLLING_KEY"]

def getEventGridName():
    return configdata["EVENT_GRID_POLLING_NAME"]

def getEventGridSubPort():
    return configdata["EVENT_GRID_SUBSCRIBER_PORT"]

def getQueueType():
    return configdata["QUEUE_TYPE"]

def getPostExtractionSvcPort():
    return configdata["POST_EXTRACTION_SVC_PORT"]

def getPostExtractionSvcIP():
    return configdata["POST_EXTRACTION_SVC_IP"]

def getUIDownloadFolder():
    return configdata["DOWNLOAD_RESULT_FOLDER"]

def getVendorMasterData():
    return configdata["VENDOR_MASTERDATA_CSV"]

def getBuyerMasterData():
    return configdata["BUYER_MASTERDATA_CSV"]


