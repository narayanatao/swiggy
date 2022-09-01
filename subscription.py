import json
import datetime
from azure.storage.blob import BlobClient
import os

class Subscription:
    """
    Subscription Class
    Individual Subscription Fields:
    subscription_id: Subscription ID,
    subscription_tier: Subscription Tier,
    expiry_date: Subscription Expiry Date in DD-MM-YYYY Format,
    subscribed_pages: Subscribed Pages,
    page_processed: Pages Processed till Date,
    page_remaining: Pages Remaining
    """

    def __init__(self, subscription_id, subscription_data_path):
        """

        :param subscription_id: Subscription ID, JSON Field: subscription_id
        :param subscription_data_path: JSON Path to read Subscription Lists
        __list_sub: List of all Subscriptions
        __subscription_status: Subscription Status of subscription_id
        """
        self.subscription_id = subscription_id
        self.subscription_data_path = subscription_data_path
        self.__list_sub = []
        self.__subscription_status = {}
        self.__get_subscription_list_blob()


    def __get_subscription_list_blob(self):
        """

        :return:
        """
        connect_str = "DefaultEndpointsProtocol=https;AccountName=tapp2data;AccountKey=C38zpM1CfufDmqcelnI/VvjIUpB6Fyoj8QUtsKrFs4f7pAKCpzMFRClSOhJW1thKSOdZB7Jm3OWughlSKEsuxg==;EndpointSuffix=core.windows.net"
        container = os.path.split(self.subscription_data_path)[0]
        blob_name = os.path.split(self.subscription_data_path)[1]
        blob = BlobClient.from_connection_string(conn_str=connect_str, container_name=container, blob_name=blob_name)

        blob_data = blob.download_blob()
        data_bytes = blob_data.readall()

        self.__list_sub = json.loads(data_bytes)


    # def __get_subscription_list(self):
    #     """
    #     Get all subscriptions
    #     :return:
    #     """
    #     try:
    #         with open(self.subscription_data_path, 'rb') as read_file:
    #             self.__list_sub = json.load(read_file)
    #     except Exception as e:
    #         print(e)
    #         pass

    def __dump_subscription_list_blob(self):
        """

        :return:
        """
        connect_str = "DefaultEndpointsProtocol=https;AccountName=tapp2data;AccountKey=C38zpM1CfufDmqcelnI/VvjIUpB6Fyoj8QUtsKrFs4f7pAKCpzMFRClSOhJW1thKSOdZB7Jm3OWughlSKEsuxg==;EndpointSuffix=core.windows.net"
        try:
            container = os.path.split(self.subscription_data_path)[0]
            blob_name = os.path.split(self.subscription_data_path)[1]
            blob = BlobClient.from_connection_string(conn_str=connect_str, container_name=container, blob_name=blob_name)

            blob.upload_blob(json.dumps(self.__list_sub, indent=4), overwrite=True)
            return True
        except Exception as e:
            print(e)
            return False


    # def __dump_subscription_list(self):
    #     """
    #     Internal method called by update_subscription_status
    #     update_subscription_status updates __list_sub to be written back
    #     DO NOT REFRESH __list_sub before dump
    #     :return:
    #     """
    #     try:
    #         with open(self.subscription_data_path, 'w') as out_file:
    #             json.dump(self.__list_sub, out_file, indent=4)
    #         return True
    #     except Exception as e:
    #         print(e)
    #         return False


    def subscription_status(self):
        """
        Returns the current status of the subscription
        Refresh __list_sub before getting the __subscription_status
        :return:
        """
        self.__get_subscription_list_blob()
        subscription_keys = [subs["subscription_id"] for subs in self.__list_sub]

        if self.subscription_id in subscription_keys:
            self.__subscription_status = [subs for subs in self.__list_sub
                                          if subs["subscription_id"] == self.subscription_id][0]

        return self.__subscription_status


    def authenticate_subscription(self):
        """
        Authenticates the subscription for expiry_date and page_remaining
        expiry_date should be less than or equal to current date
        Refresh subscription_status before authenticating
        :return:
        """
        self.subscription_status()
        today = datetime.date.today()

        if len(self.__subscription_status) > 0:
            self.__subscription_status['expiry_date'] = datetime.datetime.strptime(self.__subscription_status['expiry_date'],
                                                                            "%d-%m-%Y").date()
            if self.__subscription_status['expiry_date'] < today:
                return False, "subscription_expired"
            if self.__subscription_status['page_remaining'] <= 0:
                return False, "page_limit_exceeded"
        else:
            return False, "invalid_subscription_id"

        return True, "Ok"


    def update_subscription_status(self, current_status):
        """
        Takes current_status of subscription with updated page_processed
        and page_remaining and updates the value in the subscription storage
        Refresh __list_sub before getting the update_subscription_status
        :return:
        """
        print("update_subscription_status:", current_status)
        self.__get_subscription_list_blob()
        subscriptions = [subs["subscription_id"] for subs in self.__list_sub]
        try:
            if current_status["subscription_id"] in subscriptions:
                print("Updating Subscription Status")
                for sub in self.__list_sub:
                    if sub["subscription_id"] == current_status["subscription_id"]:
                        sub["page_processed"] = current_status["page_processed"]
                        sub["page_remaining"] = current_status["page_remaining"]

                return self.__dump_subscription_list_blob()
            else:
                return False
        except Exception as e:
            print(e)
            return False