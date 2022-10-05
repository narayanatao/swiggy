# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 23:03:49 2021

@author: Hari
"""

from kafka import KafkaConsumer
from kafka import TopicPartition
from json import loads
# from time import sleep
from sys import argv
import TAPPconfig as cfg
import tapp_client as cli
import traceback

partition = int(argv[1])
cnt = 0
serverName = cfg.getKafkaServer()
consGroup = cfg.getKafkaConsumerGroup()
topic = cfg.getKafkaTopic()

while True:
    consumer = KafkaConsumer(
         bootstrap_servers = [serverName],
         auto_offset_reset = 'earliest',
         enable_auto_commit = True,
         group_id = consGroup,
         value_deserializer=lambda x: loads(x.decode('utf-8')))
    print("Partition assigned", partition,topic)
    consumer.assign([TopicPartition(topic, partition)])

    for message in consumer:
        message_ = message.value
        sub_id = message_["sub_id"]
        auth_token = message_["auth_token"]
        documentId = message_["documentId"]
        s_delta = message_["s_delta"]
        callbackUrl = message_["callbackUrl"]
        try:
            extraction = cli.getExtractionResults(auth_token,
                                                  s_delta,
                                                  documentId,
                                                  callbackUrl,
                                                  sub_id)
        except:
            print("Kafka consumer",
                  traceback.print_exc())
            pass
    consumer.close()


