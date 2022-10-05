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

partition = int(argv[1])
cnt = 0
serverName = cfg.getKafkaServer()
consGroup = cfg.getKafkaConsumerGroup()
topic = cfg.getKafkaTopic()

#Commit all the open messages when starting the kafka_consumer
print("Starting called again")
consumer = KafkaConsumer(
     bootstrap_servers=[serverName],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id=consGroup,
     value_deserializer=lambda x: loads(x.decode('utf-8')))
print("Partition assigned", partition,topic)
consumer.assign([TopicPartition(topic, partition)])

for message in consumer:
    message_ = message.value
    print(message)
consumer.close()
print("exiting readall")
