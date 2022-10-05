# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:12:32 2021

@author: Hari
"""

# from time import sleep
from json import dumps
from kafka import KafkaProducer
from kafka import KafkaAdminClient
from kafka.admin import NewPartitions, NewTopic
import TAPPconfig as cfg
import traceback

topicName = cfg.getKafkaTopic()
noPartitions = cfg.getKafkaTopicPartitions()
kafkaServer = cfg.getKafkaServer()

try:
    admin_client = KafkaAdminClient(bootstrap_servers=[kafkaServer])
    topic_list = []
    print("Create new topic: ", topicName)
    topic_list.append(NewTopic(name=topicName,
                               num_partitions = 1,
                               replication_factor=1))
    admin_client.create_topics(new_topics=topic_list,
                               validate_only=False)
    print("Create new topic created: ", topicName)
except:
    print("New topic not created:", traceback.print_exc())
    pass

try:
    print("Create new partition: ", noPartitions)
    topic_partitions = {}
    topic_partitions[topicName] = NewPartitions(total_count=noPartitions)
    admin_client.create_partitions(topic_partitions)
except:
    print(traceback.print_exc())
    pass

# producer = KafkaProducer(bootstrap_servers=[kafkaServer],
#                          value_serializer=lambda x: 
#                          dumps(x).encode('utf-8'))

# partitions = [0,1,2,3]
partitionNumber = 0

def sendMessage(data):

    try:
        global partitionNumber
        producer = KafkaProducer(bootstrap_servers=[kafkaServer],
                                 value_serializer=lambda x: 
                                 dumps(x).encode('utf-8'))
        p = partitionNumber % noPartitions
        print("Producer send: ", data, p,topicName)
        producer.send(topicName,
                      value=data,
                      partition = p)
        partitionNumber += 1
        return True
    except:
        print(traceback.print_exc())
        return False


# for e in range(20):
#     data = {"number":e}
#     p = e % len(partitions)
#     producer.send('numtest',
#                   value=data,
#                   partition = p
#                   )
    # producer.send('numtest',
    #               value=data
    #               )
    # sleep(1)
