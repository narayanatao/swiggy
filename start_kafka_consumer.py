# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:24:44 2021

@author: Hari
"""

import subprocess
import sys
import TAPPconfig as cfg

partitions = cfg.getKafkaTopicPartitions()

for i in range(partitions):
    subprocess.Popen(['python',
                      'kafka_consumer.py',
                      str(i)])

sys.exit(0)

