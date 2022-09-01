import os
import paramiko

hostname = '52.172.153.247'
username = 'TAPPUI'
password = 'TAPP_UI_123456'

remotefilepath = '/home/TAPPUI/tapp_3.0/files/VELANKANI_00111_0-1_pred.csv'
localfilepath = 'VELANKANI_00111_0-1_pred.csv'

ssh = paramiko.SSHClient() 
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname,username=username,password=password)

sftp = ssh.open_sftp()
sftp.get(remotefilepath, localfilepath)
sftp.close()
ssh.close()
