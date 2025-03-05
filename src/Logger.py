import os
import torch
import datetime

class Logger():
    def __init__(self):
        self.pid_str =  str(os.getpid())
        logfile = 'logfile_' +  self.pid_str + '.txt'
        self.logfile = os.path.join(os.getcwd(), 'out', logfile)

        if torch.cuda.is_available():
            self.log('Cuda is available')
        else:
            self.log('Cuda is unavailable')

    def log(self, output):
        with open(self.logfile, 'a') as file:
            file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '  ' + str(output) + '\n')
            print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '  ' + str(output)))