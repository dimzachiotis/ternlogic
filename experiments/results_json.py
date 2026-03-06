#This file is typically used to:
#Run an experiment / script / training job
#Log results over time (loss, accuracy, metrics, etc.)
#Save everything to a JSON file
#Load the results later for analysis, plotting, comparison, debugging, resuming experiments. 

import json
#Used to convert Python objects to JSON strings and back.
import time
#Used to measure time, such as how long an experiment took.
import socket
#Used to get the hostname of the machine running the code.
import os
#Used for file paths (so it works on any OS).

#This class is designed to collect experiment results, track timing and save/load everything as a JSON file
class ResultsJSON(object):

    def __init__(self, eid: int, path: str):
        self.eid = eid
        #experiment id
        self.path = path
        #folder where the JSON file will be saved

        self.init_time = time.time()
        self.save_time = None
        self.total_time = None
        #timestamps

        self.args = None
        #Placeholder to store experiment arguments later

        self.server_name = socket.gethostname().split('.')[0]
        #Stores machine name
    
    #Method to store experiment arguments
    def store_args(self, args):

        self.args = vars(args)
        #converts an object into a dictionary

    #Used to store results over time
    def store_results(self, results: dict):

        #Loop through result values
        for key, val in results.items():
            if not hasattr(self, key):
                setattr(self, key, list())
            #If self.loss does not exist → create it as an empty list

            getattr(self, key).append(val)
            #Append value to the list

    def store_final_results(self, results: dict):

        for key, val in results.items():
            key = key + '_'
        #Adds underscore to distinguish final results
            setattr(self, key, val)

    #Saves all stored data to a file
    def save(self):
        self.save_time = time.time()
        self.total_time = self.save_time - self.init_time
        #Computes total runtime

        json_str = json.dumps(self.__dict__)
        #Converts everything into a JSON string


        with open(os.path.join(self.path, f'{self.eid}_total_tern_2.json'), mode='w') as f:
            f.write(json_str)
        #Writes JSON string to file
    
    #Loads experiment results from a file
    @staticmethod
    def load(eid: int, path: str, get_dict=False):
        with open(os.path.join(path, f'{eid}_total_tern_2.json'), mode='r') as f:
            data = json.loads(f.read())
        #Converts JSON back into a Python dictionary

        if get_dict:
            return data
        #if true, return raw dictionary

        self = ResultsJSON(-1, '')
        #Creates a dummy object
        self.__dict__.update(data)
        #Copies all saved values into the object

        assert eid == self.eid
        #Ensures loaded file matches requested experiment ID
        return self

#Main script - Example usage
if __name__ == '__main__':

    r = ResultsJSON(101, './')
    #Creates experiment with ID 101

    print(r.__dict__)
    #Shows all attributes before saving

    r.save()
    #Saves results to 00000101.json

    r2 = ResultsJSON.load(101, './')
    #Loads file back into an object

    print(r2.__dict__)
    #Print loaded state