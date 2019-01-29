import os
import json
import ast


for filename in os.listdir('dataset/'):

    with open('dataset/' + filename, 'r') as fp:
        data = ast.literal_eval(fp.read())
        
        json.dump(data, open(filename[:-4] + '.json', 'w'))
