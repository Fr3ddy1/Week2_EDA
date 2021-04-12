import logging
import os
import subprocess
import yaml
import pandas as pd
import datetime 
import gc
import re

#CREATE A SUPPORT FILE

################
# File Reading #
################

def read_yaml_file(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
