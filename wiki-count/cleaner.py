
import threading
from queue import Queue
import sys, time
import pandas as pd
import requests
import time
import os
from os import listdir
from os import listdir
from os.path import isfile, join
import gc
from collections import defaultdict
import gzip
import logging
import time
import re

logging.basicConfig(level=logging.INFO)

def get_files(path):
    """ Returns a list of files in a directory
        Input parameter: path to directory
    """
    mypath = path
    print(os.listdir(mypath))
    complete = [os.path.join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and re.match(r"pageviews\-[0-9]{8}\-[0-9]{6}\.gz", f)]
    return complete

def load_names_df(path):
    """ Loads pandas df with names from a csv file
    """    
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(["names_u"], axis=1)
    # print(df)
    df.columns = ["name"]
    return df 

def download(url, save_path, lib="requests", retry=5, sleep_time=0.5):
    """ Downloads file to specified path, if server returns status codes other than 200 url is saved
    """
    # the speed diference is insignificant between 'requests' and 'wget'

    f_name = url.split("/")[-1]
    save = save_path+"/"+f_name

    if lib == "requests":
        
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            # next_run.append(url)
            if retry > 0:
                time.sleep(sleep_time)
                download(url, save_path, lib=lib, retry=retry-1, sleep_time=sleep_time * 2)
            else:
                print(f"Failed with {url}")
            # print("Added {} to NEW ROUND (bach size: {})".format(f_name, len(next_run)))
        
        print("Downloading {}".format(url))

        with open(save, 'wb') as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)


def parse(path_old, re_download):
    """ Reads file, eliminates unneeded data, filters
    """

    save_path = os.path.dirname(path_old)
    filename = os.path.basename(path_old)

    try:
        print(f"reading {path_old}")
        with open(path_old, "rb") as f:
            if (f.read(2) != b'\x1f\x8b'):
                raise Exception(f"The file is not a GZIP file: {path_old}")
        with gzip.open(path_old) as f:
            _ = f.read()
    except Exception:
        logging.info(f"Bad GZIP file: {path_old}")
        os.remove(path_old)
        if re_download:
            date = filename.split("-")[1]
            download(f"https://dumps.wikimedia.org/other/pageviews/{date[:4]}/{date[:4]}-{date[4:6]}/{filename}", save_path)



def threader():
    global q, re_download
    while q.empty() != True:
        # gets a worker from the queue
        worker = q.get()

        # Run the example job with the avail worker in queue (thread)
        parse(worker, re_download) 

        # completed with the job
        q.task_done()


def start_threads(num_threads):

    for x in range(num_threads):
        time.sleep(0.05)
        t = threading.Thread(target=threader)

         # classifying as a daemon, so they will die when the main dies
        t.daemon = True
        # print("Thread started")
        # begins, must come after daemon definition
        t.start()


def main():
    global q, re_download
    # bad_files = []
    start = time.time()
    files_dir = sys.argv[1]
    num_threads = int(sys.argv[2])
    re_download = sys.argv[3].lower().strip() == "true"


    print("Loading Files dir: ", files_dir)
    files = get_files(files_dir)

    print(f"Num files: {len(files)}")

    q = Queue()

    for worker in files:
        q.put(worker)

    start_threads(num_threads)

    q.join()


    end = time.time()
    duration = end-start

    if duration > 60:
        print("Time used: {} {}".format(duration/60, "minutes"))
    if duration > 3600:
        print("Time used: {} {}".format(duration/3600, "hours"))
    else:
        print("Time used: {} {}".format(duration, "seconds"))

    # pd.DataFrame(bad_files).to_csv("bad_"+names_file.split("/")[-1])

if __name__ == '__main__':
    main() # type: ignore