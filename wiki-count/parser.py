
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

def get_files(path):
    """ Returns a list of files in a directory
        Input parameter: path to directory
    """
    mypath = path
    complete = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return complete

def load_names_df(path):
    """ Loads pandas df with names from a csv file
    """    
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(["names_u"], axis=1)
    # print(df)
    df.columns = ["name"]
    return df 


def parse(path_old, path_new, project="en"):
    """ Reads file, eliminates unneeded data, filters for project "en" and sspecified names
    """
    global bad_files

    f_name = path_old.split("/")[-1]

    try:
        df = pd.read_csv(path_old, sep=" ")
        df.columns = ["project", "name", "views", "size"]
        df = df[df["project"] == project]
        df = df.drop(["size","project"], axis=1)
        path_new = path_new + f_name
        df.to_csv(path_new, sep=" ",compression="gzip", index=False, header=False)
        print("{} > {}, DONE! ".format(path_old, path_new))
    except:
        try:
            df = pd.read_csv(path_old, sep=" ", encoding="latin_1")
            df.columns = ["project", "name", "views", "size"]
            df = df[df["project"] == project]
            df = df.drop(["size","project"], axis=1)
            path_new = path_new + f_name
            df.to_csv(path_new, sep=" ",compression="gzip", index=False, header=False)
            print("{} > {}, DONE! ".format(path_old, path_new))
        except:
            print("SKIP")


def threader(save_dir, project):
    global q
    while q.empty() != True:
        # gets a worker from the queue
        worker = q.get()

        # Run the example job with the avail worker in queue (thread)
        parse(worker, save_dir, project) 

        # completed with the job
        q.task_done()


def start_threads(num_threads, save_dir, project):

    for x in range(num_threads):
        time.sleep(0.05)
        t = threading.Thread(target=threader, args=(save_dir, project,))

         # classifying as a daemon, so they will die when the main dies
        t.daemon = True
        # print("Thread started")
        # begins, must come after daemon definition
        t.start()


def main():
    global q
    # bad_files = []
    start = time.time()
    files_dir = sys.argv[1]
    save_dir = sys.argv[2]
    project = sys.argv[3]
    num_threads = int(sys.argv[4])

    print("Loading Files dir: ", files_dir)
    files = get_files(files_dir)

    q = Queue()

    for worker in files:
        q.put(worker)

    start_threads(num_threads, save_dir, project)

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