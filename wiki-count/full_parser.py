
import threading
from multiprocessing import Queue
import sys, time
import pandas as pd
import requests
import time
import os
from os import listdir
from os import listdir
from os.path import isfile, join
import gc
from collections import Counter, defaultdict
import gzip
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)

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


def parse(path_old, data):
    """ Reads file, eliminates unneeded data, filters
    """

    try:
        with gzip.open(path_old, "rb") as f:
            line = f.readline().decode().strip()
            while line:
                parts = line.split(" ")
                if len(parts) < 4:
                    continue
                domain, *title, views, _ = parts
                title = " ".join(title)
                data[(domain, title)] += int(views)
                line = f.readline().decode().strip()
    except Exception as e:
        logging.error(f"SKIP {path_old}", exc_info=e)

def threader(q: Queue, data_queue: Queue, soft_limit: int):
    data = Counter()
    accumulations = 0
    while True:
        # gets a worker from the queue
        worker = q.get()
        if worker is None:
            break

        # Run the example job with the avail worker in queue (thread)
        start = time.time()

        parse(worker, data)
        delay = time.time() - start

        if data_queue.qsize() < soft_limit:
            print(f"Last file: {delay:.2f}s. Queue is free (size={data_queue.qsize()}), sending data for {accumulations + 1} files")
            start = time.time()
            data_queue.put(data)
            delay = time.time() - start
            print(f"Delay sending data: {delay:.2f}s")
            data = Counter()
            accumulations = 0
        else:
            accumulations += 1
            print(f"Last file: {delay:.2f}s. Queue has reached soft limit ((size={data_queue.qsize()})), will send data later for {accumulations} accumulated files")

    while accumulations > 0 and data_queue.qsize() < soft_limit:
        time.sleep(0.5)
    data_queue.put(data)
    
    data_queue.put(None)


def depile_thread(num_threads, save_dir, data_queue: Queue):
    none_count = 0
    data = Counter()
    while none_count < num_threads:
        res = data_queue.get()
        if res is None:
            none_count += 1
        else:
            #data[res[0]][res[1]] += res[2]
            data.update(res)
    
    print(f"Writing data in {save_dir}")
    new_data = defaultdict(lambda: defaultdict(int))
    for (domain, title), views in data.items():
        new_data[domain][title] = views
    for domain in data:
        with open(os.path.join(save_dir, f"{domain}.csv"), "w") as f:
            title_with_count = [(title, count) for title, count in new_data[domain].items()]
            title_with_count.sort(key=lambda x: x[1], reverse=True)
            for title, count in title_with_count:
                f.write(f"{title} {count}\n")



def start_threads(num_threads, save_dir, q: Queue, data_queue: Queue, soft_limit: int):
    threads = []
    for x in range(num_threads):
        time.sleep(0.05)
        t = multiprocessing.Process(target=threader, args=(q, data_queue, soft_limit))

         # classifying as a daemon, so they will die when the main dies
        t.daemon = True
        # print("Thread started")
        # begins, must come after daemon definition
        t.start()
        threads.append(t)

    gather_t = multiprocessing.Process(target=depile_thread, args=(num_threads,save_dir, data_queue))
    gather_t.daemon =True
    gather_t.start()
    threads.append(gather_t)
    return threads


def main():

    # bad_files = []
    start = time.time()
    files_dir = sys.argv[1]
    save_dir = sys.argv[2]
    num_threads = int(sys.argv[3])
    soft_limit = 2 * num_threads if len(sys.argv) < 5 else int(sys.argv[4])

    

    print("Loading Files dir: ", files_dir)
    files = get_files(files_dir)

    q = Queue()
    data_queue = Queue()

    for worker in files:
        q.put(worker)
    q.put(None)

    threads = start_threads(num_threads, save_dir, q, data_queue, soft_limit)

    for t in threads:
        t.join()

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