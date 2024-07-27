
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
from tqdm import tqdm

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

def threader(q: Queue, data_queue: Queue, tqdm_queue: Queue):
    data = Counter()
    while True:
        # gets a worker from the queue
        worker = q.get()
        if worker is None:
            break

        # Run the example job with the avail worker in queue (thread)

        parse(worker, data)

        # This is to get a bit of advance over the uploading of the data
        if data_queue.empty():
            data_queue.put(data)
            data = Counter()

        tqdm_queue.put("file")
    tqdm_queue.put(None)
    logging.debug("Putting data on the queue")
    data_queue.put(data)
    logging.debug("Data sent")
    data_queue.put(None)


def depile_thread(num_threads, save_dir, data_queue: Queue, tqdm_queue: Queue):
    data = Counter()
    none_count = 0
    while none_count < num_threads:
        logging.debug("Getting data from the queue")
        res = data_queue.get()
        if res is None:
            tqdm_queue.put("process")
            logging.debug(f"Processes finished")
            none_count += 1
            continue
        logging.debug("Updating data")
        data.update(res)
    
    logging.info(f"Writing data in {save_dir}")
    new_data = defaultdict(lambda: defaultdict(int))
    for (domain, title), views in data.items():
        new_data[domain][title] = views
    for domain in new_data:
        with open(os.path.join(save_dir, f"{domain}.csv"), "w") as f:
            title_with_count = [(title, count) for title, count in new_data[domain].items()]
            title_with_count.sort(key=lambda x: x[1], reverse=True)
            for title, count in title_with_count:
                f.write(f"{title} {count}\n")

def tqdm_process(tqdm_queue: Queue, num_threads: int, n_files: int):
    none_count = 0
    process_early_finish = 0
    with tqdm(total=n_files, dynamic_ncols=True) as pbar:
        while none_count < num_threads:
            res = tqdm_queue.get()
            if res == "file":
                pbar.update()
            elif res == "process":
                process_early_finish += 1
            else:
                none_count += 1

    with tqdm(total=num_threads, dynamic_ncols=True) as pbar:
        pbar.update(process_early_finish)
        while process_early_finish < num_threads:
            res = tqdm_queue.get()
            if res == "process":
                pbar.update()
                process_early_finish += 1



def start_threads(num_threads, save_dir, q: Queue, data_queue: Queue, tqdm_queue: Queue, n_files: int):
    threads = []
    for x in range(num_threads):
        time.sleep(0.05)
        t = multiprocessing.Process(target=threader, args=(q, data_queue, tqdm_queue))

         # classifying as a daemon, so they will die when the main dies
        t.daemon = True
        # print("Thread started")
        # begins, must come after daemon definition
        t.start()
        threads.append(t)

    gather_t = multiprocessing.Process(target=depile_thread, args=(num_threads,save_dir, data_queue, tqdm_queue))
    gather_t.daemon =True
    gather_t.start()
    threads.append(gather_t)
    tqdm_t = multiprocessing.Process(target=tqdm_process, args=(tqdm_queue, num_threads, n_files))
    tqdm_t.daemon =True
    tqdm_t.start()
    threads.append(tqdm_t)
    return threads


def main():

    # bad_files = []
    start = time.time()
    files_dir = sys.argv[1]
    save_dir = sys.argv[2]
    num_threads = int(sys.argv[3])
    soft_limit = 0 if len(sys.argv) < 5 else int(sys.argv[4])

    print("Loading Files dir: ", files_dir)
    files = get_files(files_dir)

    q = Queue()
    data_queue = Queue(maxsize=soft_limit)
    tqdm_queue = Queue()

    for worker in files:
        q.put(worker)
    for _ in range(num_threads):
        q.put(None)

    threads = start_threads(num_threads, save_dir, q, data_queue, tqdm_queue, len(files))

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