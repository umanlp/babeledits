# Counts pageviews per languages

This is a Python pipeline that allows to count the pageviews for specific pages
on Wikipedia and separates them by languages. 

It is a fork of
[gesiscss/wiki-donwload-parse-page-views](https://github.com/gesiscss/wiki-download-parse-page-views)
which separates pageviews by languages.

# How it works?

This pipeline follows three steps:

1. fetch file names of the dump files to download (with `fetch_file_names.py`)
2. download the files (with `downloader.py`)
3. parse the dump files to build the counting data (`full_parser`)

**Warning**: the Wikipedia dumps can be quite large, so plan ahead for disk space. For example, if you are downloading 2021 data, you will need 427GB of space, which contains 419GB of dump files (that you can delete once the pipeline ended) and the result itself is 8.1GB.

## Installation

This repository uses `uv` to manage dependencies. Alternatively, a `requirements.txt` is available for creating any kind of virtual environment.

To install everything, just run (provided that `uv` is installed):

```
uv sync
```

## Fetch file names

First, you need to fetch the address of the Wikipedia dumps that correspond to the time window that you're interested in.

```
uv run fetch_file_names.py [start_year] [end_year] [destination]
```

For each year between `[start_year]` and `[end_year]` (bounds included), it will create a file `[year].csv` inside `[destination]`, which will look like this:

file | size |url |
--- |--- |--- |
pagecounts-20140101-000000.gz |82| https://.. |
pagecounts-20140201-000000.gz |81| https://.. |
... | ... | ... |

## Downloading data

This script concurently downloads [Wikipedia pagecount dumps](https://dumps.wikimedia.org/other/pagecounts-raw/) [qzip]. The file previously generated **file.csv** contains a list of urls for the files mentioned. The **dumps_directory** refers to directory where files should be downloaded. 

```
uv run downloader.py [file.csv] [dumps_directory] [thread_number]
```
**THE SERVER IS CURRENTLY BLOCKING IN CASE OF USING MORE THEN 3 THREADS**

If it fails donwloading some of the files, you can run the `cleaner.py` scripts that tries to donwload them again.

## Parsing data

To parse the dumps, you can run:

```
uv run full_parser.py [dumps_directory] [output_directory] [num_threads]
```

It will create inside `[output_directory]` one file for each language under the name `[lang_code].txt` which looks like follows:

```{txt}
Main_Page 2146239511
Special:Search 520609143
- 122015857
Bible 48749947
Cleopatra 46014921
Deaths_in_2021 44892478
Microsoft_Office 26266066
Elon_Musk 25335944
XXXX 25287619
Elizabeth_II 24744205
```