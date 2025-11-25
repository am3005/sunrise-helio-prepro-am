"""
This file is adapted from the Kasper-Heliophysics-MDP/Prepro-F25 repository.
Original implementation: Fall 2025 Kasper Heliophysics MDP cohort (Aashi Mishra listed as coauthor).

Adapted & refactored for personal preprocessing project work presentation:
Aashi Mishra
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
from astropy.io import fits
import gzip
import re
from typing import List
import io
from tqdm import tqdm
import sys

BASE_URL = "https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"
BASE_LABELS_URL = "https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein"

def download_fits_from_gz(url: str) -> np.ndarray:
    """Download a .fit.gz URL and return the FITS data as a numpy array."""
    r = requests.get(url, stream=True)
    r.raise_for_status()

    # Decompress in memory
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            data = hdul[0].data
            return np.array(data)
        
def circular_sort(files: List[str], offset: str, url: str) -> List[str]:
    """
    The times on eCallisto are in UTC. So the beginning of the day locally
    is at an offset time, and the times are sequential from then.
    Circularly sort files by time (HHMMSS in filename) starting from offset.
    
    Args:
        files: list of filenames
        offset: string "HHMMSS" (UTC offset start)
        url: just putting the url to print in case of an error
        
    Returns:
        List[str]: circularly sorted list of filenames
    """
    
    def hhmmss_to_seconds(hhmmss: str) -> int:
        h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
        return h * 3600 + m * 60 + s

    # regex to extract the middle part with time
    time_re = re.compile(r"_(\d{6})_")

    #testing on the first file
    match = time_re.search(files[0])

    #some of the older files have an i (intensity after the time)
    if match is None:
        time_re = re.compile(r"_(\d{6})i")
        match = time_re.search(files[0])

    #error if there's some other time format that we haven't noticed
    if match is None:
        print(f"ERROR: No matching time formats found at: {url}")
        exit(0)
    
    # map files to (time_in_seconds, filename)
    time_file_pairs = []
    for f in files:
        match = time_re.search(f)
        if match:
            t = hhmmss_to_seconds(match.group(1))
            time_file_pairs.append((t, f))
    
    # sort by time normally
    time_file_pairs.sort(key=lambda x: x[0])
    
    # compute offset in seconds
    offset_sec = hhmmss_to_seconds(offset)
    
    # rotate list so it starts from the first time >= offset
    times = [t for t, _ in time_file_pairs]
    idx = next((i for i, t in enumerate(times) if t >= offset_sec), 0)
    
    sorted_files = [f for _, f in time_file_pairs[idx:]] + [f for _, f in time_file_pairs[:idx]]
    
    return sorted_files

def find_bursts(arr, burst_list, filename, results, current_idx):
    """
    Checks if there are any bursts in this file, if so,
    finds the indices of the burst start/end 
    
    Args:
        arr (np.array): the spectrogram
        burst_list (dict): dict of strings specifying the burst start/ends as given in the eCallisto labels txt file 
        filename (str): the name of this file
        results (dict): A running dictionary that maps the labels to the start/end indices
        current_idx (int): Keeps a running track of the time index as the arr get concatenated
        
    Returns:
        updated results dict and current_idx int
    """

    def parse_time_str(tstr):
        """Convert HH:MM or HH:MM:SS string into seconds since midnight."""
        parts = list(map(int, tstr.split(":")))
        if len(parts) == 2:  # HH:MM
            h, m = parts
            s = 0
        else:                # HH:MM:SS
            h, m, s = parts
        return h * 3600 + m * 60 + s
    
    _, nx = arr.shape
    file_start_time = re.search(r"_(\d{6})_", filename)
    hhmmss = file_start_time.group(1)
    file_start_sec = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])
    file_end_sec = file_start_sec + nx / 4  # should be file_start + 900

    for burst in burst_list:
        bstart_str, bend_str = burst.split("-")
        bstart_sec = parse_time_str(bstart_str)
        bend_sec = parse_time_str(bend_str)

        # Check if burst overlaps file range
        if bstart_sec < file_end_sec and bend_sec >= file_start_sec:
            # Clip burst to file boundaries
            start_idx = max(0, int((bstart_sec - file_start_sec) * 4))
            end_idx = min(nx - 1, int((bend_sec - file_start_sec) * 4))
            start_idx = start_idx + current_idx - 4*60 #subtract 60 seconds
            end_idx = end_idx + current_idx + 4*60 #add 60 seconds
            results.append({
                "burst": burst,
                "start_idx": start_idx,
                "end_idx": end_idx
            })

    current_idx = current_idx + nx

    return results, current_idx

def one_day(station: str, year: int, month: int, day: int, time: str = "000000", burst_list: dict = None):
    """
    Collects all eCallisto recordings from a given station on a given day
    Puts them in order, concatenates them, and returns them as a numpy array
    
    Args:
        station (str): The name of the eCallisto station 
        year (int): 
        month (int):
        day (int): The date to look at
        time (str): UTC time that designates the beginning of the day in "HHMMSS" format
        
    Returns:
        numpy arr: 2D spectrogram over all files found
    """
    #get the url from the date
    path = f"{int(year):04d}/{int(month):02d}/{int(day):02d}/"
    url = urljoin(BASE_URL.rstrip("/") + "/", path)

    print(url)
    #extract a list of all files
    response = requests.get(url)
    response.raise_for_status()  # raise error if bad request
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    files = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # skip going back to parent dir
        if href in ("../", "./"):
            continue
        # construct absolute URL
        full_url = urljoin(url, href)
        files.append(full_url)

    station_files = [f for f in files if station in f] #extract only the files at this station
    if len(station_files) == 0:
        print(f"ERROR: No match found for {station} found at {str(url)}")
        exit(0)

    sorted_files = circular_sort(station_files, time, url) #put them in order

    arrays = []
    burst_indices = []
    current_idx = 0
    
    for url in tqdm(sorted_files, desc="Downloading FITS files"):

        arr = download_fits_from_gz(url)
        if arr is not None:

            if burst_list is not None:
                burst_indices, current_idx = find_bursts(arr, burst_list, url, burst_indices, current_idx)
                
            arrays.append(arr)

    if not arrays:
        raise ValueError("No valid FITS data found.")
    
    # Concatenate along time axis (axis=0)
    big_array = np.concatenate(arrays, axis=1)
    big_array = np.flipud(big_array)
    return big_array, burst_indices

def extract_bursts(station, year, month, day):
    """
    Downloads a txt file from eCallisto of burst labels on a given day.
    Extracts the starting and ending times for all bursts observed at station
    
    Args:
        station (str): The name of the eCallisto station 
        year (int): 
        month (int):
        day (int): The date to look at
        
    Returns:
        numpy arr (str): 
            The value under the time column for every burst observed at station (eg "03:00-03:01")
    """

    #WARNING: this url will not work if the year is before 2010
    #get the url from the date
    path = f"{int(year):04d}/e-CALLISTO_{int(year):04d}_{int(month):02d}.txt"
    url = urljoin(BASE_LABELS_URL.rstrip("/") + "/", path)

    # download text file
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    
    bursts = []
    date = f"{int(year):04d}{int(month):02d}{int(day):02d}"
    for line in lines:
        if not line.strip() or line.startswith("#") or line.startswith("-"):
            continue  # skip comments/headers
        
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        
        line_date, time_range, _, stations = parts
        if line_date == str(date):
            station_list = [s.strip() for s in stations.split(",")]
            if station in station_list:
                bursts.append(time_range)
    
    return bursts

def usage():
    print("Usage: python one_day.py <station> <month> <day> <year> [start_time] [--save_burst_labels]")
    print()
    print("Arguments:")
    print("  station              Name of the observatory station (e.g. GERMANY-DLR)")
    print("  year                 Four-digit year (e.g. 2025)")
    print("  month                Two-digit month (01)")
    print("  day                  Two-digit day (31)")
    print("  start_time (optional) Time of the first recording as seen in the filename (e.g. 093000 for 09:30:00)")
    print("  --save_burst_labels  Optional switch; if present, creates a npy file with the indexes when bursts start/end")
    sys.exit(1)

def parse_args(argv):
    # Required minimum = 4 args + script name
    if len(argv) < 5:
        usage()

    station = argv[1]
    month = argv[2]
    day = argv[3]
    year = argv[4]

    start_time = None
    save_burst_labels = False

    # Check for optional args
    for arg in argv[5:]:
        if arg == "--save_burst_labels":
            save_burst_labels = True
        else:
            # If it looks like HHMMSS, treat as start_time
            if arg.isdigit() and len(arg) == 6:
                start_time = arg
            else:
                print(f"Unknown argument: {arg}")
                usage()

    if start_time is None:
        start_time = "000000"

    return station, year, month, day, start_time, save_burst_labels

# Example usage:
if __name__ == "__main__":

    station, year, month, day, start_time, save_burst_labels = parse_args(sys.argv)

    save_file_post_fix = "-" + station + "-" + str(f"{int(month):02d}") + "-" + str(f"{int(day):02d}") + "-" + str(f"{int(year):04d}") + ".npy"
    if save_burst_labels:
        data, indices = one_day(station, year, month, day, start_time, extract_bursts(station, year, month, day))
        np.save("labels" + save_file_post_fix, indices) 
    else:
        data, _ = one_day(station, year, month, day, start_time)

    np.save("spec" + save_file_post_fix, data)