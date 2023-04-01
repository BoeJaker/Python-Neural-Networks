"""
filename: wet_requests.py
Description:
"""

import requests
import gzip
from io import BytesIO
import warcio

# Set the URL of the WET file paths for the March 2023 crawl
url = 'https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-06/wet.paths.gz'

# Download the list of WET file paths
response = requests.get(url)
compressed_file = response.content

# Decompress the file
file_content = gzip.decompress(compressed_file)

# Split the file content into individual file paths
file_paths = file_content.decode().split()

# Loop over each file path, download the corresponding WET file, and print its contents
for path in file_paths:
    # Construct the URL of the WET file
    print(path)
    wet_url = 'https://data.commoncrawl.org/' + path
    
    # Download the WET file
    response = requests.get(wet_url)
    compressed_file = response.content
    
    # Decompress the file
    file_content = gzip.decompress(compressed_file)
    
    # Create a WARC iterator
    records = warcio.ArchiveIterator(BytesIO(file_content))
    
    # Iterate over the records and print the contents of the first record
    for record in records:
        print(record.content_stream().read().decode('utf-8','ignore'))
        