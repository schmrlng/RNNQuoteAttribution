#!/bin/sh

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd futurama
scrapy runspider scrape_imsdb.py
cd ../prideprejudice
wget https://webdocs.cs.ualberta.ca/~kondrak/austen/PridePrejudice_Austen.zip
unzip PridePrejudice_Austen.zip