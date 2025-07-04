#!/bin/bash

# Get the filename without extension
filename=$1
python $filename.py
pdflatex $filename.tex 