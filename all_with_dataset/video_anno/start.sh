#!/bin/bash
num=$1
python track_labeler.py data/${num}.mov data/labels_name.txt output
