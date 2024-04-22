#!/bin/zsh

cd TWA-sensetagged

for file in *.test
do
  echo "Processing $file....."
  python ../wsd_copy.py "$file"

done
