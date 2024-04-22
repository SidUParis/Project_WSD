#!/bin/zsh

cd TWA-sensetagged

for file in *.test
do
  echo "Processing $file....."
  python ../wsd.py "$file"

done
