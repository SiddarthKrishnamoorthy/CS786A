#!/bin/sh

iters=$1
file=$2
i=0
counter=0
while [ $i -le $iters ]
do
    n=$(python3 ques$file.py | grep "Number of unique retrievals" | grep -o '[0-9]*')
    ((counter += $n))
    ((i++))
done
echo $counter
