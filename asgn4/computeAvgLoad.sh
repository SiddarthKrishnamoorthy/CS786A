#!/bin/sh

iters=$1
file=$2
i=0
counter=0
while [ $i -le $iters ]
do
    n=$(python3 ques$file.py | grep "Scheduling load" | grep -o '[0-9]*\.[0-9]*')
    counter=$(echo "$counter + $n" | bc -l)
    ((i++))
done
#echo $((counter / iters))
echo $(echo "$counter / $iters" | bc -l)
