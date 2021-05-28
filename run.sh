#!/bin/bash
# example run script.

EXPDIR='../test_output'
mkdir -p ${EXPDIR}

for top in FC
do
  for n in 3000 5000
  do
    for pfixed in 0
    do
      python3 main.py --nagents ${n} \
                       --top ${top} \
                       --k 20 \
                       --micro \
                       --d 0.2 \
                       --pfixed ${pfixed} \
                       --saveloc ${EXPDIR} > ${EXPDIR}/alphafix_log_${top}_${n}_with_rational_households.txt &
    done
  done
done