#!/bin/bash
# example run script.

EXPDIR='test_output/sweep1'
mkdir -p ${EXPDIR}

pfixed=0
top='FC'
for n in 500 600
do
  for tau in 50 100 1000
  do
    python3 main.py --nagents ${n} \
                     --top ${top} \
                     --k 20 \
                     --micro \
                     --d 0.2 \
                     --tau ${tau} \
                     --pfixed ${pfixed} \
                     --saveloc ${EXPDIR} > ${EXPDIR}/log_file_${top}_${n}_${tau}_with_rational_households.txt &
  done
done
