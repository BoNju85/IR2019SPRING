#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.
python3 vsm.py -i queries/query-test.xml -o submission.csv -m model -d CIRB010
