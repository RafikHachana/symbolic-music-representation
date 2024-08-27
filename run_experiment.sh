#!/bin/bash

# Remove the hard cache
rm absolute_start.tmp
python3 model.py -pe time_pitch -maxlen 512 -b 28
python3 model.py -pe base -maxlen 512 -b 28
python3 model.py -pe time_pitch -maxlen 512 -b 28 --no-concepts
python3 model.py -pe base -maxlen 512 -b 28 --no-concepts
# python3 model.py -pe time -maxlen 512 -b 28
# python3 model.py -pe pitch -maxlen 512 -b 28

# Remove the hard cache
# rm absolute_start.tmp


# python3 model.py -pe time_pitch -maxlen 256 -b 28
# python3 model.py -pe base -maxlen 256 -b 28
# python3 model.py -pe time -maxlen 256 -b 28
# python3 model.py -pe pitch -maxlen 256 -b 28


