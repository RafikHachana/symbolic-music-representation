#!/bin/bash

wget --no-check-certificate "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
mkdir dataset
tar -xvzf clean_midi.tar.gz -C dataset
rm clean_midi.tar.gz