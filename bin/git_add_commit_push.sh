#!/bin/bash

# build and pip install for test
cd .. 
git add --all ./** 
git commit -m $1 
git push origin master
echo $?
