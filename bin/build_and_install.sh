#!/bin/bash

# build and pip install for test
cd .. && python setup.py sdist
echo $pwd
ls ./dist/ | xargs -i echo pip install ./dist/{} | sh

echo $?
