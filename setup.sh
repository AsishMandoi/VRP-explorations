#!/bin/bash

echo "Hello World!"
echo

if [ -d "_venv_" ] ; then
  echo "Existing virtual environment found."
else
  echo "Creating new virtual environment..."
  python -m venv _venv_
fi

source _venv_/Scripts/activate

echo
echo "Installing requirements..."
python -m pip install -U pip
python -m pip install -r requirements.txt

echo
echo "Running 'dwave setup'..."
echo

dwave setup

echo
echo "You're all set!"