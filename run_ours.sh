#!/bin/bash

echo "#########################"
echo "COMPILING JAVA PROJECT..."
echo "#########################"

# Compile Java files with optimization
javac -d out -sourcepath src src/Main.java

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "#########################"
echo "   RUNNING JAVA CODE    "
echo "#########################"

# Run the Java program and pass hyperparameters if needed
java -cp out src.Main

if [ $? -ne 0 ]; then
  echo "Java program execution failed!"
  exit 1
fi

echo "#########################"
echo " RUNNING PYTHON SCRIPT  "
echo "#########################"

# Run the Python script for generating graphs and evaluation
python3 src/show_the_picture.py

if [ $? -ne 0 ]; then
  echo "Python script execution failed!"
  exit 1
fi

echo "#########################"
echo "  FINISHED SUCCESSFULLY "
echo "#########################"
