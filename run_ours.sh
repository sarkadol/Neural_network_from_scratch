#!/bin/bash

echo "#########################"
echo "CLEANING OLD FILES..."
echo "#########################"
rm -rf out/*

echo "#########################"
echo "COMPILING JAVA PROJECT..."
echo "#########################"
javac -d out -sourcepath src $(find src -name "*.java")

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "#########################"
echo "   RUNNING JAVA CODE    "
echo "#########################"
java -cp out src.Main

if [ $? -ne 0 ]; then
  echo "Java program execution failed!"
  exit 1
fi

echo "#########################"
echo "  FINISHED SUCCESSFULLY "
echo "#########################"
