#!/bin/bash

echo "#########################"
echo "CLEANING OLD FILES..."
echo "#########################"
rm -rf out/*

echo "#########################"
echo "COMPILING JAVA PROJECT..."
echo "#########################"
javac -J-Xmx4G -J-XX:+UnlockExperimentalVMOptions -J-XX:+UseParallelGC -d out -sourcepath src $(find src -name "*.java")
#java -cp out -XX:+TieredCompilation -XX:+AggressiveOpts -XX:+UseParallelGC src.Main


if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "#########################"
echo "    RUNNING JAVA CODE    "
echo "#########################"
java -cp out src.Main

if [ $? -ne 0 ]; then
  echo "Java program execution failed!"
  exit 1
fi

echo "#########################"
echo " RUNNING PYTHON SCRIPT  "
echo "#########################"
python src/helpers/Print_results.py

if [ $? -ne 0 ]; then
  echo "Python script execution failed!"
  exit 1
fi

echo "#########################"
echo "  FINISHED SUCCESSFULLY  "
echo "#########################"
