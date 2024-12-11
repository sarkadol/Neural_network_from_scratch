#!/bin/bash
## change this file to your needs

echo "Adding some modules"

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use compiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network
rm -rf out/*
module load jdk-17

#javac -d out -sourcepath src $(find src -name "*.java")
javac -J-Xmx4G -J-XX:+UnlockExperimentalVMOptions -J-XX:+UseParallelGC -d out -sourcepath src $(find src -name "*.java")
# javac: Java compiler to compile .java files into .class files.
# -J-Xmx4G: Allocates up to 4 GB heap memory for the compiler's JVM.
# -J-XX:+UnlockExperimentalVMOptions: Enables experimental JVM options.
# -J-XX:+UseParallelGC: Uses Parallel Garbage Collector for faster compilation.
# -d out: Outputs compiled .class files into the "out" directory.
# -sourcepath src: Specifies the "src" directory for source files.
# $(find src -name "*.java"): Finds all .java files in the "src" directory.

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network

nice -n 19 java -cp out src.Main

if [ $? -ne 0 ]; then
  echo "Java program execution failed!"
  exit 1
fi

