#!/bin/bash

mkdir -p src-128/Data/Primitive/SIMD
mkdir -p src-256/Data/Primitive/SIMD
mkdir -p src-512/Data/Primitive/SIMD

runhaskell src-generate/Generator.hs 
