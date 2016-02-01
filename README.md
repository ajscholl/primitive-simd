This package provides a lifted version of the SIMD data types
and instructions from ghc-prim.

The implementation is based on type families to provide a
uniform interface to all different vector types. Instances
for numeric operations, Prim, Storable and unboxed vector
instances are provided.

Although not all processors support 256 or 512-bit vectors
this package provides a uniform interface. Vectors which
are bigger than supported are modeled by combining smaller
vectors. If the same code is compiled on a computer supporting
larger vectors the smaller vectors are replaced by larger
vectors.

Note: This package needs to be compiled with LLVM as the NCG
does not know how to deal with SIMD-instructions. If LLVM is
not available, use -f no-vec to disable the use of SIMD instructions.
While this will give you no speedup, it will work with plain
Haskell (and should even work with GHCJS).
