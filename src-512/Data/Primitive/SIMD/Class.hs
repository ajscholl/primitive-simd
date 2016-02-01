{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE MagicHash             #-}
{-# OPTIONS_GHC -fno-warn-orphans  #-}
module Data.Primitive.SIMD.Class where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Control.Monad.Primitive
import Data.Primitive

import GHC.Exts

-- | The compiler only supports tuples up to 62 elements, so we have to use our own data type.
data Tuple64 a = Tuple64 a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a

-- * SIMD type classes

-- | This class provides basic operations to create and consume SIMD types.
--   Numeric operations on members of this class should compile to single
--   SIMD instructions although not all operations are (yet) supported by
--   GHC (e.g. 'sqrt', it is currently implemented as @mapVector sqrt@ which
--   has to unpack the vector, compute the results and pack them again).
class (Num v, Real (Elem v)) => SIMDVector v where
    -- | Type of the elements in the vector
    type Elem v
    -- | Type used to pack or unpack the vector
    type ElemTuple v
    -- | Vector with all elements initialized to zero.
    nullVector       :: v
    -- | Number of components (scalar elements) in the vector. The argument is not evaluated.
    vectorSize       :: v -> Int
    -- | Size of each (scalar) element in the vector in bytes. The argument is not evaluated.
    elementSize      :: v -> Int
    -- | Broadcast a scalar to all elements of a vector.
    broadcastVector  :: Elem v -> v
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range an exception is thrown.
    insertVector     :: v -> Elem v -> Int -> v
    insertVector v e i | i < 0            = error $ "insertVector: negative argument: " ++ show i
                       | i < vectorSize v = unsafeInsertVector v e i
                       | otherwise        = error $ "insertVector: argument too large: " ++ show i
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range the behavior is undefined.
    unsafeInsertVector     :: v -> Elem v -> Int -> v
    -- | Apply a function to each element of a vector. Be very careful not to map
    --   branching functions over a vector as they could lead to quite a bit of
    --   code bloat (or make sure they are tagged with NOINLINE).
    mapVector        :: (Elem v -> Elem v) -> v -> v
    -- | Zip two vectors together using a combining function.
    zipVector        :: (Elem v -> Elem v -> Elem v) -> v -> v -> v
    -- | Fold the elements of a vector to a single value. The order in which
    --   the elements are combined is not specified.
    foldVector       :: (Elem v -> Elem v -> Elem v) -> v -> Elem v
    -- | Sum up the components of the vector. Equivalent to @foldVector (+)@.
    sumVector        :: v -> Elem v
    sumVector        = foldVector (+)
    -- | Pack some elements to a vector.
    packVector       :: ElemTuple v -> v
    -- | Unpack a vector.
    unpackVector     :: v -> ElemTuple v

-- | Provides vectorized versions of 'quot' and 'rem'. Implementing their 
--   type class is not possible for SIMD types as it would require
--   implementing 'toInteger'.
class SIMDVector v => SIMDIntVector v where
    -- | Rounds towards zero element-wise.
    quotVector :: v -> v -> v
    -- | Satisfies @(quotVector x y) * y + (remVector x y) == x@.
    remVector  :: v -> v -> v

{-# INLINE setByteArrayGeneric #-}
setByteArrayGeneric :: (Prim a, PrimMonad m) => MutableByteArray (PrimState m) -> Int -> Int -> a -> m ()
setByteArrayGeneric mba off n v | n <= 0 = return ()
                                | otherwise = do
    writeByteArray mba off v
    setByteArrayGeneric mba (off + 1) (n - 1) v

{-# INLINE setOffAddrGeneric #-}
setOffAddrGeneric :: (Prim a, PrimMonad m) => Addr -> Int -> Int -> a -> m ()
setOffAddrGeneric addr off n v | n <= 0 = return ()
                               | otherwise = do
    writeOffAddr addr off v
    setOffAddrGeneric addr (off + 1) (n - 1) v

{-# RULES "unpack/pack Int8X16#" forall x . unpackInt8X16# (packInt8X16# x) = x #-}
{-# RULES "pack/unpack Int8X16#" forall x . packInt8X16# (unpackInt8X16# x) = x #-}
{-# RULES "unpack/pack Int8X32#" forall x . unpackInt8X32# (packInt8X32# x) = x #-}
{-# RULES "pack/unpack Int8X32#" forall x . packInt8X32# (unpackInt8X32# x) = x #-}
{-# RULES "unpack/pack Int8X64#" forall x . unpackInt8X64# (packInt8X64# x) = x #-}
{-# RULES "pack/unpack Int8X64#" forall x . packInt8X64# (unpackInt8X64# x) = x #-}
{-# RULES "unpack/pack Int16X8#" forall x . unpackInt16X8# (packInt16X8# x) = x #-}
{-# RULES "pack/unpack Int16X8#" forall x . packInt16X8# (unpackInt16X8# x) = x #-}
{-# RULES "unpack/pack Int16X16#" forall x . unpackInt16X16# (packInt16X16# x) = x #-}
{-# RULES "pack/unpack Int16X16#" forall x . packInt16X16# (unpackInt16X16# x) = x #-}
{-# RULES "unpack/pack Int16X32#" forall x . unpackInt16X32# (packInt16X32# x) = x #-}
{-# RULES "pack/unpack Int16X32#" forall x . packInt16X32# (unpackInt16X32# x) = x #-}
{-# RULES "unpack/pack Int32X4#" forall x . unpackInt32X4# (packInt32X4# x) = x #-}
{-# RULES "pack/unpack Int32X4#" forall x . packInt32X4# (unpackInt32X4# x) = x #-}
{-# RULES "unpack/pack Int32X8#" forall x . unpackInt32X8# (packInt32X8# x) = x #-}
{-# RULES "pack/unpack Int32X8#" forall x . packInt32X8# (unpackInt32X8# x) = x #-}
{-# RULES "unpack/pack Int32X16#" forall x . unpackInt32X16# (packInt32X16# x) = x #-}
{-# RULES "pack/unpack Int32X16#" forall x . packInt32X16# (unpackInt32X16# x) = x #-}
{-# RULES "unpack/pack Int64X2#" forall x . unpackInt64X2# (packInt64X2# x) = x #-}
{-# RULES "pack/unpack Int64X2#" forall x . packInt64X2# (unpackInt64X2# x) = x #-}
{-# RULES "unpack/pack Int64X4#" forall x . unpackInt64X4# (packInt64X4# x) = x #-}
{-# RULES "pack/unpack Int64X4#" forall x . packInt64X4# (unpackInt64X4# x) = x #-}
{-# RULES "unpack/pack Int64X8#" forall x . unpackInt64X8# (packInt64X8# x) = x #-}
{-# RULES "pack/unpack Int64X8#" forall x . packInt64X8# (unpackInt64X8# x) = x #-}
{-# RULES "unpack/pack Word8X16#" forall x . unpackWord8X16# (packWord8X16# x) = x #-}
{-# RULES "pack/unpack Word8X16#" forall x . packWord8X16# (unpackWord8X16# x) = x #-}
{-# RULES "unpack/pack Word8X32#" forall x . unpackWord8X32# (packWord8X32# x) = x #-}
{-# RULES "pack/unpack Word8X32#" forall x . packWord8X32# (unpackWord8X32# x) = x #-}
{-# RULES "unpack/pack Word8X64#" forall x . unpackWord8X64# (packWord8X64# x) = x #-}
{-# RULES "pack/unpack Word8X64#" forall x . packWord8X64# (unpackWord8X64# x) = x #-}
{-# RULES "unpack/pack Word16X8#" forall x . unpackWord16X8# (packWord16X8# x) = x #-}
{-# RULES "pack/unpack Word16X8#" forall x . packWord16X8# (unpackWord16X8# x) = x #-}
{-# RULES "unpack/pack Word16X16#" forall x . unpackWord16X16# (packWord16X16# x) = x #-}
{-# RULES "pack/unpack Word16X16#" forall x . packWord16X16# (unpackWord16X16# x) = x #-}
{-# RULES "unpack/pack Word16X32#" forall x . unpackWord16X32# (packWord16X32# x) = x #-}
{-# RULES "pack/unpack Word16X32#" forall x . packWord16X32# (unpackWord16X32# x) = x #-}
{-# RULES "unpack/pack Word32X4#" forall x . unpackWord32X4# (packWord32X4# x) = x #-}
{-# RULES "pack/unpack Word32X4#" forall x . packWord32X4# (unpackWord32X4# x) = x #-}
{-# RULES "unpack/pack Word32X8#" forall x . unpackWord32X8# (packWord32X8# x) = x #-}
{-# RULES "pack/unpack Word32X8#" forall x . packWord32X8# (unpackWord32X8# x) = x #-}
{-# RULES "unpack/pack Word32X16#" forall x . unpackWord32X16# (packWord32X16# x) = x #-}
{-# RULES "pack/unpack Word32X16#" forall x . packWord32X16# (unpackWord32X16# x) = x #-}
{-# RULES "unpack/pack Word64X2#" forall x . unpackWord64X2# (packWord64X2# x) = x #-}
{-# RULES "pack/unpack Word64X2#" forall x . packWord64X2# (unpackWord64X2# x) = x #-}
{-# RULES "unpack/pack Word64X4#" forall x . unpackWord64X4# (packWord64X4# x) = x #-}
{-# RULES "pack/unpack Word64X4#" forall x . packWord64X4# (unpackWord64X4# x) = x #-}
{-# RULES "unpack/pack Word64X8#" forall x . unpackWord64X8# (packWord64X8# x) = x #-}
{-# RULES "pack/unpack Word64X8#" forall x . packWord64X8# (unpackWord64X8# x) = x #-}
{-# RULES "unpack/pack FloatX4#" forall x . unpackFloatX4# (packFloatX4# x) = x #-}
{-# RULES "pack/unpack FloatX4#" forall x . packFloatX4# (unpackFloatX4# x) = x #-}
{-# RULES "unpack/pack FloatX8#" forall x . unpackFloatX8# (packFloatX8# x) = x #-}
{-# RULES "pack/unpack FloatX8#" forall x . packFloatX8# (unpackFloatX8# x) = x #-}
{-# RULES "unpack/pack FloatX16#" forall x . unpackFloatX16# (packFloatX16# x) = x #-}
{-# RULES "pack/unpack FloatX16#" forall x . packFloatX16# (unpackFloatX16# x) = x #-}
{-# RULES "unpack/pack DoubleX2#" forall x . unpackDoubleX2# (packDoubleX2# x) = x #-}
{-# RULES "pack/unpack DoubleX2#" forall x . packDoubleX2# (unpackDoubleX2# x) = x #-}
{-# RULES "unpack/pack DoubleX4#" forall x . unpackDoubleX4# (packDoubleX4# x) = x #-}
{-# RULES "pack/unpack DoubleX4#" forall x . packDoubleX4# (unpackDoubleX4# x) = x #-}
{-# RULES "unpack/pack DoubleX8#" forall x . unpackDoubleX8# (packDoubleX8# x) = x #-}
{-# RULES "pack/unpack DoubleX8#" forall x . packDoubleX8# (unpackDoubleX8# x) = x #-}

