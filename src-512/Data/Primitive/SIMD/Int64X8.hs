{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

#include "MachDeps.h"

module Data.Primitive.SIMD.Int64X8 (Int64X8) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

import GHC.Int

import GHC.Types
import GHC.Exts
import GHC.ST

import Foreign.Storable

import Control.Monad.Primitive

import Data.Primitive.Types
import Data.Primitive.ByteArray
import Data.Primitive.Addr
import Data.Monoid
import Data.Typeable

import qualified Data.Vector.Primitive as PV
import qualified Data.Vector.Primitive.Mutable as PMV
import Data.Vector.Unboxed (Unbox)
import qualified Data.Vector.Unboxed as UV
import Data.Vector.Generic (Vector(..))
import Data.Vector.Generic.Mutable (MVector(..))

#if WORD_SIZE_IN_BITS == 64
type RealInt64# = Int#
#elif WORD_SIZE_IN_BITS == 32
type RealInt64# = Int64#
#else
#error "WORD_SIZE_IN_BITS is neither 64 or 32"
#endif

-- ** Int64X8
data Int64X8 = Int64X8 Int64X8# deriving Typeable

abs' :: Int64 -> Int64
abs' (I64# x) = I64# (abs# x)

{-# NOINLINE abs# #-}
abs# :: RealInt64# -> RealInt64#
abs# x = case abs (I64# x) of
    I64# y -> y

signum' :: Int64 -> Int64
signum' (I64# x) = I64# (signum# x)

{-# NOINLINE signum# #-}
signum# :: RealInt64# -> RealInt64#
signum# x = case signum (I64# x) of
    I64# y -> y

instance Eq Int64X8 where
    a == b = case unpackInt64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt64X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Int64X8 where
    a `compare` b = case unpackInt64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt64X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Int64X8 where
    showsPrec _ a s = case unpackInt64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Int64X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Int64X8 where
    (+) = plusInt64X8
    (-) = minusInt64X8
    (*) = timesInt64X8
    negate = negateInt64X8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int64X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int64X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int64X8 where
    type Elem Int64X8 = Int64
    type ElemTuple Int64X8 = (Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 8
    broadcastVector    = broadcastInt64X8
    generateVector     = generateInt64X8
    unsafeInsertVector = unsafeInsertInt64X8
    packVector         = packInt64X8
    unpackVector       = unpackInt64X8
    mapVector          = mapInt64X8
    zipVector          = zipInt64X8
    foldVector         = foldInt64X8

instance SIMDIntVector Int64X8 where
    quotVector = quotInt64X8
    remVector  = remInt64X8

instance Prim Int64X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt64X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt64X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt64X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt64X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt64X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt64X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int64X8 = V_Int64X8 (PV.Vector Int64X8)
newtype instance UV.MVector s Int64X8 = MV_Int64X8 (PMV.MVector s Int64X8)

instance Vector UV.Vector Int64X8 where
    basicUnsafeFreeze (MV_Int64X8 v) = V_Int64X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int64X8 v) = MV_Int64X8 <$> PV.unsafeThaw v
    basicLength (V_Int64X8 v) = PV.length v
    basicUnsafeSlice start len (V_Int64X8 v) = V_Int64X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int64X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int64X8 m) (V_Int64X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int64X8 where
    basicLength (MV_Int64X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int64X8 v) = MV_Int64X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int64X8 v) (MV_Int64X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int64X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int64X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int64X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int64X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int64X8

{-# INLINE broadcastInt64X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt64X8 :: Int64 -> Int64X8
broadcastInt64X8 (I64# x) = Int64X8 (broadcastInt64X8# x)

{-# INLINE[1] generateInt64X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt64X8 :: (Int -> Int64) -> Int64X8
generateInt64X8 f = packInt64X8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packInt64X8 #-}
-- | Pack the elements of a tuple into a vector.
packInt64X8 :: (Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64) -> Int64X8
packInt64X8 (I64# x1, I64# x2, I64# x3, I64# x4, I64# x5, I64# x6, I64# x7, I64# x8) = Int64X8 (packInt64X8# (# x1, x2, x3, x4, x5, x6, x7, x8 #))

{-# INLINE unpackInt64X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt64X8 :: Int64X8 -> (Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64)
unpackInt64X8 (Int64X8 m1) = case unpackInt64X8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> (I64# x1, I64# x2, I64# x3, I64# x4, I64# x5, I64# x6, I64# x7, I64# x8)

{-# INLINE unsafeInsertInt64X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt64X8 :: Int64X8 -> Int64 -> Int -> Int64X8
unsafeInsertInt64X8 (Int64X8 m1) (I64# y) _i@(I# ip) = Int64X8 (insertInt64X8# m1 y (ip -# 0#))

{-# INLINE[1] mapInt64X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt64X8 :: (Int64 -> Int64) -> Int64X8 -> Int64X8
mapInt64X8 f = mapInt64X8# (\ x -> case f (I64# x) of { I64# y -> y})

{-# RULES "mapVector abs" mapInt64X8 abs = abs #-}
{-# RULES "mapVector signum" mapInt64X8 signum = signum #-}
{-# RULES "mapVector negate" mapInt64X8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt64X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt64X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt64X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt64X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt64X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt64X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt64X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt64X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt64X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt64X8# #-}
-- | Unboxed helper function.
mapInt64X8# :: (RealInt64# -> RealInt64#) -> Int64X8 -> Int64X8
mapInt64X8# f = \ v -> case unpackInt64X8 v of
    (I64# x1, I64# x2, I64# x3, I64# x4, I64# x5, I64# x6, I64# x7, I64# x8) -> packInt64X8 (I64# (f x1), I64# (f x2), I64# (f x3), I64# (f x4), I64# (f x5), I64# (f x6), I64# (f x7), I64# (f x8))

{-# INLINE[1] zipInt64X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt64X8 :: (Int64 -> Int64 -> Int64) -> Int64X8 -> Int64X8 -> Int64X8
zipInt64X8 f = \ v1 v2 -> case unpackInt64X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt64X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packInt64X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipInt64X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt64X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt64X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt64X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt64X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt64X8 #-}
-- | Fold the elements of a vector to a single value
foldInt64X8 :: (Int64 -> Int64 -> Int64) -> Int64X8 -> Int64
foldInt64X8 f' = \ v -> case unpackInt64X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusInt64X8 #-}
-- | Add two vectors element-wise.
plusInt64X8 :: Int64X8 -> Int64X8 -> Int64X8
plusInt64X8 (Int64X8 m1_1) (Int64X8 m1_2) = Int64X8 (plusInt64X8# m1_1 m1_2)

{-# INLINE minusInt64X8 #-}
-- | Subtract two vectors element-wise.
minusInt64X8 :: Int64X8 -> Int64X8 -> Int64X8
minusInt64X8 (Int64X8 m1_1) (Int64X8 m1_2) = Int64X8 (minusInt64X8# m1_1 m1_2)

{-# INLINE timesInt64X8 #-}
-- | Multiply two vectors element-wise.
timesInt64X8 :: Int64X8 -> Int64X8 -> Int64X8
timesInt64X8 (Int64X8 m1_1) (Int64X8 m1_2) = Int64X8 (timesInt64X8# m1_1 m1_2)

{-# INLINE quotInt64X8 #-}
-- | Rounds towards zero element-wise.
quotInt64X8 :: Int64X8 -> Int64X8 -> Int64X8
quotInt64X8 (Int64X8 m1_1) (Int64X8 m1_2) = Int64X8 (quotInt64X8# m1_1 m1_2)

{-# INLINE remInt64X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt64X8 :: Int64X8 -> Int64X8 -> Int64X8
remInt64X8 (Int64X8 m1_1) (Int64X8 m1_2) = Int64X8 (remInt64X8# m1_1 m1_2)

{-# INLINE negateInt64X8 #-}
-- | Negate element-wise.
negateInt64X8 :: Int64X8 -> Int64X8
negateInt64X8 (Int64X8 m1_1) = Int64X8 (negateInt64X8# m1_1)

{-# INLINE indexInt64X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt64X8Array :: ByteArray -> Int -> Int64X8
indexInt64X8Array (ByteArray a) (I# i) = Int64X8 (indexInt64X8Array# a i)

{-# INLINE readInt64X8Array #-}
-- | Read a vector from specified index of the mutable array.
readInt64X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int64X8
readInt64X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt64X8Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int64X8 m1 #))

{-# INLINE writeInt64X8Array #-}
-- | Write a vector to specified index of mutable array.
writeInt64X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int64X8 -> m ()
writeInt64X8Array (MutableByteArray a) (I# i) (Int64X8 m1) = primitive_ (writeInt64X8Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt64X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt64X8OffAddr :: Addr -> Int -> Int64X8
indexInt64X8OffAddr (Addr a) (I# i) = Int64X8 (indexInt64X8OffAddr# (plusAddr# a (i *# 64#)) 0#)

{-# INLINE readInt64X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt64X8OffAddr :: PrimMonad m => Addr -> Int -> m Int64X8
readInt64X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt64X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int64X8 m1 #))

{-# INLINE writeInt64X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt64X8OffAddr :: PrimMonad m => Addr -> Int -> Int64X8 -> m ()
writeInt64X8OffAddr (Addr a) (I# i) (Int64X8 m1) = primitive_ (writeInt64X8OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1)


