{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

#include "MachDeps.h"

module Data.Primitive.SIMD.Int64X4 (Int64X4) where

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

-- ** Int64X4
data Int64X4 = Int64X4 RealInt64# RealInt64# RealInt64# RealInt64# deriving Typeable

broadcastInt64# :: RealInt64# -> RealInt64#
broadcastInt64# v = v

packInt64# :: (# RealInt64# #) -> RealInt64#
packInt64# (# v #) = v

unpackInt64# :: RealInt64# -> (# RealInt64# #)
unpackInt64# v = (# v #)

insertInt64# :: RealInt64# -> RealInt64# -> Int# -> RealInt64#
insertInt64# _ v _ = v

negateInt64# :: RealInt64# -> RealInt64#
negateInt64# a = case negate (I64# a) of I64# b -> b

plusInt64# :: RealInt64# -> RealInt64# -> RealInt64#
plusInt64# a b = case I64# a + I64# b of I64# c -> c

minusInt64# :: RealInt64# -> RealInt64# -> RealInt64#
minusInt64# a b = case I64# a - I64# b of I64# c -> c

timesInt64# :: RealInt64# -> RealInt64# -> RealInt64#
timesInt64# a b = case I64# a * I64# b of I64# c -> c

quotInt64# :: RealInt64# -> RealInt64# -> RealInt64#
quotInt64# a b = case I64# a `quot` I64# b of I64# c -> c

remInt64# :: RealInt64# -> RealInt64# -> RealInt64#
remInt64# a b = case I64# a `rem` I64# b of I64# c -> c

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

instance Eq Int64X4 where
    a == b = case unpackInt64X4 a of
        (x1, x2, x3, x4) -> case unpackInt64X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Int64X4 where
    a `compare` b = case unpackInt64X4 a of
        (x1, x2, x3, x4) -> case unpackInt64X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Int64X4 where
    showsPrec _ a s = case unpackInt64X4 a of
        (x1, x2, x3, x4) -> "Int64X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Int64X4 where
    (+) = plusInt64X4
    (-) = minusInt64X4
    (*) = timesInt64X4
    negate = negateInt64X4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int64X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int64X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int64X4 where
    type Elem Int64X4 = Int64
    type ElemTuple Int64X4 = (Int64, Int64, Int64, Int64)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 8
    broadcastVector    = broadcastInt64X4
    unsafeInsertVector = unsafeInsertInt64X4
    packVector         = packInt64X4
    unpackVector       = unpackInt64X4
    mapVector          = mapInt64X4
    zipVector          = zipInt64X4
    foldVector         = foldInt64X4

instance SIMDIntVector Int64X4 where
    quotVector = quotInt64X4
    remVector  = remInt64X4

instance Prim Int64X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt64X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt64X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt64X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt64X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt64X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt64X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int64X4 = V_Int64X4 (PV.Vector Int64X4)
newtype instance UV.MVector s Int64X4 = MV_Int64X4 (PMV.MVector s Int64X4)

instance Vector UV.Vector Int64X4 where
    basicUnsafeFreeze (MV_Int64X4 v) = V_Int64X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int64X4 v) = MV_Int64X4 <$> PV.unsafeThaw v
    basicLength (V_Int64X4 v) = PV.length v
    basicUnsafeSlice start len (V_Int64X4 v) = V_Int64X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int64X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int64X4 m) (V_Int64X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int64X4 where
    basicLength (MV_Int64X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int64X4 v) = MV_Int64X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int64X4 v) (MV_Int64X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int64X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int64X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int64X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int64X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int64X4

{-# INLINE broadcastInt64X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt64X4 :: Int64 -> Int64X4
broadcastInt64X4 (I64# x) = case broadcastInt64# x of
    v -> Int64X4 v v v v

{-# INLINE packInt64X4 #-}
-- | Pack the elements of a tuple into a vector.
packInt64X4 :: (Int64, Int64, Int64, Int64) -> Int64X4
packInt64X4 (I64# x1, I64# x2, I64# x3, I64# x4) = Int64X4 (packInt64# (# x1 #)) (packInt64# (# x2 #)) (packInt64# (# x3 #)) (packInt64# (# x4 #))

{-# INLINE unpackInt64X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt64X4 :: Int64X4 -> (Int64, Int64, Int64, Int64)
unpackInt64X4 (Int64X4 m1 m2 m3 m4) = case unpackInt64# m1 of
    (# x1 #) -> case unpackInt64# m2 of
        (# x2 #) -> case unpackInt64# m3 of
            (# x3 #) -> case unpackInt64# m4 of
                (# x4 #) -> (I64# x1, I64# x2, I64# x3, I64# x4)

{-# INLINE unsafeInsertInt64X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt64X4 :: Int64X4 -> Int64 -> Int -> Int64X4
unsafeInsertInt64X4 (Int64X4 m1 m2 m3 m4) (I64# y) _i@(I# ip) | _i < 1 = Int64X4 (insertInt64# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 2 = Int64X4 m1 (insertInt64# m2 y (ip -# 1#)) m3 m4
                                                              | _i < 3 = Int64X4 m1 m2 (insertInt64# m3 y (ip -# 2#)) m4
                                                              | otherwise = Int64X4 m1 m2 m3 (insertInt64# m4 y (ip -# 3#))

{-# INLINE[1] mapInt64X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt64X4 :: (Int64 -> Int64) -> Int64X4 -> Int64X4
mapInt64X4 f = mapInt64X4# (\ x -> case f (I64# x) of { I64# y -> y})

{-# RULES "mapVector abs" mapInt64X4 abs = abs #-}
{-# RULES "mapVector signum" mapInt64X4 signum = signum #-}
{-# RULES "mapVector negate" mapInt64X4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt64X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt64X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt64X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt64X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt64X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt64X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt64X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt64X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt64X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt64X4# #-}
-- | Unboxed helper function.
mapInt64X4# :: (RealInt64# -> RealInt64#) -> Int64X4 -> Int64X4
mapInt64X4# f = \ v -> case unpackInt64X4 v of
    (I64# x1, I64# x2, I64# x3, I64# x4) -> packInt64X4 (I64# (f x1), I64# (f x2), I64# (f x3), I64# (f x4))

{-# INLINE[1] zipInt64X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt64X4 :: (Int64 -> Int64 -> Int64) -> Int64X4 -> Int64X4 -> Int64X4
zipInt64X4 f = \ v1 v2 -> case unpackInt64X4 v1 of
    (x1, x2, x3, x4) -> case unpackInt64X4 v2 of
        (y1, y2, y3, y4) -> packInt64X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipInt64X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt64X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt64X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt64X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt64X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt64X4 #-}
-- | Fold the elements of a vector to a single value
foldInt64X4 :: (Int64 -> Int64 -> Int64) -> Int64X4 -> Int64
foldInt64X4 f' = \ v -> case unpackInt64X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusInt64X4 #-}
-- | Add two vectors element-wise.
plusInt64X4 :: Int64X4 -> Int64X4 -> Int64X4
plusInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) (Int64X4 m1_2 m2_2 m3_2 m4_2) = Int64X4 (plusInt64# m1_1 m1_2) (plusInt64# m2_1 m2_2) (plusInt64# m3_1 m3_2) (plusInt64# m4_1 m4_2)

{-# INLINE minusInt64X4 #-}
-- | Subtract two vectors element-wise.
minusInt64X4 :: Int64X4 -> Int64X4 -> Int64X4
minusInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) (Int64X4 m1_2 m2_2 m3_2 m4_2) = Int64X4 (minusInt64# m1_1 m1_2) (minusInt64# m2_1 m2_2) (minusInt64# m3_1 m3_2) (minusInt64# m4_1 m4_2)

{-# INLINE timesInt64X4 #-}
-- | Multiply two vectors element-wise.
timesInt64X4 :: Int64X4 -> Int64X4 -> Int64X4
timesInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) (Int64X4 m1_2 m2_2 m3_2 m4_2) = Int64X4 (timesInt64# m1_1 m1_2) (timesInt64# m2_1 m2_2) (timesInt64# m3_1 m3_2) (timesInt64# m4_1 m4_2)

{-# INLINE quotInt64X4 #-}
-- | Rounds towards zero element-wise.
quotInt64X4 :: Int64X4 -> Int64X4 -> Int64X4
quotInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) (Int64X4 m1_2 m2_2 m3_2 m4_2) = Int64X4 (quotInt64# m1_1 m1_2) (quotInt64# m2_1 m2_2) (quotInt64# m3_1 m3_2) (quotInt64# m4_1 m4_2)

{-# INLINE remInt64X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt64X4 :: Int64X4 -> Int64X4 -> Int64X4
remInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) (Int64X4 m1_2 m2_2 m3_2 m4_2) = Int64X4 (remInt64# m1_1 m1_2) (remInt64# m2_1 m2_2) (remInt64# m3_1 m3_2) (remInt64# m4_1 m4_2)

{-# INLINE negateInt64X4 #-}
-- | Negate element-wise.
negateInt64X4 :: Int64X4 -> Int64X4
negateInt64X4 (Int64X4 m1_1 m2_1 m3_1 m4_1) = Int64X4 (negateInt64# m1_1) (negateInt64# m2_1) (negateInt64# m3_1) (negateInt64# m4_1)

{-# INLINE indexInt64X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt64X4Array :: ByteArray -> Int -> Int64X4
indexInt64X4Array (ByteArray a) (I# i) = Int64X4 (indexInt64Array# a ((i *# 4#) +# 0#)) (indexInt64Array# a ((i *# 4#) +# 1#)) (indexInt64Array# a ((i *# 4#) +# 2#)) (indexInt64Array# a ((i *# 4#) +# 3#))

{-# INLINE readInt64X4Array #-}
-- | Read a vector from specified index of the mutable array.
readInt64X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int64X4
readInt64X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt64Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt64Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt64Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt64Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Int64X4 m1 m2 m3 m4 #))

{-# INLINE writeInt64X4Array #-}
-- | Write a vector to specified index of mutable array.
writeInt64X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int64X4 -> m ()
writeInt64X4Array (MutableByteArray a) (I# i) (Int64X4 m1 m2 m3 m4) = primitive_ (writeInt64Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeInt64Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeInt64Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeInt64Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexInt64X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt64X4OffAddr :: Addr -> Int -> Int64X4
indexInt64X4OffAddr (Addr a) (I# i) = Int64X4 (indexInt64OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexInt64OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexInt64OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexInt64OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#)

{-# INLINE readInt64X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt64X4OffAddr :: PrimMonad m => Addr -> Int -> m Int64X4
readInt64X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt64OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt64OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt64OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt64OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s3 of
                (# s4, m4 #) -> (# s4, Int64X4 m1 m2 m3 m4 #))

{-# INLINE writeInt64X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt64X4OffAddr :: PrimMonad m => Addr -> Int -> Int64X4 -> m ()
writeInt64X4OffAddr (Addr a) (I# i) (Int64X4 m1 m2 m3 m4) = primitive_ (writeInt64OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeInt64OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m2) >> primitive_ (writeInt64OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m3) >> primitive_ (writeInt64OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m4)


