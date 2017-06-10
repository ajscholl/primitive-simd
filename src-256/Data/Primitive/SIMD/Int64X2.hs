{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

#include "MachDeps.h"

module Data.Primitive.SIMD.Int64X2 (Int64X2) where

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

-- ** Int64X2
data Int64X2 = Int64X2 Int64X2# deriving Typeable

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

instance Eq Int64X2 where
    a == b = case unpackInt64X2 a of
        (x1, x2) -> case unpackInt64X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Int64X2 where
    a `compare` b = case unpackInt64X2 a of
        (x1, x2) -> case unpackInt64X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Int64X2 where
    showsPrec _ a s = case unpackInt64X2 a of
        (x1, x2) -> "Int64X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Int64X2 where
    (+) = plusInt64X2
    (-) = minusInt64X2
    (*) = timesInt64X2
    negate = negateInt64X2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int64X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int64X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int64X2 where
    type Elem Int64X2 = Int64
    type ElemTuple Int64X2 = (Int64, Int64)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 8
    broadcastVector    = broadcastInt64X2
    generateVector     = generateInt64X2
    unsafeInsertVector = unsafeInsertInt64X2
    packVector         = packInt64X2
    unpackVector       = unpackInt64X2
    mapVector          = mapInt64X2
    zipVector          = zipInt64X2
    foldVector         = foldInt64X2

instance SIMDIntVector Int64X2 where
    quotVector = quotInt64X2
    remVector  = remInt64X2

instance Prim Int64X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt64X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt64X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt64X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt64X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt64X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt64X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int64X2 = V_Int64X2 (PV.Vector Int64X2)
newtype instance UV.MVector s Int64X2 = MV_Int64X2 (PMV.MVector s Int64X2)

instance Vector UV.Vector Int64X2 where
    basicUnsafeFreeze (MV_Int64X2 v) = V_Int64X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int64X2 v) = MV_Int64X2 <$> PV.unsafeThaw v
    basicLength (V_Int64X2 v) = PV.length v
    basicUnsafeSlice start len (V_Int64X2 v) = V_Int64X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int64X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int64X2 m) (V_Int64X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int64X2 where
    basicLength (MV_Int64X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int64X2 v) = MV_Int64X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int64X2 v) (MV_Int64X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int64X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int64X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int64X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int64X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int64X2

{-# INLINE broadcastInt64X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt64X2 :: Int64 -> Int64X2
broadcastInt64X2 (I64# x) = Int64X2 (broadcastInt64X2# x)

{-# INLINE[1] generateInt64X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt64X2 :: (Int -> Int64) -> Int64X2
generateInt64X2 f = packInt64X2 (f 0, f 1)

{-# INLINE packInt64X2 #-}
-- | Pack the elements of a tuple into a vector.
packInt64X2 :: (Int64, Int64) -> Int64X2
packInt64X2 (I64# x1, I64# x2) = Int64X2 (packInt64X2# (# x1, x2 #))

{-# INLINE unpackInt64X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt64X2 :: Int64X2 -> (Int64, Int64)
unpackInt64X2 (Int64X2 m1) = case unpackInt64X2# m1 of
    (# x1, x2 #) -> (I64# x1, I64# x2)

{-# INLINE unsafeInsertInt64X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt64X2 :: Int64X2 -> Int64 -> Int -> Int64X2
unsafeInsertInt64X2 (Int64X2 m1) (I64# y) _i@(I# ip) = Int64X2 (insertInt64X2# m1 y (ip -# 0#))

{-# INLINE[1] mapInt64X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt64X2 :: (Int64 -> Int64) -> Int64X2 -> Int64X2
mapInt64X2 f = mapInt64X2# (\ x -> case f (I64# x) of { I64# y -> y})

{-# RULES "mapVector abs" mapInt64X2 abs = abs #-}
{-# RULES "mapVector signum" mapInt64X2 signum = signum #-}
{-# RULES "mapVector negate" mapInt64X2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt64X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt64X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt64X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt64X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt64X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt64X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt64X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt64X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt64X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt64X2# #-}
-- | Unboxed helper function.
mapInt64X2# :: (RealInt64# -> RealInt64#) -> Int64X2 -> Int64X2
mapInt64X2# f = \ v -> case unpackInt64X2 v of
    (I64# x1, I64# x2) -> packInt64X2 (I64# (f x1), I64# (f x2))

{-# INLINE[1] zipInt64X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt64X2 :: (Int64 -> Int64 -> Int64) -> Int64X2 -> Int64X2 -> Int64X2
zipInt64X2 f = \ v1 v2 -> case unpackInt64X2 v1 of
    (x1, x2) -> case unpackInt64X2 v2 of
        (y1, y2) -> packInt64X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipInt64X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt64X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt64X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt64X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt64X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt64X2 #-}
-- | Fold the elements of a vector to a single value
foldInt64X2 :: (Int64 -> Int64 -> Int64) -> Int64X2 -> Int64
foldInt64X2 f' = \ v -> case unpackInt64X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusInt64X2 #-}
-- | Add two vectors element-wise.
plusInt64X2 :: Int64X2 -> Int64X2 -> Int64X2
plusInt64X2 (Int64X2 m1_1) (Int64X2 m1_2) = Int64X2 (plusInt64X2# m1_1 m1_2)

{-# INLINE minusInt64X2 #-}
-- | Subtract two vectors element-wise.
minusInt64X2 :: Int64X2 -> Int64X2 -> Int64X2
minusInt64X2 (Int64X2 m1_1) (Int64X2 m1_2) = Int64X2 (minusInt64X2# m1_1 m1_2)

{-# INLINE timesInt64X2 #-}
-- | Multiply two vectors element-wise.
timesInt64X2 :: Int64X2 -> Int64X2 -> Int64X2
timesInt64X2 (Int64X2 m1_1) (Int64X2 m1_2) = Int64X2 (timesInt64X2# m1_1 m1_2)

{-# INLINE quotInt64X2 #-}
-- | Rounds towards zero element-wise.
quotInt64X2 :: Int64X2 -> Int64X2 -> Int64X2
quotInt64X2 (Int64X2 m1_1) (Int64X2 m1_2) = Int64X2 (quotInt64X2# m1_1 m1_2)

{-# INLINE remInt64X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt64X2 :: Int64X2 -> Int64X2 -> Int64X2
remInt64X2 (Int64X2 m1_1) (Int64X2 m1_2) = Int64X2 (remInt64X2# m1_1 m1_2)

{-# INLINE negateInt64X2 #-}
-- | Negate element-wise.
negateInt64X2 :: Int64X2 -> Int64X2
negateInt64X2 (Int64X2 m1_1) = Int64X2 (negateInt64X2# m1_1)

{-# INLINE indexInt64X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt64X2Array :: ByteArray -> Int -> Int64X2
indexInt64X2Array (ByteArray a) (I# i) = Int64X2 (indexInt64X2Array# a i)

{-# INLINE readInt64X2Array #-}
-- | Read a vector from specified index of the mutable array.
readInt64X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int64X2
readInt64X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt64X2Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int64X2 m1 #))

{-# INLINE writeInt64X2Array #-}
-- | Write a vector to specified index of mutable array.
writeInt64X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int64X2 -> m ()
writeInt64X2Array (MutableByteArray a) (I# i) (Int64X2 m1) = primitive_ (writeInt64X2Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt64X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt64X2OffAddr :: Addr -> Int -> Int64X2
indexInt64X2OffAddr (Addr a) (I# i) = Int64X2 (indexInt64X2OffAddr# (plusAddr# a (i *# 16#)) 0#)

{-# INLINE readInt64X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt64X2OffAddr :: PrimMonad m => Addr -> Int -> m Int64X2
readInt64X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt64X2OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int64X2 m1 #))

{-# INLINE writeInt64X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt64X2OffAddr :: PrimMonad m => Addr -> Int -> Int64X2 -> m ()
writeInt64X2OffAddr (Addr a) (I# i) (Int64X2 m1) = primitive_ (writeInt64X2OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1)


