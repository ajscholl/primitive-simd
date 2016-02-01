{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

#include "MachDeps.h"

module Data.Primitive.SIMD.Word64X2 (Word64X2) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

import GHC.Word
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
type RealWord64# = Word#
#elif WORD_SIZE_IN_BITS == 32
type RealWord64# = Word64#
#else
#error "WORD_SIZE_IN_BITS is neither 64 or 32"
#endif

-- ** Word64X2
data Word64X2 = Word64X2 RealWord64# RealWord64# deriving Typeable

broadcastWord64# :: RealWord64# -> RealWord64#
broadcastWord64# v = v

packWord64# :: (# RealWord64# #) -> RealWord64#
packWord64# (# v #) = v

unpackWord64# :: RealWord64# -> (# RealWord64# #)
unpackWord64# v = (# v #)

insertWord64# :: RealWord64# -> RealWord64# -> Int# -> RealWord64#
insertWord64# _ v _ = v

plusWord64# :: RealWord64# -> RealWord64# -> RealWord64#
plusWord64# a b = case W64# a + W64# b of W64# c -> c

minusWord64# :: RealWord64# -> RealWord64# -> RealWord64#
minusWord64# a b = case W64# a - W64# b of W64# c -> c

timesWord64# :: RealWord64# -> RealWord64# -> RealWord64#
timesWord64# a b = case W64# a * W64# b of W64# c -> c

quotWord64# :: RealWord64# -> RealWord64# -> RealWord64#
quotWord64# a b = case W64# a `quot` W64# b of W64# c -> c

remWord64# :: RealWord64# -> RealWord64# -> RealWord64#
remWord64# a b = case W64# a `rem` W64# b of W64# c -> c

abs' :: Word64 -> Word64
abs' (W64# x) = W64# (abs# x)

{-# INLINE abs# #-}
abs# :: RealWord64# -> RealWord64#
abs# x = case abs (W64# x) of
    W64# y -> y

signum' :: Word64 -> Word64
signum' (W64# x) = W64# (signum# x)

{-# NOINLINE signum# #-}
signum# :: RealWord64# -> RealWord64#
signum# x = case signum (W64# x) of
    W64# y -> y

instance Eq Word64X2 where
    a == b = case unpackWord64X2 a of
        (x1, x2) -> case unpackWord64X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Word64X2 where
    a `compare` b = case unpackWord64X2 a of
        (x1, x2) -> case unpackWord64X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Word64X2 where
    showsPrec _ a s = case unpackWord64X2 a of
        (x1, x2) -> "Word64X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Word64X2 where
    (+) = plusWord64X2
    (-) = minusWord64X2
    (*) = timesWord64X2
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word64X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word64X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word64X2 where
    type Elem Word64X2 = Word64
    type ElemTuple Word64X2 = (Word64, Word64)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 8
    broadcastVector    = broadcastWord64X2
    unsafeInsertVector = unsafeInsertWord64X2
    packVector         = packWord64X2
    unpackVector       = unpackWord64X2
    mapVector          = mapWord64X2
    zipVector          = zipWord64X2
    foldVector         = foldWord64X2

instance SIMDIntVector Word64X2 where
    quotVector = quotWord64X2
    remVector  = remWord64X2

instance Prim Word64X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord64X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord64X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord64X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord64X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord64X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord64X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word64X2 = V_Word64X2 (PV.Vector Word64X2)
newtype instance UV.MVector s Word64X2 = MV_Word64X2 (PMV.MVector s Word64X2)

instance Vector UV.Vector Word64X2 where
    basicUnsafeFreeze (MV_Word64X2 v) = V_Word64X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word64X2 v) = MV_Word64X2 <$> PV.unsafeThaw v
    basicLength (V_Word64X2 v) = PV.length v
    basicUnsafeSlice start len (V_Word64X2 v) = V_Word64X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word64X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word64X2 m) (V_Word64X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word64X2 where
    basicLength (MV_Word64X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word64X2 v) = MV_Word64X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word64X2 v) (MV_Word64X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word64X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word64X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word64X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word64X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word64X2

{-# INLINE broadcastWord64X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord64X2 :: Word64 -> Word64X2
broadcastWord64X2 (W64# x) = case broadcastWord64# x of
    v -> Word64X2 v v

{-# INLINE packWord64X2 #-}
-- | Pack the elements of a tuple into a vector.
packWord64X2 :: (Word64, Word64) -> Word64X2
packWord64X2 (W64# x1, W64# x2) = Word64X2 (packWord64# (# x1 #)) (packWord64# (# x2 #))

{-# INLINE unpackWord64X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord64X2 :: Word64X2 -> (Word64, Word64)
unpackWord64X2 (Word64X2 m1 m2) = case unpackWord64# m1 of
    (# x1 #) -> case unpackWord64# m2 of
        (# x2 #) -> (W64# x1, W64# x2)

{-# INLINE unsafeInsertWord64X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord64X2 :: Word64X2 -> Word64 -> Int -> Word64X2
unsafeInsertWord64X2 (Word64X2 m1 m2) (W64# y) _i@(I# ip) | _i < 1 = Word64X2 (insertWord64# m1 y (ip -# 0#)) m2
                                                          | otherwise = Word64X2 m1 (insertWord64# m2 y (ip -# 1#))

{-# INLINE[1] mapWord64X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord64X2 :: (Word64 -> Word64) -> Word64X2 -> Word64X2
mapWord64X2 f = mapWord64X2# (\ x -> case f (W64# x) of { W64# y -> y})

{-# RULES "mapVector abs" mapWord64X2 abs = abs #-}
{-# RULES "mapVector signum" mapWord64X2 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord64X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord64X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord64X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord64X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord64X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord64X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord64X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord64X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord64X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord64X2# #-}
-- | Unboxed helper function.
mapWord64X2# :: (RealWord64# -> RealWord64#) -> Word64X2 -> Word64X2
mapWord64X2# f = \ v -> case unpackWord64X2 v of
    (W64# x1, W64# x2) -> packWord64X2 (W64# (f x1), W64# (f x2))

{-# INLINE[1] zipWord64X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord64X2 :: (Word64 -> Word64 -> Word64) -> Word64X2 -> Word64X2 -> Word64X2
zipWord64X2 f = \ v1 v2 -> case unpackWord64X2 v1 of
    (x1, x2) -> case unpackWord64X2 v2 of
        (y1, y2) -> packWord64X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipWord64X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord64X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord64X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord64X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord64X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord64X2 #-}
-- | Fold the elements of a vector to a single value
foldWord64X2 :: (Word64 -> Word64 -> Word64) -> Word64X2 -> Word64
foldWord64X2 f' = \ v -> case unpackWord64X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusWord64X2 #-}
-- | Add two vectors element-wise.
plusWord64X2 :: Word64X2 -> Word64X2 -> Word64X2
plusWord64X2 (Word64X2 m1_1 m2_1) (Word64X2 m1_2 m2_2) = Word64X2 (plusWord64# m1_1 m1_2) (plusWord64# m2_1 m2_2)

{-# INLINE minusWord64X2 #-}
-- | Subtract two vectors element-wise.
minusWord64X2 :: Word64X2 -> Word64X2 -> Word64X2
minusWord64X2 (Word64X2 m1_1 m2_1) (Word64X2 m1_2 m2_2) = Word64X2 (minusWord64# m1_1 m1_2) (minusWord64# m2_1 m2_2)

{-# INLINE timesWord64X2 #-}
-- | Multiply two vectors element-wise.
timesWord64X2 :: Word64X2 -> Word64X2 -> Word64X2
timesWord64X2 (Word64X2 m1_1 m2_1) (Word64X2 m1_2 m2_2) = Word64X2 (timesWord64# m1_1 m1_2) (timesWord64# m2_1 m2_2)

{-# INLINE quotWord64X2 #-}
-- | Rounds towards zero element-wise.
quotWord64X2 :: Word64X2 -> Word64X2 -> Word64X2
quotWord64X2 (Word64X2 m1_1 m2_1) (Word64X2 m1_2 m2_2) = Word64X2 (quotWord64# m1_1 m1_2) (quotWord64# m2_1 m2_2)

{-# INLINE remWord64X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord64X2 :: Word64X2 -> Word64X2 -> Word64X2
remWord64X2 (Word64X2 m1_1 m2_1) (Word64X2 m1_2 m2_2) = Word64X2 (remWord64# m1_1 m1_2) (remWord64# m2_1 m2_2)

{-# INLINE indexWord64X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord64X2Array :: ByteArray -> Int -> Word64X2
indexWord64X2Array (ByteArray a) (I# i) = Word64X2 (indexWord64Array# a ((i *# 2#) +# 0#)) (indexWord64Array# a ((i *# 2#) +# 1#))

{-# INLINE readWord64X2Array #-}
-- | Read a vector from specified index of the mutable array.
readWord64X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word64X2
readWord64X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord64Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord64Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Word64X2 m1 m2 #))

{-# INLINE writeWord64X2Array #-}
-- | Write a vector to specified index of mutable array.
writeWord64X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word64X2 -> m ()
writeWord64X2Array (MutableByteArray a) (I# i) (Word64X2 m1 m2) = primitive_ (writeWord64Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeWord64Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexWord64X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord64X2OffAddr :: Addr -> Int -> Word64X2
indexWord64X2OffAddr (Addr a) (I# i) = Word64X2 (indexWord64OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#)

{-# INLINE readWord64X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord64X2OffAddr :: PrimMonad m => Addr -> Int -> m Word64X2
readWord64X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s1 of
        (# s2, m2 #) -> (# s2, Word64X2 m1 m2 #))

{-# INLINE writeWord64X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord64X2OffAddr :: PrimMonad m => Addr -> Int -> Word64X2 -> m ()
writeWord64X2OffAddr (Addr a) (I# i) (Word64X2 m1 m2) = primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m2)


