{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

#include "MachDeps.h"

module Data.Primitive.SIMD.Word64X8 (Word64X8) where

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

-- ** Word64X8
data Word64X8 = Word64X8 RealWord64# RealWord64# RealWord64# RealWord64# RealWord64# RealWord64# RealWord64# RealWord64# deriving Typeable

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

instance Eq Word64X8 where
    a == b = case unpackWord64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord64X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Word64X8 where
    a `compare` b = case unpackWord64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord64X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Word64X8 where
    showsPrec _ a s = case unpackWord64X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Word64X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Word64X8 where
    (+) = plusWord64X8
    (-) = minusWord64X8
    (*) = timesWord64X8
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word64X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word64X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word64X8 where
    type Elem Word64X8 = Word64
    type ElemTuple Word64X8 = (Word64, Word64, Word64, Word64, Word64, Word64, Word64, Word64)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 8
    broadcastVector    = broadcastWord64X8
    generateVector     = generateWord64X8
    unsafeInsertVector = unsafeInsertWord64X8
    packVector         = packWord64X8
    unpackVector       = unpackWord64X8
    mapVector          = mapWord64X8
    zipVector          = zipWord64X8
    foldVector         = foldWord64X8

instance SIMDIntVector Word64X8 where
    quotVector = quotWord64X8
    remVector  = remWord64X8

instance Prim Word64X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord64X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord64X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord64X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord64X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord64X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord64X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word64X8 = V_Word64X8 (PV.Vector Word64X8)
newtype instance UV.MVector s Word64X8 = MV_Word64X8 (PMV.MVector s Word64X8)

instance Vector UV.Vector Word64X8 where
    basicUnsafeFreeze (MV_Word64X8 v) = V_Word64X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word64X8 v) = MV_Word64X8 <$> PV.unsafeThaw v
    basicLength (V_Word64X8 v) = PV.length v
    basicUnsafeSlice start len (V_Word64X8 v) = V_Word64X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word64X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word64X8 m) (V_Word64X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word64X8 where
    basicLength (MV_Word64X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word64X8 v) = MV_Word64X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word64X8 v) (MV_Word64X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word64X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word64X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word64X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word64X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word64X8

{-# INLINE broadcastWord64X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord64X8 :: Word64 -> Word64X8
broadcastWord64X8 (W64# x) = case broadcastWord64# x of
    v -> Word64X8 v v v v v v v v

{-# INLINE[1] generateWord64X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord64X8 :: (Int -> Word64) -> Word64X8
generateWord64X8 f = packWord64X8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packWord64X8 #-}
-- | Pack the elements of a tuple into a vector.
packWord64X8 :: (Word64, Word64, Word64, Word64, Word64, Word64, Word64, Word64) -> Word64X8
packWord64X8 (W64# x1, W64# x2, W64# x3, W64# x4, W64# x5, W64# x6, W64# x7, W64# x8) = Word64X8 (packWord64# (# x1 #)) (packWord64# (# x2 #)) (packWord64# (# x3 #)) (packWord64# (# x4 #)) (packWord64# (# x5 #)) (packWord64# (# x6 #)) (packWord64# (# x7 #)) (packWord64# (# x8 #))

{-# INLINE unpackWord64X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord64X8 :: Word64X8 -> (Word64, Word64, Word64, Word64, Word64, Word64, Word64, Word64)
unpackWord64X8 (Word64X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackWord64# m1 of
    (# x1 #) -> case unpackWord64# m2 of
        (# x2 #) -> case unpackWord64# m3 of
            (# x3 #) -> case unpackWord64# m4 of
                (# x4 #) -> case unpackWord64# m5 of
                    (# x5 #) -> case unpackWord64# m6 of
                        (# x6 #) -> case unpackWord64# m7 of
                            (# x7 #) -> case unpackWord64# m8 of
                                (# x8 #) -> (W64# x1, W64# x2, W64# x3, W64# x4, W64# x5, W64# x6, W64# x7, W64# x8)

{-# INLINE unsafeInsertWord64X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord64X8 :: Word64X8 -> Word64 -> Int -> Word64X8
unsafeInsertWord64X8 (Word64X8 m1 m2 m3 m4 m5 m6 m7 m8) (W64# y) _i@(I# ip) | _i < 1 = Word64X8 (insertWord64# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                            | _i < 2 = Word64X8 m1 (insertWord64# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                            | _i < 3 = Word64X8 m1 m2 (insertWord64# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                            | _i < 4 = Word64X8 m1 m2 m3 (insertWord64# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                            | _i < 5 = Word64X8 m1 m2 m3 m4 (insertWord64# m5 y (ip -# 4#)) m6 m7 m8
                                                                            | _i < 6 = Word64X8 m1 m2 m3 m4 m5 (insertWord64# m6 y (ip -# 5#)) m7 m8
                                                                            | _i < 7 = Word64X8 m1 m2 m3 m4 m5 m6 (insertWord64# m7 y (ip -# 6#)) m8
                                                                            | otherwise = Word64X8 m1 m2 m3 m4 m5 m6 m7 (insertWord64# m8 y (ip -# 7#))

{-# INLINE[1] mapWord64X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord64X8 :: (Word64 -> Word64) -> Word64X8 -> Word64X8
mapWord64X8 f = mapWord64X8# (\ x -> case f (W64# x) of { W64# y -> y})

{-# RULES "mapVector abs" mapWord64X8 abs = abs #-}
{-# RULES "mapVector signum" mapWord64X8 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord64X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord64X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord64X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord64X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord64X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord64X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord64X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord64X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord64X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord64X8# #-}
-- | Unboxed helper function.
mapWord64X8# :: (RealWord64# -> RealWord64#) -> Word64X8 -> Word64X8
mapWord64X8# f = \ v -> case unpackWord64X8 v of
    (W64# x1, W64# x2, W64# x3, W64# x4, W64# x5, W64# x6, W64# x7, W64# x8) -> packWord64X8 (W64# (f x1), W64# (f x2), W64# (f x3), W64# (f x4), W64# (f x5), W64# (f x6), W64# (f x7), W64# (f x8))

{-# INLINE[1] zipWord64X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord64X8 :: (Word64 -> Word64 -> Word64) -> Word64X8 -> Word64X8 -> Word64X8
zipWord64X8 f = \ v1 v2 -> case unpackWord64X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord64X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packWord64X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipWord64X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord64X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord64X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord64X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord64X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord64X8 #-}
-- | Fold the elements of a vector to a single value
foldWord64X8 :: (Word64 -> Word64 -> Word64) -> Word64X8 -> Word64
foldWord64X8 f' = \ v -> case unpackWord64X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusWord64X8 #-}
-- | Add two vectors element-wise.
plusWord64X8 :: Word64X8 -> Word64X8 -> Word64X8
plusWord64X8 (Word64X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word64X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word64X8 (plusWord64# m1_1 m1_2) (plusWord64# m2_1 m2_2) (plusWord64# m3_1 m3_2) (plusWord64# m4_1 m4_2) (plusWord64# m5_1 m5_2) (plusWord64# m6_1 m6_2) (plusWord64# m7_1 m7_2) (plusWord64# m8_1 m8_2)

{-# INLINE minusWord64X8 #-}
-- | Subtract two vectors element-wise.
minusWord64X8 :: Word64X8 -> Word64X8 -> Word64X8
minusWord64X8 (Word64X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word64X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word64X8 (minusWord64# m1_1 m1_2) (minusWord64# m2_1 m2_2) (minusWord64# m3_1 m3_2) (minusWord64# m4_1 m4_2) (minusWord64# m5_1 m5_2) (minusWord64# m6_1 m6_2) (minusWord64# m7_1 m7_2) (minusWord64# m8_1 m8_2)

{-# INLINE timesWord64X8 #-}
-- | Multiply two vectors element-wise.
timesWord64X8 :: Word64X8 -> Word64X8 -> Word64X8
timesWord64X8 (Word64X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word64X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word64X8 (timesWord64# m1_1 m1_2) (timesWord64# m2_1 m2_2) (timesWord64# m3_1 m3_2) (timesWord64# m4_1 m4_2) (timesWord64# m5_1 m5_2) (timesWord64# m6_1 m6_2) (timesWord64# m7_1 m7_2) (timesWord64# m8_1 m8_2)

{-# INLINE quotWord64X8 #-}
-- | Rounds towards zero element-wise.
quotWord64X8 :: Word64X8 -> Word64X8 -> Word64X8
quotWord64X8 (Word64X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word64X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word64X8 (quotWord64# m1_1 m1_2) (quotWord64# m2_1 m2_2) (quotWord64# m3_1 m3_2) (quotWord64# m4_1 m4_2) (quotWord64# m5_1 m5_2) (quotWord64# m6_1 m6_2) (quotWord64# m7_1 m7_2) (quotWord64# m8_1 m8_2)

{-# INLINE remWord64X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord64X8 :: Word64X8 -> Word64X8 -> Word64X8
remWord64X8 (Word64X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word64X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word64X8 (remWord64# m1_1 m1_2) (remWord64# m2_1 m2_2) (remWord64# m3_1 m3_2) (remWord64# m4_1 m4_2) (remWord64# m5_1 m5_2) (remWord64# m6_1 m6_2) (remWord64# m7_1 m7_2) (remWord64# m8_1 m8_2)

{-# INLINE indexWord64X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord64X8Array :: ByteArray -> Int -> Word64X8
indexWord64X8Array (ByteArray a) (I# i) = Word64X8 (indexWord64Array# a ((i *# 8#) +# 0#)) (indexWord64Array# a ((i *# 8#) +# 1#)) (indexWord64Array# a ((i *# 8#) +# 2#)) (indexWord64Array# a ((i *# 8#) +# 3#)) (indexWord64Array# a ((i *# 8#) +# 4#)) (indexWord64Array# a ((i *# 8#) +# 5#)) (indexWord64Array# a ((i *# 8#) +# 6#)) (indexWord64Array# a ((i *# 8#) +# 7#))

{-# INLINE readWord64X8Array #-}
-- | Read a vector from specified index of the mutable array.
readWord64X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word64X8
readWord64X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord64Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord64Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord64Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord64Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord64Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord64Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord64Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord64Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Word64X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord64X8Array #-}
-- | Write a vector to specified index of mutable array.
writeWord64X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word64X8 -> m ()
writeWord64X8Array (MutableByteArray a) (I# i) (Word64X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord64Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeWord64Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexWord64X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord64X8OffAddr :: Addr -> Int -> Word64X8
indexWord64X8OffAddr (Addr a) (I# i) = Word64X8 (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexWord64OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#)

{-# INLINE readWord64X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord64X8OffAddr :: PrimMonad m => Addr -> Int -> m Word64X8
readWord64X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord64OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s7 of
                                (# s8, m8 #) -> (# s8, Word64X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord64X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord64X8OffAddr :: PrimMonad m => Addr -> Int -> Word64X8 -> m ()
writeWord64X8OffAddr (Addr a) (I# i) (Word64X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m2) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m3) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m4) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m5) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m6) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m7) >> primitive_ (writeWord64OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m8)


