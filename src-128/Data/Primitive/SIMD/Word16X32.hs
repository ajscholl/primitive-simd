{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word16X32 (Word16X32) where

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

-- ** Word16X32
data Word16X32 = Word16X32 Word16X8# Word16X8# Word16X8# Word16X8# deriving Typeable

abs' :: Word16 -> Word16
abs' (W16# x) = W16# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W16# x) of
    W16# y -> y

signum' :: Word16 -> Word16
signum' (W16# x) = W16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W16# x) of
    W16# y -> y

instance Eq Word16X32 where
    a == b = case unpackWord16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord16X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Word16X32 where
    a `compare` b = case unpackWord16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord16X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Word16X32 where
    showsPrec _ a s = case unpackWord16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Word16X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Word16X32 where
    (+) = plusWord16X32
    (-) = minusWord16X32
    (*) = timesWord16X32
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word16X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word16X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word16X32 where
    type Elem Word16X32 = Word16
    type ElemTuple Word16X32 = (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 2
    broadcastVector    = broadcastWord16X32
    generateVector     = generateWord16X32
    unsafeInsertVector = unsafeInsertWord16X32
    packVector         = packWord16X32
    unpackVector       = unpackWord16X32
    mapVector          = mapWord16X32
    zipVector          = zipWord16X32
    foldVector         = foldWord16X32
    sumVector          = sumWord16X32

instance SIMDIntVector Word16X32 where
    quotVector = quotWord16X32
    remVector  = remWord16X32

instance Prim Word16X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord16X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord16X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord16X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord16X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord16X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord16X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word16X32 = V_Word16X32 (PV.Vector Word16X32)
newtype instance UV.MVector s Word16X32 = MV_Word16X32 (PMV.MVector s Word16X32)

instance Vector UV.Vector Word16X32 where
    basicUnsafeFreeze (MV_Word16X32 v) = V_Word16X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word16X32 v) = MV_Word16X32 <$> PV.unsafeThaw v
    basicLength (V_Word16X32 v) = PV.length v
    basicUnsafeSlice start len (V_Word16X32 v) = V_Word16X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word16X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word16X32 m) (V_Word16X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word16X32 where
    basicLength (MV_Word16X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word16X32 v) = MV_Word16X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word16X32 v) (MV_Word16X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word16X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word16X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word16X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word16X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word16X32

{-# INLINE broadcastWord16X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord16X32 :: Word16 -> Word16X32
broadcastWord16X32 (W16# x) = case broadcastWord16X8# x of
    v -> Word16X32 v v v v

{-# INLINE[1] generateWord16X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord16X32 :: (Int -> Word16) -> Word16X32
generateWord16X32 f = packWord16X32 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15, f 16, f 17, f 18, f 19, f 20, f 21, f 22, f 23, f 24, f 25, f 26, f 27, f 28, f 29, f 30, f 31)

{-# INLINE packWord16X32 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X32 :: (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16) -> Word16X32
packWord16X32 (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16, W16# x17, W16# x18, W16# x19, W16# x20, W16# x21, W16# x22, W16# x23, W16# x24, W16# x25, W16# x26, W16# x27, W16# x28, W16# x29, W16# x30, W16# x31, W16# x32) = Word16X32 (packWord16X8# (# x1, x2, x3, x4, x5, x6, x7, x8 #)) (packWord16X8# (# x9, x10, x11, x12, x13, x14, x15, x16 #)) (packWord16X8# (# x17, x18, x19, x20, x21, x22, x23, x24 #)) (packWord16X8# (# x25, x26, x27, x28, x29, x30, x31, x32 #))

{-# INLINE unpackWord16X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X32 :: Word16X32 -> (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
unpackWord16X32 (Word16X32 m1 m2 m3 m4) = case unpackWord16X8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> case unpackWord16X8# m2 of
        (# x9, x10, x11, x12, x13, x14, x15, x16 #) -> case unpackWord16X8# m3 of
            (# x17, x18, x19, x20, x21, x22, x23, x24 #) -> case unpackWord16X8# m4 of
                (# x25, x26, x27, x28, x29, x30, x31, x32 #) -> (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16, W16# x17, W16# x18, W16# x19, W16# x20, W16# x21, W16# x22, W16# x23, W16# x24, W16# x25, W16# x26, W16# x27, W16# x28, W16# x29, W16# x30, W16# x31, W16# x32)

{-# INLINE unsafeInsertWord16X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X32 :: Word16X32 -> Word16 -> Int -> Word16X32
unsafeInsertWord16X32 (Word16X32 m1 m2 m3 m4) (W16# y) _i@(I# ip) | _i < 8 = Word16X32 (insertWord16X8# m1 y (ip -# 0#)) m2 m3 m4
                                                                  | _i < 16 = Word16X32 m1 (insertWord16X8# m2 y (ip -# 8#)) m3 m4
                                                                  | _i < 24 = Word16X32 m1 m2 (insertWord16X8# m3 y (ip -# 16#)) m4
                                                                  | otherwise = Word16X32 m1 m2 m3 (insertWord16X8# m4 y (ip -# 24#))

{-# INLINE[1] mapWord16X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord16X32 :: (Word16 -> Word16) -> Word16X32 -> Word16X32
mapWord16X32 f = mapWord16X32# (\ x -> case f (W16# x) of { W16# y -> y})

{-# RULES "mapVector abs" mapWord16X32 abs = abs #-}
{-# RULES "mapVector signum" mapWord16X32 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord16X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord16X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord16X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord16X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord16X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord16X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord16X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord16X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord16X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord16X32# #-}
-- | Unboxed helper function.
mapWord16X32# :: (Word# -> Word#) -> Word16X32 -> Word16X32
mapWord16X32# f = \ v -> case unpackWord16X32 v of
    (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16, W16# x17, W16# x18, W16# x19, W16# x20, W16# x21, W16# x22, W16# x23, W16# x24, W16# x25, W16# x26, W16# x27, W16# x28, W16# x29, W16# x30, W16# x31, W16# x32) -> packWord16X32 (W16# (f x1), W16# (f x2), W16# (f x3), W16# (f x4), W16# (f x5), W16# (f x6), W16# (f x7), W16# (f x8), W16# (f x9), W16# (f x10), W16# (f x11), W16# (f x12), W16# (f x13), W16# (f x14), W16# (f x15), W16# (f x16), W16# (f x17), W16# (f x18), W16# (f x19), W16# (f x20), W16# (f x21), W16# (f x22), W16# (f x23), W16# (f x24), W16# (f x25), W16# (f x26), W16# (f x27), W16# (f x28), W16# (f x29), W16# (f x30), W16# (f x31), W16# (f x32))

{-# INLINE[1] zipWord16X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord16X32 :: (Word16 -> Word16 -> Word16) -> Word16X32 -> Word16X32 -> Word16X32
zipWord16X32 f = \ v1 v2 -> case unpackWord16X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord16X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packWord16X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipWord16X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord16X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord16X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord16X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord16X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord16X32 #-}
-- | Fold the elements of a vector to a single value
foldWord16X32 :: (Word16 -> Word16 -> Word16) -> Word16X32 -> Word16
foldWord16X32 f' = \ v -> case unpackWord16X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldWord16X32 (+) = sumVector #-}

{-# INLINE sumWord16X32 #-}
-- | Sum up the elements of a vector to a single value.
sumWord16X32 :: Word16X32 -> Word16
sumWord16X32 (Word16X32 x1 x2 x3 x4) = case unpackWord16X8# (plusWord16X8# x1 (plusWord16X8# x2 (plusWord16X8# x3 x4))) of
    (# y1, y2, y3, y4, y5, y6, y7, y8 #) -> W16# y1 + W16# y2 + W16# y3 + W16# y4 + W16# y5 + W16# y6 + W16# y7 + W16# y8

{-# INLINE plusWord16X32 #-}
-- | Add two vectors element-wise.
plusWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
plusWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1) (Word16X32 m1_2 m2_2 m3_2 m4_2) = Word16X32 (plusWord16X8# m1_1 m1_2) (plusWord16X8# m2_1 m2_2) (plusWord16X8# m3_1 m3_2) (plusWord16X8# m4_1 m4_2)

{-# INLINE minusWord16X32 #-}
-- | Subtract two vectors element-wise.
minusWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
minusWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1) (Word16X32 m1_2 m2_2 m3_2 m4_2) = Word16X32 (minusWord16X8# m1_1 m1_2) (minusWord16X8# m2_1 m2_2) (minusWord16X8# m3_1 m3_2) (minusWord16X8# m4_1 m4_2)

{-# INLINE timesWord16X32 #-}
-- | Multiply two vectors element-wise.
timesWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
timesWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1) (Word16X32 m1_2 m2_2 m3_2 m4_2) = Word16X32 (timesWord16X8# m1_1 m1_2) (timesWord16X8# m2_1 m2_2) (timesWord16X8# m3_1 m3_2) (timesWord16X8# m4_1 m4_2)

{-# INLINE quotWord16X32 #-}
-- | Rounds towards zero element-wise.
quotWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
quotWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1) (Word16X32 m1_2 m2_2 m3_2 m4_2) = Word16X32 (quotWord16X8# m1_1 m1_2) (quotWord16X8# m2_1 m2_2) (quotWord16X8# m3_1 m3_2) (quotWord16X8# m4_1 m4_2)

{-# INLINE remWord16X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
remWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1) (Word16X32 m1_2 m2_2 m3_2 m4_2) = Word16X32 (remWord16X8# m1_1 m1_2) (remWord16X8# m2_1 m2_2) (remWord16X8# m3_1 m3_2) (remWord16X8# m4_1 m4_2)

{-# INLINE indexWord16X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X32Array :: ByteArray -> Int -> Word16X32
indexWord16X32Array (ByteArray a) (I# i) = Word16X32 (indexWord16X8Array# a ((i *# 4#) +# 0#)) (indexWord16X8Array# a ((i *# 4#) +# 1#)) (indexWord16X8Array# a ((i *# 4#) +# 2#)) (indexWord16X8Array# a ((i *# 4#) +# 3#))

{-# INLINE readWord16X32Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X32
readWord16X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16X8Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord16X8Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord16X8Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord16X8Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Word16X32 m1 m2 m3 m4 #))

{-# INLINE writeWord16X32Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X32 -> m ()
writeWord16X32Array (MutableByteArray a) (I# i) (Word16X32 m1 m2 m3 m4) = primitive_ (writeWord16X8Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeWord16X8Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeWord16X8Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeWord16X8Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexWord16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X32OffAddr :: Addr -> Int -> Word16X32
indexWord16X32OffAddr (Addr a) (I# i) = Word16X32 (indexWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#)

{-# INLINE readWord16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X32OffAddr :: PrimMonad m => Addr -> Int -> m Word16X32
readWord16X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s3 of
                (# s4, m4 #) -> (# s4, Word16X32 m1 m2 m3 m4 #))

{-# INLINE writeWord16X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X32OffAddr :: PrimMonad m => Addr -> Int -> Word16X32 -> m ()
writeWord16X32OffAddr (Addr a) (I# i) (Word16X32 m1 m2 m3 m4) = primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m2) >> primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m3) >> primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m4)


