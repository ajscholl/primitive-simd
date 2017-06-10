{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word32X16 (Word32X16) where

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

-- ** Word32X16
data Word32X16 = Word32X16 Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord32# :: Word# -> Word#
broadcastWord32# v = v

packWord32# :: (# Word# #) -> Word#
packWord32# (# v #) = v

unpackWord32# :: Word# -> (# Word# #)
unpackWord32# v = (# v #)

insertWord32# :: Word# -> Word# -> Int# -> Word#
insertWord32# _ v _ = v

plusWord32# :: Word# -> Word# -> Word#
plusWord32# a b = case W32# a + W32# b of W32# c -> c

minusWord32# :: Word# -> Word# -> Word#
minusWord32# a b = case W32# a - W32# b of W32# c -> c

timesWord32# :: Word# -> Word# -> Word#
timesWord32# a b = case W32# a * W32# b of W32# c -> c

quotWord32# :: Word# -> Word# -> Word#
quotWord32# a b = case W32# a `quot` W32# b of W32# c -> c

remWord32# :: Word# -> Word# -> Word#
remWord32# a b = case W32# a `rem` W32# b of W32# c -> c

abs' :: Word32 -> Word32
abs' (W32# x) = W32# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W32# x) of
    W32# y -> y

signum' :: Word32 -> Word32
signum' (W32# x) = W32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W32# x) of
    W32# y -> y

instance Eq Word32X16 where
    a == b = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Word32X16 where
    a `compare` b = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Word32X16 where
    showsPrec _ a s = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Word32X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Word32X16 where
    (+) = plusWord32X16
    (-) = minusWord32X16
    (*) = timesWord32X16
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word32X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word32X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word32X16 where
    type Elem Word32X16 = Word32
    type ElemTuple Word32X16 = (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 4
    broadcastVector    = broadcastWord32X16
    generateVector     = generateWord32X16
    unsafeInsertVector = unsafeInsertWord32X16
    packVector         = packWord32X16
    unpackVector       = unpackWord32X16
    mapVector          = mapWord32X16
    zipVector          = zipWord32X16
    foldVector         = foldWord32X16

instance SIMDIntVector Word32X16 where
    quotVector = quotWord32X16
    remVector  = remWord32X16

instance Prim Word32X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord32X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord32X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord32X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord32X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord32X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord32X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word32X16 = V_Word32X16 (PV.Vector Word32X16)
newtype instance UV.MVector s Word32X16 = MV_Word32X16 (PMV.MVector s Word32X16)

instance Vector UV.Vector Word32X16 where
    basicUnsafeFreeze (MV_Word32X16 v) = V_Word32X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word32X16 v) = MV_Word32X16 <$> PV.unsafeThaw v
    basicLength (V_Word32X16 v) = PV.length v
    basicUnsafeSlice start len (V_Word32X16 v) = V_Word32X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word32X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word32X16 m) (V_Word32X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word32X16 where
    basicLength (MV_Word32X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word32X16 v) = MV_Word32X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word32X16 v) (MV_Word32X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word32X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word32X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word32X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word32X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word32X16

{-# INLINE broadcastWord32X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord32X16 :: Word32 -> Word32X16
broadcastWord32X16 (W32# x) = case broadcastWord32# x of
    v -> Word32X16 v v v v v v v v v v v v v v v v

{-# INLINE[1] generateWord32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord32X16 :: (Int -> Word32) -> Word32X16
generateWord32X16 f = packWord32X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packWord32X16 #-}
-- | Pack the elements of a tuple into a vector.
packWord32X16 :: (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32) -> Word32X16
packWord32X16 (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16) = Word32X16 (packWord32# (# x1 #)) (packWord32# (# x2 #)) (packWord32# (# x3 #)) (packWord32# (# x4 #)) (packWord32# (# x5 #)) (packWord32# (# x6 #)) (packWord32# (# x7 #)) (packWord32# (# x8 #)) (packWord32# (# x9 #)) (packWord32# (# x10 #)) (packWord32# (# x11 #)) (packWord32# (# x12 #)) (packWord32# (# x13 #)) (packWord32# (# x14 #)) (packWord32# (# x15 #)) (packWord32# (# x16 #))

{-# INLINE unpackWord32X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord32X16 :: Word32X16 -> (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
unpackWord32X16 (Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackWord32# m1 of
    (# x1 #) -> case unpackWord32# m2 of
        (# x2 #) -> case unpackWord32# m3 of
            (# x3 #) -> case unpackWord32# m4 of
                (# x4 #) -> case unpackWord32# m5 of
                    (# x5 #) -> case unpackWord32# m6 of
                        (# x6 #) -> case unpackWord32# m7 of
                            (# x7 #) -> case unpackWord32# m8 of
                                (# x8 #) -> case unpackWord32# m9 of
                                    (# x9 #) -> case unpackWord32# m10 of
                                        (# x10 #) -> case unpackWord32# m11 of
                                            (# x11 #) -> case unpackWord32# m12 of
                                                (# x12 #) -> case unpackWord32# m13 of
                                                    (# x13 #) -> case unpackWord32# m14 of
                                                        (# x14 #) -> case unpackWord32# m15 of
                                                            (# x15 #) -> case unpackWord32# m16 of
                                                                (# x16 #) -> (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16)

{-# INLINE unsafeInsertWord32X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord32X16 :: Word32X16 -> Word32 -> Int -> Word32X16
unsafeInsertWord32X16 (Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (W32# y) _i@(I# ip) | _i < 1 = Word32X16 (insertWord32# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 2 = Word32X16 m1 (insertWord32# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 3 = Word32X16 m1 m2 (insertWord32# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 4 = Word32X16 m1 m2 m3 (insertWord32# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 5 = Word32X16 m1 m2 m3 m4 (insertWord32# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 6 = Word32X16 m1 m2 m3 m4 m5 (insertWord32# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 7 = Word32X16 m1 m2 m3 m4 m5 m6 (insertWord32# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 8 = Word32X16 m1 m2 m3 m4 m5 m6 m7 (insertWord32# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 9 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 (insertWord32# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 10 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertWord32# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                             | _i < 11 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertWord32# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                             | _i < 12 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertWord32# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                             | _i < 13 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertWord32# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                             | _i < 14 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertWord32# m14 y (ip -# 13#)) m15 m16
                                                                                                             | _i < 15 = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertWord32# m15 y (ip -# 14#)) m16
                                                                                                             | otherwise = Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertWord32# m16 y (ip -# 15#))

{-# INLINE[1] mapWord32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord32X16 :: (Word32 -> Word32) -> Word32X16 -> Word32X16
mapWord32X16 f = mapWord32X16# (\ x -> case f (W32# x) of { W32# y -> y})

{-# RULES "mapVector abs" mapWord32X16 abs = abs #-}
{-# RULES "mapVector signum" mapWord32X16 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord32X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord32X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord32X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord32X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord32X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord32X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord32X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord32X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord32X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord32X16# #-}
-- | Unboxed helper function.
mapWord32X16# :: (Word# -> Word#) -> Word32X16 -> Word32X16
mapWord32X16# f = \ v -> case unpackWord32X16 v of
    (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16) -> packWord32X16 (W32# (f x1), W32# (f x2), W32# (f x3), W32# (f x4), W32# (f x5), W32# (f x6), W32# (f x7), W32# (f x8), W32# (f x9), W32# (f x10), W32# (f x11), W32# (f x12), W32# (f x13), W32# (f x14), W32# (f x15), W32# (f x16))

{-# INLINE[1] zipWord32X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord32X16 :: (Word32 -> Word32 -> Word32) -> Word32X16 -> Word32X16 -> Word32X16
zipWord32X16 f = \ v1 v2 -> case unpackWord32X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packWord32X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipWord32X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord32X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord32X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord32X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord32X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord32X16 #-}
-- | Fold the elements of a vector to a single value
foldWord32X16 :: (Word32 -> Word32 -> Word32) -> Word32X16 -> Word32
foldWord32X16 f' = \ v -> case unpackWord32X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusWord32X16 #-}
-- | Add two vectors element-wise.
plusWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
plusWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word32X16 (plusWord32# m1_1 m1_2) (plusWord32# m2_1 m2_2) (plusWord32# m3_1 m3_2) (plusWord32# m4_1 m4_2) (plusWord32# m5_1 m5_2) (plusWord32# m6_1 m6_2) (plusWord32# m7_1 m7_2) (plusWord32# m8_1 m8_2) (plusWord32# m9_1 m9_2) (plusWord32# m10_1 m10_2) (plusWord32# m11_1 m11_2) (plusWord32# m12_1 m12_2) (plusWord32# m13_1 m13_2) (plusWord32# m14_1 m14_2) (plusWord32# m15_1 m15_2) (plusWord32# m16_1 m16_2)

{-# INLINE minusWord32X16 #-}
-- | Subtract two vectors element-wise.
minusWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
minusWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word32X16 (minusWord32# m1_1 m1_2) (minusWord32# m2_1 m2_2) (minusWord32# m3_1 m3_2) (minusWord32# m4_1 m4_2) (minusWord32# m5_1 m5_2) (minusWord32# m6_1 m6_2) (minusWord32# m7_1 m7_2) (minusWord32# m8_1 m8_2) (minusWord32# m9_1 m9_2) (minusWord32# m10_1 m10_2) (minusWord32# m11_1 m11_2) (minusWord32# m12_1 m12_2) (minusWord32# m13_1 m13_2) (minusWord32# m14_1 m14_2) (minusWord32# m15_1 m15_2) (minusWord32# m16_1 m16_2)

{-# INLINE timesWord32X16 #-}
-- | Multiply two vectors element-wise.
timesWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
timesWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word32X16 (timesWord32# m1_1 m1_2) (timesWord32# m2_1 m2_2) (timesWord32# m3_1 m3_2) (timesWord32# m4_1 m4_2) (timesWord32# m5_1 m5_2) (timesWord32# m6_1 m6_2) (timesWord32# m7_1 m7_2) (timesWord32# m8_1 m8_2) (timesWord32# m9_1 m9_2) (timesWord32# m10_1 m10_2) (timesWord32# m11_1 m11_2) (timesWord32# m12_1 m12_2) (timesWord32# m13_1 m13_2) (timesWord32# m14_1 m14_2) (timesWord32# m15_1 m15_2) (timesWord32# m16_1 m16_2)

{-# INLINE quotWord32X16 #-}
-- | Rounds towards zero element-wise.
quotWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
quotWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word32X16 (quotWord32# m1_1 m1_2) (quotWord32# m2_1 m2_2) (quotWord32# m3_1 m3_2) (quotWord32# m4_1 m4_2) (quotWord32# m5_1 m5_2) (quotWord32# m6_1 m6_2) (quotWord32# m7_1 m7_2) (quotWord32# m8_1 m8_2) (quotWord32# m9_1 m9_2) (quotWord32# m10_1 m10_2) (quotWord32# m11_1 m11_2) (quotWord32# m12_1 m12_2) (quotWord32# m13_1 m13_2) (quotWord32# m14_1 m14_2) (quotWord32# m15_1 m15_2) (quotWord32# m16_1 m16_2)

{-# INLINE remWord32X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
remWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word32X16 (remWord32# m1_1 m1_2) (remWord32# m2_1 m2_2) (remWord32# m3_1 m3_2) (remWord32# m4_1 m4_2) (remWord32# m5_1 m5_2) (remWord32# m6_1 m6_2) (remWord32# m7_1 m7_2) (remWord32# m8_1 m8_2) (remWord32# m9_1 m9_2) (remWord32# m10_1 m10_2) (remWord32# m11_1 m11_2) (remWord32# m12_1 m12_2) (remWord32# m13_1 m13_2) (remWord32# m14_1 m14_2) (remWord32# m15_1 m15_2) (remWord32# m16_1 m16_2)

{-# INLINE indexWord32X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord32X16Array :: ByteArray -> Int -> Word32X16
indexWord32X16Array (ByteArray a) (I# i) = Word32X16 (indexWord32Array# a ((i *# 16#) +# 0#)) (indexWord32Array# a ((i *# 16#) +# 1#)) (indexWord32Array# a ((i *# 16#) +# 2#)) (indexWord32Array# a ((i *# 16#) +# 3#)) (indexWord32Array# a ((i *# 16#) +# 4#)) (indexWord32Array# a ((i *# 16#) +# 5#)) (indexWord32Array# a ((i *# 16#) +# 6#)) (indexWord32Array# a ((i *# 16#) +# 7#)) (indexWord32Array# a ((i *# 16#) +# 8#)) (indexWord32Array# a ((i *# 16#) +# 9#)) (indexWord32Array# a ((i *# 16#) +# 10#)) (indexWord32Array# a ((i *# 16#) +# 11#)) (indexWord32Array# a ((i *# 16#) +# 12#)) (indexWord32Array# a ((i *# 16#) +# 13#)) (indexWord32Array# a ((i *# 16#) +# 14#)) (indexWord32Array# a ((i *# 16#) +# 15#))

{-# INLINE readWord32X16Array #-}
-- | Read a vector from specified index of the mutable array.
readWord32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word32X16
readWord32X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord32Array# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord32Array# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord32Array# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord32Array# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord32Array# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord32Array# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord32Array# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord32Array# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readWord32Array# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readWord32Array# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readWord32Array# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readWord32Array# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readWord32Array# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readWord32Array# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readWord32Array# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readWord32Array# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeWord32X16Array #-}
-- | Write a vector to specified index of mutable array.
writeWord32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word32X16 -> m ()
writeWord32X16Array (MutableByteArray a) (I# i) (Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeWord32Array# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeWord32Array# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexWord32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord32X16OffAddr :: Addr -> Int -> Word32X16
indexWord32X16OffAddr (Addr a) (I# i) = Word32X16 (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0#)

{-# INLINE readWord32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord32X16OffAddr :: PrimMonad m => Addr -> Int -> m Word32X16
readWord32X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 28#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 36#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 44#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 52#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 60#) s15 of
                                                                (# s16, m16 #) -> (# s16, Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeWord32X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord32X16OffAddr :: PrimMonad m => Addr -> Int -> Word32X16 -> m ()
writeWord32X16OffAddr (Addr a) (I# i) (Word32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0# m2) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m3) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0# m4) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m5) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0# m6) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m7) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0# m8) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m9) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0# m10) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m11) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0# m12) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m13) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0# m14) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m15) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0# m16)


