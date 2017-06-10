{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X32 (Int16X32) where

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

-- ** Int16X32
data Int16X32 = Int16X32 Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt16# :: Int# -> Int#
broadcastInt16# v = v

packInt16# :: (# Int# #) -> Int#
packInt16# (# v #) = v

unpackInt16# :: Int# -> (# Int# #)
unpackInt16# v = (# v #)

insertInt16# :: Int# -> Int# -> Int# -> Int#
insertInt16# _ v _ = v

negateInt16# :: Int# -> Int#
negateInt16# a = case negate (I16# a) of I16# b -> b

plusInt16# :: Int# -> Int# -> Int#
plusInt16# a b = case I16# a + I16# b of I16# c -> c

minusInt16# :: Int# -> Int# -> Int#
minusInt16# a b = case I16# a - I16# b of I16# c -> c

timesInt16# :: Int# -> Int# -> Int#
timesInt16# a b = case I16# a * I16# b of I16# c -> c

quotInt16# :: Int# -> Int# -> Int#
quotInt16# a b = case I16# a `quot` I16# b of I16# c -> c

remInt16# :: Int# -> Int# -> Int#
remInt16# a b = case I16# a `rem` I16# b of I16# c -> c

abs' :: Int16 -> Int16
abs' (I16# x) = I16# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I16# x) of
    I16# y -> y

signum' :: Int16 -> Int16
signum' (I16# x) = I16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I16# x) of
    I16# y -> y

instance Eq Int16X32 where
    a == b = case unpackInt16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt16X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Int16X32 where
    a `compare` b = case unpackInt16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt16X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Int16X32 where
    showsPrec _ a s = case unpackInt16X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Int16X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Int16X32 where
    (+) = plusInt16X32
    (-) = minusInt16X32
    (*) = timesInt16X32
    negate = negateInt16X32
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X32 where
    type Elem Int16X32 = Int16
    type ElemTuple Int16X32 = (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 2
    broadcastVector    = broadcastInt16X32
    generateVector     = generateInt16X32
    unsafeInsertVector = unsafeInsertInt16X32
    packVector         = packInt16X32
    unpackVector       = unpackInt16X32
    mapVector          = mapInt16X32
    zipVector          = zipInt16X32
    foldVector         = foldInt16X32

instance SIMDIntVector Int16X32 where
    quotVector = quotInt16X32
    remVector  = remInt16X32

instance Prim Int16X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X32 = V_Int16X32 (PV.Vector Int16X32)
newtype instance UV.MVector s Int16X32 = MV_Int16X32 (PMV.MVector s Int16X32)

instance Vector UV.Vector Int16X32 where
    basicUnsafeFreeze (MV_Int16X32 v) = V_Int16X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X32 v) = MV_Int16X32 <$> PV.unsafeThaw v
    basicLength (V_Int16X32 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X32 v) = V_Int16X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X32 m) (V_Int16X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X32 where
    basicLength (MV_Int16X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X32 v) = MV_Int16X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X32 v) (MV_Int16X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X32

{-# INLINE broadcastInt16X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X32 :: Int16 -> Int16X32
broadcastInt16X32 (I16# x) = case broadcastInt16# x of
    v -> Int16X32 v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

{-# INLINE[1] generateInt16X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt16X32 :: (Int -> Int16) -> Int16X32
generateInt16X32 f = packInt16X32 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15, f 16, f 17, f 18, f 19, f 20, f 21, f 22, f 23, f 24, f 25, f 26, f 27, f 28, f 29, f 30, f 31)

{-# INLINE packInt16X32 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X32 :: (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16) -> Int16X32
packInt16X32 (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16, I16# x17, I16# x18, I16# x19, I16# x20, I16# x21, I16# x22, I16# x23, I16# x24, I16# x25, I16# x26, I16# x27, I16# x28, I16# x29, I16# x30, I16# x31, I16# x32) = Int16X32 (packInt16# (# x1 #)) (packInt16# (# x2 #)) (packInt16# (# x3 #)) (packInt16# (# x4 #)) (packInt16# (# x5 #)) (packInt16# (# x6 #)) (packInt16# (# x7 #)) (packInt16# (# x8 #)) (packInt16# (# x9 #)) (packInt16# (# x10 #)) (packInt16# (# x11 #)) (packInt16# (# x12 #)) (packInt16# (# x13 #)) (packInt16# (# x14 #)) (packInt16# (# x15 #)) (packInt16# (# x16 #)) (packInt16# (# x17 #)) (packInt16# (# x18 #)) (packInt16# (# x19 #)) (packInt16# (# x20 #)) (packInt16# (# x21 #)) (packInt16# (# x22 #)) (packInt16# (# x23 #)) (packInt16# (# x24 #)) (packInt16# (# x25 #)) (packInt16# (# x26 #)) (packInt16# (# x27 #)) (packInt16# (# x28 #)) (packInt16# (# x29 #)) (packInt16# (# x30 #)) (packInt16# (# x31 #)) (packInt16# (# x32 #))

{-# INLINE unpackInt16X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X32 :: Int16X32 -> (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
unpackInt16X32 (Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = case unpackInt16# m1 of
    (# x1 #) -> case unpackInt16# m2 of
        (# x2 #) -> case unpackInt16# m3 of
            (# x3 #) -> case unpackInt16# m4 of
                (# x4 #) -> case unpackInt16# m5 of
                    (# x5 #) -> case unpackInt16# m6 of
                        (# x6 #) -> case unpackInt16# m7 of
                            (# x7 #) -> case unpackInt16# m8 of
                                (# x8 #) -> case unpackInt16# m9 of
                                    (# x9 #) -> case unpackInt16# m10 of
                                        (# x10 #) -> case unpackInt16# m11 of
                                            (# x11 #) -> case unpackInt16# m12 of
                                                (# x12 #) -> case unpackInt16# m13 of
                                                    (# x13 #) -> case unpackInt16# m14 of
                                                        (# x14 #) -> case unpackInt16# m15 of
                                                            (# x15 #) -> case unpackInt16# m16 of
                                                                (# x16 #) -> case unpackInt16# m17 of
                                                                    (# x17 #) -> case unpackInt16# m18 of
                                                                        (# x18 #) -> case unpackInt16# m19 of
                                                                            (# x19 #) -> case unpackInt16# m20 of
                                                                                (# x20 #) -> case unpackInt16# m21 of
                                                                                    (# x21 #) -> case unpackInt16# m22 of
                                                                                        (# x22 #) -> case unpackInt16# m23 of
                                                                                            (# x23 #) -> case unpackInt16# m24 of
                                                                                                (# x24 #) -> case unpackInt16# m25 of
                                                                                                    (# x25 #) -> case unpackInt16# m26 of
                                                                                                        (# x26 #) -> case unpackInt16# m27 of
                                                                                                            (# x27 #) -> case unpackInt16# m28 of
                                                                                                                (# x28 #) -> case unpackInt16# m29 of
                                                                                                                    (# x29 #) -> case unpackInt16# m30 of
                                                                                                                        (# x30 #) -> case unpackInt16# m31 of
                                                                                                                            (# x31 #) -> case unpackInt16# m32 of
                                                                                                                                (# x32 #) -> (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16, I16# x17, I16# x18, I16# x19, I16# x20, I16# x21, I16# x22, I16# x23, I16# x24, I16# x25, I16# x26, I16# x27, I16# x28, I16# x29, I16# x30, I16# x31, I16# x32)

{-# INLINE unsafeInsertInt16X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X32 :: Int16X32 -> Int16 -> Int -> Int16X32
unsafeInsertInt16X32 (Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) (I16# y) _i@(I# ip) | _i < 1 = Int16X32 (insertInt16# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 2 = Int16X32 m1 (insertInt16# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 3 = Int16X32 m1 m2 (insertInt16# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 4 = Int16X32 m1 m2 m3 (insertInt16# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 5 = Int16X32 m1 m2 m3 m4 (insertInt16# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 6 = Int16X32 m1 m2 m3 m4 m5 (insertInt16# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 7 = Int16X32 m1 m2 m3 m4 m5 m6 (insertInt16# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 8 = Int16X32 m1 m2 m3 m4 m5 m6 m7 (insertInt16# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 9 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 (insertInt16# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 10 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertInt16# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 11 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertInt16# m11 y (ip -# 10#)) m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 12 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertInt16# m12 y (ip -# 11#)) m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 13 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertInt16# m13 y (ip -# 12#)) m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 14 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertInt16# m14 y (ip -# 13#)) m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 15 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertInt16# m15 y (ip -# 14#)) m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 16 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertInt16# m16 y (ip -# 15#)) m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 17 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 (insertInt16# m17 y (ip -# 16#)) m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 18 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 (insertInt16# m18 y (ip -# 17#)) m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 19 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 (insertInt16# m19 y (ip -# 18#)) m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 20 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 (insertInt16# m20 y (ip -# 19#)) m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 21 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 (insertInt16# m21 y (ip -# 20#)) m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 22 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 (insertInt16# m22 y (ip -# 21#)) m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 23 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 (insertInt16# m23 y (ip -# 22#)) m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 24 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 (insertInt16# m24 y (ip -# 23#)) m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 25 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 (insertInt16# m25 y (ip -# 24#)) m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 26 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 (insertInt16# m26 y (ip -# 25#)) m27 m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 27 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 (insertInt16# m27 y (ip -# 26#)) m28 m29 m30 m31 m32
                                                                                                                                                                           | _i < 28 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 (insertInt16# m28 y (ip -# 27#)) m29 m30 m31 m32
                                                                                                                                                                           | _i < 29 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 (insertInt16# m29 y (ip -# 28#)) m30 m31 m32
                                                                                                                                                                           | _i < 30 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 (insertInt16# m30 y (ip -# 29#)) m31 m32
                                                                                                                                                                           | _i < 31 = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 (insertInt16# m31 y (ip -# 30#)) m32
                                                                                                                                                                           | otherwise = Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 (insertInt16# m32 y (ip -# 31#))

{-# INLINE[1] mapInt16X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X32 :: (Int16 -> Int16) -> Int16X32 -> Int16X32
mapInt16X32 f = mapInt16X32# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X32 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X32 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X32 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X32# #-}
-- | Unboxed helper function.
mapInt16X32# :: (Int# -> Int#) -> Int16X32 -> Int16X32
mapInt16X32# f = \ v -> case unpackInt16X32 v of
    (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16, I16# x17, I16# x18, I16# x19, I16# x20, I16# x21, I16# x22, I16# x23, I16# x24, I16# x25, I16# x26, I16# x27, I16# x28, I16# x29, I16# x30, I16# x31, I16# x32) -> packInt16X32 (I16# (f x1), I16# (f x2), I16# (f x3), I16# (f x4), I16# (f x5), I16# (f x6), I16# (f x7), I16# (f x8), I16# (f x9), I16# (f x10), I16# (f x11), I16# (f x12), I16# (f x13), I16# (f x14), I16# (f x15), I16# (f x16), I16# (f x17), I16# (f x18), I16# (f x19), I16# (f x20), I16# (f x21), I16# (f x22), I16# (f x23), I16# (f x24), I16# (f x25), I16# (f x26), I16# (f x27), I16# (f x28), I16# (f x29), I16# (f x30), I16# (f x31), I16# (f x32))

{-# INLINE[1] zipInt16X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X32 :: (Int16 -> Int16 -> Int16) -> Int16X32 -> Int16X32 -> Int16X32
zipInt16X32 f = \ v1 v2 -> case unpackInt16X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt16X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packInt16X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipInt16X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X32 #-}
-- | Fold the elements of a vector to a single value
foldInt16X32 :: (Int16 -> Int16 -> Int16) -> Int16X32 -> Int16
foldInt16X32 f' = \ v -> case unpackInt16X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# INLINE plusInt16X32 #-}
-- | Add two vectors element-wise.
plusInt16X32 :: Int16X32 -> Int16X32 -> Int16X32
plusInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int16X32 (plusInt16# m1_1 m1_2) (plusInt16# m2_1 m2_2) (plusInt16# m3_1 m3_2) (plusInt16# m4_1 m4_2) (plusInt16# m5_1 m5_2) (plusInt16# m6_1 m6_2) (plusInt16# m7_1 m7_2) (plusInt16# m8_1 m8_2) (plusInt16# m9_1 m9_2) (plusInt16# m10_1 m10_2) (plusInt16# m11_1 m11_2) (plusInt16# m12_1 m12_2) (plusInt16# m13_1 m13_2) (plusInt16# m14_1 m14_2) (plusInt16# m15_1 m15_2) (plusInt16# m16_1 m16_2) (plusInt16# m17_1 m17_2) (plusInt16# m18_1 m18_2) (plusInt16# m19_1 m19_2) (plusInt16# m20_1 m20_2) (plusInt16# m21_1 m21_2) (plusInt16# m22_1 m22_2) (plusInt16# m23_1 m23_2) (plusInt16# m24_1 m24_2) (plusInt16# m25_1 m25_2) (plusInt16# m26_1 m26_2) (plusInt16# m27_1 m27_2) (plusInt16# m28_1 m28_2) (plusInt16# m29_1 m29_2) (plusInt16# m30_1 m30_2) (plusInt16# m31_1 m31_2) (plusInt16# m32_1 m32_2)

{-# INLINE minusInt16X32 #-}
-- | Subtract two vectors element-wise.
minusInt16X32 :: Int16X32 -> Int16X32 -> Int16X32
minusInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int16X32 (minusInt16# m1_1 m1_2) (minusInt16# m2_1 m2_2) (minusInt16# m3_1 m3_2) (minusInt16# m4_1 m4_2) (minusInt16# m5_1 m5_2) (minusInt16# m6_1 m6_2) (minusInt16# m7_1 m7_2) (minusInt16# m8_1 m8_2) (minusInt16# m9_1 m9_2) (minusInt16# m10_1 m10_2) (minusInt16# m11_1 m11_2) (minusInt16# m12_1 m12_2) (minusInt16# m13_1 m13_2) (minusInt16# m14_1 m14_2) (minusInt16# m15_1 m15_2) (minusInt16# m16_1 m16_2) (minusInt16# m17_1 m17_2) (minusInt16# m18_1 m18_2) (minusInt16# m19_1 m19_2) (minusInt16# m20_1 m20_2) (minusInt16# m21_1 m21_2) (minusInt16# m22_1 m22_2) (minusInt16# m23_1 m23_2) (minusInt16# m24_1 m24_2) (minusInt16# m25_1 m25_2) (minusInt16# m26_1 m26_2) (minusInt16# m27_1 m27_2) (minusInt16# m28_1 m28_2) (minusInt16# m29_1 m29_2) (minusInt16# m30_1 m30_2) (minusInt16# m31_1 m31_2) (minusInt16# m32_1 m32_2)

{-# INLINE timesInt16X32 #-}
-- | Multiply two vectors element-wise.
timesInt16X32 :: Int16X32 -> Int16X32 -> Int16X32
timesInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int16X32 (timesInt16# m1_1 m1_2) (timesInt16# m2_1 m2_2) (timesInt16# m3_1 m3_2) (timesInt16# m4_1 m4_2) (timesInt16# m5_1 m5_2) (timesInt16# m6_1 m6_2) (timesInt16# m7_1 m7_2) (timesInt16# m8_1 m8_2) (timesInt16# m9_1 m9_2) (timesInt16# m10_1 m10_2) (timesInt16# m11_1 m11_2) (timesInt16# m12_1 m12_2) (timesInt16# m13_1 m13_2) (timesInt16# m14_1 m14_2) (timesInt16# m15_1 m15_2) (timesInt16# m16_1 m16_2) (timesInt16# m17_1 m17_2) (timesInt16# m18_1 m18_2) (timesInt16# m19_1 m19_2) (timesInt16# m20_1 m20_2) (timesInt16# m21_1 m21_2) (timesInt16# m22_1 m22_2) (timesInt16# m23_1 m23_2) (timesInt16# m24_1 m24_2) (timesInt16# m25_1 m25_2) (timesInt16# m26_1 m26_2) (timesInt16# m27_1 m27_2) (timesInt16# m28_1 m28_2) (timesInt16# m29_1 m29_2) (timesInt16# m30_1 m30_2) (timesInt16# m31_1 m31_2) (timesInt16# m32_1 m32_2)

{-# INLINE quotInt16X32 #-}
-- | Rounds towards zero element-wise.
quotInt16X32 :: Int16X32 -> Int16X32 -> Int16X32
quotInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int16X32 (quotInt16# m1_1 m1_2) (quotInt16# m2_1 m2_2) (quotInt16# m3_1 m3_2) (quotInt16# m4_1 m4_2) (quotInt16# m5_1 m5_2) (quotInt16# m6_1 m6_2) (quotInt16# m7_1 m7_2) (quotInt16# m8_1 m8_2) (quotInt16# m9_1 m9_2) (quotInt16# m10_1 m10_2) (quotInt16# m11_1 m11_2) (quotInt16# m12_1 m12_2) (quotInt16# m13_1 m13_2) (quotInt16# m14_1 m14_2) (quotInt16# m15_1 m15_2) (quotInt16# m16_1 m16_2) (quotInt16# m17_1 m17_2) (quotInt16# m18_1 m18_2) (quotInt16# m19_1 m19_2) (quotInt16# m20_1 m20_2) (quotInt16# m21_1 m21_2) (quotInt16# m22_1 m22_2) (quotInt16# m23_1 m23_2) (quotInt16# m24_1 m24_2) (quotInt16# m25_1 m25_2) (quotInt16# m26_1 m26_2) (quotInt16# m27_1 m27_2) (quotInt16# m28_1 m28_2) (quotInt16# m29_1 m29_2) (quotInt16# m30_1 m30_2) (quotInt16# m31_1 m31_2) (quotInt16# m32_1 m32_2)

{-# INLINE remInt16X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X32 :: Int16X32 -> Int16X32 -> Int16X32
remInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int16X32 (remInt16# m1_1 m1_2) (remInt16# m2_1 m2_2) (remInt16# m3_1 m3_2) (remInt16# m4_1 m4_2) (remInt16# m5_1 m5_2) (remInt16# m6_1 m6_2) (remInt16# m7_1 m7_2) (remInt16# m8_1 m8_2) (remInt16# m9_1 m9_2) (remInt16# m10_1 m10_2) (remInt16# m11_1 m11_2) (remInt16# m12_1 m12_2) (remInt16# m13_1 m13_2) (remInt16# m14_1 m14_2) (remInt16# m15_1 m15_2) (remInt16# m16_1 m16_2) (remInt16# m17_1 m17_2) (remInt16# m18_1 m18_2) (remInt16# m19_1 m19_2) (remInt16# m20_1 m20_2) (remInt16# m21_1 m21_2) (remInt16# m22_1 m22_2) (remInt16# m23_1 m23_2) (remInt16# m24_1 m24_2) (remInt16# m25_1 m25_2) (remInt16# m26_1 m26_2) (remInt16# m27_1 m27_2) (remInt16# m28_1 m28_2) (remInt16# m29_1 m29_2) (remInt16# m30_1 m30_2) (remInt16# m31_1 m31_2) (remInt16# m32_1 m32_2)

{-# INLINE negateInt16X32 #-}
-- | Negate element-wise.
negateInt16X32 :: Int16X32 -> Int16X32
negateInt16X32 (Int16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) = Int16X32 (negateInt16# m1_1) (negateInt16# m2_1) (negateInt16# m3_1) (negateInt16# m4_1) (negateInt16# m5_1) (negateInt16# m6_1) (negateInt16# m7_1) (negateInt16# m8_1) (negateInt16# m9_1) (negateInt16# m10_1) (negateInt16# m11_1) (negateInt16# m12_1) (negateInt16# m13_1) (negateInt16# m14_1) (negateInt16# m15_1) (negateInt16# m16_1) (negateInt16# m17_1) (negateInt16# m18_1) (negateInt16# m19_1) (negateInt16# m20_1) (negateInt16# m21_1) (negateInt16# m22_1) (negateInt16# m23_1) (negateInt16# m24_1) (negateInt16# m25_1) (negateInt16# m26_1) (negateInt16# m27_1) (negateInt16# m28_1) (negateInt16# m29_1) (negateInt16# m30_1) (negateInt16# m31_1) (negateInt16# m32_1)

{-# INLINE indexInt16X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X32Array :: ByteArray -> Int -> Int16X32
indexInt16X32Array (ByteArray a) (I# i) = Int16X32 (indexInt16Array# a ((i *# 32#) +# 0#)) (indexInt16Array# a ((i *# 32#) +# 1#)) (indexInt16Array# a ((i *# 32#) +# 2#)) (indexInt16Array# a ((i *# 32#) +# 3#)) (indexInt16Array# a ((i *# 32#) +# 4#)) (indexInt16Array# a ((i *# 32#) +# 5#)) (indexInt16Array# a ((i *# 32#) +# 6#)) (indexInt16Array# a ((i *# 32#) +# 7#)) (indexInt16Array# a ((i *# 32#) +# 8#)) (indexInt16Array# a ((i *# 32#) +# 9#)) (indexInt16Array# a ((i *# 32#) +# 10#)) (indexInt16Array# a ((i *# 32#) +# 11#)) (indexInt16Array# a ((i *# 32#) +# 12#)) (indexInt16Array# a ((i *# 32#) +# 13#)) (indexInt16Array# a ((i *# 32#) +# 14#)) (indexInt16Array# a ((i *# 32#) +# 15#)) (indexInt16Array# a ((i *# 32#) +# 16#)) (indexInt16Array# a ((i *# 32#) +# 17#)) (indexInt16Array# a ((i *# 32#) +# 18#)) (indexInt16Array# a ((i *# 32#) +# 19#)) (indexInt16Array# a ((i *# 32#) +# 20#)) (indexInt16Array# a ((i *# 32#) +# 21#)) (indexInt16Array# a ((i *# 32#) +# 22#)) (indexInt16Array# a ((i *# 32#) +# 23#)) (indexInt16Array# a ((i *# 32#) +# 24#)) (indexInt16Array# a ((i *# 32#) +# 25#)) (indexInt16Array# a ((i *# 32#) +# 26#)) (indexInt16Array# a ((i *# 32#) +# 27#)) (indexInt16Array# a ((i *# 32#) +# 28#)) (indexInt16Array# a ((i *# 32#) +# 29#)) (indexInt16Array# a ((i *# 32#) +# 30#)) (indexInt16Array# a ((i *# 32#) +# 31#))

{-# INLINE readInt16X32Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X32
readInt16X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16Array# a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt16Array# a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt16Array# a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt16Array# a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt16Array# a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt16Array# a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt16Array# a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt16Array# a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case readInt16Array# a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readInt16Array# a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readInt16Array# a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readInt16Array# a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readInt16Array# a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readInt16Array# a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readInt16Array# a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readInt16Array# a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case readInt16Array# a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case readInt16Array# a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case readInt16Array# a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case readInt16Array# a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case readInt16Array# a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case readInt16Array# a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case readInt16Array# a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case readInt16Array# a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case readInt16Array# a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case readInt16Array# a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case readInt16Array# a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case readInt16Array# a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case readInt16Array# a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case readInt16Array# a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case readInt16Array# a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case readInt16Array# a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeInt16X32Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X32 -> m ()
writeInt16X32Array (MutableByteArray a) (I# i) (Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeInt16Array# a ((i *# 32#) +# 0#) m1) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 1#) m2) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 2#) m3) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 3#) m4) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 4#) m5) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 5#) m6) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 6#) m7) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 7#) m8) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 8#) m9) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 9#) m10) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 10#) m11) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 11#) m12) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 12#) m13) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 13#) m14) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 14#) m15) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 15#) m16) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 16#) m17) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 17#) m18) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 18#) m19) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 19#) m20) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 20#) m21) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 21#) m22) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 22#) m23) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 23#) m24) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 24#) m25) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 25#) m26) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 26#) m27) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 27#) m28) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 28#) m29) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 29#) m30) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 30#) m31) >> primitive_ (writeInt16Array# a ((i *# 32#) +# 31#) m32)

{-# INLINE indexInt16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X32OffAddr :: Addr -> Int -> Int16X32
indexInt16X32OffAddr (Addr a) (I# i) = Int16X32 (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 2#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 6#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 10#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 14#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 18#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 22#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 26#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 30#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 34#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 38#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 42#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 46#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 50#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 54#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 58#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 64#) +# 62#)) 0#)

{-# INLINE readInt16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X32OffAddr :: PrimMonad m => Addr -> Int -> m Int16X32
readInt16X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 6#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 10#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 12#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 14#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 18#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 20#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 22#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 26#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 28#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 30#) s15 of
                                                                (# s16, m16 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s16 of
                                                                    (# s17, m17 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 34#) s17 of
                                                                        (# s18, m18 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 36#) s18 of
                                                                            (# s19, m19 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 38#) s19 of
                                                                                (# s20, m20 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s20 of
                                                                                    (# s21, m21 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 42#) s21 of
                                                                                        (# s22, m22 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 44#) s22 of
                                                                                            (# s23, m23 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 46#) s23 of
                                                                                                (# s24, m24 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s24 of
                                                                                                    (# s25, m25 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 50#) s25 of
                                                                                                        (# s26, m26 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 52#) s26 of
                                                                                                            (# s27, m27 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 54#) s27 of
                                                                                                                (# s28, m28 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s28 of
                                                                                                                    (# s29, m29 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 58#) s29 of
                                                                                                                        (# s30, m30 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 60#) s30 of
                                                                                                                            (# s31, m31 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 62#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeInt16X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X32OffAddr :: PrimMonad m => Addr -> Int -> Int16X32 -> m ()
writeInt16X32OffAddr (Addr a) (I# i) (Int16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 2#)) 0# m2) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0# m3) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 6#)) 0# m4) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m5) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 10#)) 0# m6) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0# m7) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 14#)) 0# m8) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m9) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 18#)) 0# m10) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0# m11) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 22#)) 0# m12) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m13) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 26#)) 0# m14) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0# m15) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 30#)) 0# m16) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m17) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 34#)) 0# m18) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0# m19) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 38#)) 0# m20) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m21) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 42#)) 0# m22) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0# m23) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 46#)) 0# m24) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m25) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 50#)) 0# m26) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0# m27) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 54#)) 0# m28) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m29) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 58#)) 0# m30) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0# m31) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 64#) +# 62#)) 0# m32)


