{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X32 (Int8X32) where

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

-- ** Int8X32
data Int8X32 = Int8X32 Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt8# :: Int# -> Int#
broadcastInt8# v = v

packInt8# :: (# Int# #) -> Int#
packInt8# (# v #) = v

unpackInt8# :: Int# -> (# Int# #)
unpackInt8# v = (# v #)

insertInt8# :: Int# -> Int# -> Int# -> Int#
insertInt8# _ v _ = v

negateInt8# :: Int# -> Int#
negateInt8# a = case negate (I8# a) of I8# b -> b

plusInt8# :: Int# -> Int# -> Int#
plusInt8# a b = case I8# a + I8# b of I8# c -> c

minusInt8# :: Int# -> Int# -> Int#
minusInt8# a b = case I8# a - I8# b of I8# c -> c

timesInt8# :: Int# -> Int# -> Int#
timesInt8# a b = case I8# a * I8# b of I8# c -> c

quotInt8# :: Int# -> Int# -> Int#
quotInt8# a b = case I8# a `quot` I8# b of I8# c -> c

remInt8# :: Int# -> Int# -> Int#
remInt8# a b = case I8# a `rem` I8# b of I8# c -> c

abs' :: Int8 -> Int8
abs' (I8# x) = I8# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I8# x) of
    I8# y -> y

signum' :: Int8 -> Int8
signum' (I8# x) = I8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I8# x) of
    I8# y -> y

instance Eq Int8X32 where
    a == b = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Int8X32 where
    a `compare` b = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Int8X32 where
    showsPrec _ a s = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Int8X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Int8X32 where
    (+) = plusInt8X32
    (-) = minusInt8X32
    (*) = timesInt8X32
    negate = negateInt8X32
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X32 where
    type Elem Int8X32 = Int8
    type ElemTuple Int8X32 = (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 1
    broadcastVector    = broadcastInt8X32
    unsafeInsertVector = unsafeInsertInt8X32
    packVector         = packInt8X32
    unpackVector       = unpackInt8X32
    mapVector          = mapInt8X32
    zipVector          = zipInt8X32
    foldVector         = foldInt8X32

instance SIMDIntVector Int8X32 where
    quotVector = quotInt8X32
    remVector  = remInt8X32

instance Prim Int8X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X32 = V_Int8X32 (PV.Vector Int8X32)
newtype instance UV.MVector s Int8X32 = MV_Int8X32 (PMV.MVector s Int8X32)

instance Vector UV.Vector Int8X32 where
    basicUnsafeFreeze (MV_Int8X32 v) = V_Int8X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X32 v) = MV_Int8X32 <$> PV.unsafeThaw v
    basicLength (V_Int8X32 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X32 v) = V_Int8X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X32 m) (V_Int8X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X32 where
    basicLength (MV_Int8X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X32 v) = MV_Int8X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X32 v) (MV_Int8X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X32

{-# INLINE broadcastInt8X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X32 :: Int8 -> Int8X32
broadcastInt8X32 (I8# x) = case broadcastInt8# x of
    v -> Int8X32 v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

{-# INLINE packInt8X32 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X32 :: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) -> Int8X32
packInt8X32 (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32) = Int8X32 (packInt8# (# x1 #)) (packInt8# (# x2 #)) (packInt8# (# x3 #)) (packInt8# (# x4 #)) (packInt8# (# x5 #)) (packInt8# (# x6 #)) (packInt8# (# x7 #)) (packInt8# (# x8 #)) (packInt8# (# x9 #)) (packInt8# (# x10 #)) (packInt8# (# x11 #)) (packInt8# (# x12 #)) (packInt8# (# x13 #)) (packInt8# (# x14 #)) (packInt8# (# x15 #)) (packInt8# (# x16 #)) (packInt8# (# x17 #)) (packInt8# (# x18 #)) (packInt8# (# x19 #)) (packInt8# (# x20 #)) (packInt8# (# x21 #)) (packInt8# (# x22 #)) (packInt8# (# x23 #)) (packInt8# (# x24 #)) (packInt8# (# x25 #)) (packInt8# (# x26 #)) (packInt8# (# x27 #)) (packInt8# (# x28 #)) (packInt8# (# x29 #)) (packInt8# (# x30 #)) (packInt8# (# x31 #)) (packInt8# (# x32 #))

{-# INLINE unpackInt8X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X32 :: Int8X32 -> (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
unpackInt8X32 (Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = case unpackInt8# m1 of
    (# x1 #) -> case unpackInt8# m2 of
        (# x2 #) -> case unpackInt8# m3 of
            (# x3 #) -> case unpackInt8# m4 of
                (# x4 #) -> case unpackInt8# m5 of
                    (# x5 #) -> case unpackInt8# m6 of
                        (# x6 #) -> case unpackInt8# m7 of
                            (# x7 #) -> case unpackInt8# m8 of
                                (# x8 #) -> case unpackInt8# m9 of
                                    (# x9 #) -> case unpackInt8# m10 of
                                        (# x10 #) -> case unpackInt8# m11 of
                                            (# x11 #) -> case unpackInt8# m12 of
                                                (# x12 #) -> case unpackInt8# m13 of
                                                    (# x13 #) -> case unpackInt8# m14 of
                                                        (# x14 #) -> case unpackInt8# m15 of
                                                            (# x15 #) -> case unpackInt8# m16 of
                                                                (# x16 #) -> case unpackInt8# m17 of
                                                                    (# x17 #) -> case unpackInt8# m18 of
                                                                        (# x18 #) -> case unpackInt8# m19 of
                                                                            (# x19 #) -> case unpackInt8# m20 of
                                                                                (# x20 #) -> case unpackInt8# m21 of
                                                                                    (# x21 #) -> case unpackInt8# m22 of
                                                                                        (# x22 #) -> case unpackInt8# m23 of
                                                                                            (# x23 #) -> case unpackInt8# m24 of
                                                                                                (# x24 #) -> case unpackInt8# m25 of
                                                                                                    (# x25 #) -> case unpackInt8# m26 of
                                                                                                        (# x26 #) -> case unpackInt8# m27 of
                                                                                                            (# x27 #) -> case unpackInt8# m28 of
                                                                                                                (# x28 #) -> case unpackInt8# m29 of
                                                                                                                    (# x29 #) -> case unpackInt8# m30 of
                                                                                                                        (# x30 #) -> case unpackInt8# m31 of
                                                                                                                            (# x31 #) -> case unpackInt8# m32 of
                                                                                                                                (# x32 #) -> (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32)

{-# INLINE unsafeInsertInt8X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X32 :: Int8X32 -> Int8 -> Int -> Int8X32
unsafeInsertInt8X32 (Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) (I8# y) _i@(I# ip) | _i < 1 = Int8X32 (insertInt8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 2 = Int8X32 m1 (insertInt8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 3 = Int8X32 m1 m2 (insertInt8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 4 = Int8X32 m1 m2 m3 (insertInt8# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 5 = Int8X32 m1 m2 m3 m4 (insertInt8# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 6 = Int8X32 m1 m2 m3 m4 m5 (insertInt8# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 7 = Int8X32 m1 m2 m3 m4 m5 m6 (insertInt8# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 8 = Int8X32 m1 m2 m3 m4 m5 m6 m7 (insertInt8# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 9 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 (insertInt8# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 10 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertInt8# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 11 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertInt8# m11 y (ip -# 10#)) m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 12 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertInt8# m12 y (ip -# 11#)) m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 13 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertInt8# m13 y (ip -# 12#)) m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 14 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertInt8# m14 y (ip -# 13#)) m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 15 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertInt8# m15 y (ip -# 14#)) m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 16 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertInt8# m16 y (ip -# 15#)) m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 17 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 (insertInt8# m17 y (ip -# 16#)) m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 18 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 (insertInt8# m18 y (ip -# 17#)) m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 19 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 (insertInt8# m19 y (ip -# 18#)) m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 20 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 (insertInt8# m20 y (ip -# 19#)) m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 21 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 (insertInt8# m21 y (ip -# 20#)) m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 22 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 (insertInt8# m22 y (ip -# 21#)) m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 23 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 (insertInt8# m23 y (ip -# 22#)) m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 24 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 (insertInt8# m24 y (ip -# 23#)) m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 25 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 (insertInt8# m25 y (ip -# 24#)) m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 26 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 (insertInt8# m26 y (ip -# 25#)) m27 m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 27 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 (insertInt8# m27 y (ip -# 26#)) m28 m29 m30 m31 m32
                                                                                                                                                                        | _i < 28 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 (insertInt8# m28 y (ip -# 27#)) m29 m30 m31 m32
                                                                                                                                                                        | _i < 29 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 (insertInt8# m29 y (ip -# 28#)) m30 m31 m32
                                                                                                                                                                        | _i < 30 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 (insertInt8# m30 y (ip -# 29#)) m31 m32
                                                                                                                                                                        | _i < 31 = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 (insertInt8# m31 y (ip -# 30#)) m32
                                                                                                                                                                        | otherwise = Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 (insertInt8# m32 y (ip -# 31#))

{-# INLINE[1] mapInt8X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X32 :: (Int8 -> Int8) -> Int8X32 -> Int8X32
mapInt8X32 f = mapInt8X32# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X32 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X32 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X32 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X32# #-}
-- | Unboxed helper function.
mapInt8X32# :: (Int# -> Int#) -> Int8X32 -> Int8X32
mapInt8X32# f = \ v -> case unpackInt8X32 v of
    (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32) -> packInt8X32 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4), I8# (f x5), I8# (f x6), I8# (f x7), I8# (f x8), I8# (f x9), I8# (f x10), I8# (f x11), I8# (f x12), I8# (f x13), I8# (f x14), I8# (f x15), I8# (f x16), I8# (f x17), I8# (f x18), I8# (f x19), I8# (f x20), I8# (f x21), I8# (f x22), I8# (f x23), I8# (f x24), I8# (f x25), I8# (f x26), I8# (f x27), I8# (f x28), I8# (f x29), I8# (f x30), I8# (f x31), I8# (f x32))

{-# INLINE[1] zipInt8X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X32 :: (Int8 -> Int8 -> Int8) -> Int8X32 -> Int8X32 -> Int8X32
zipInt8X32 f = \ v1 v2 -> case unpackInt8X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packInt8X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipInt8X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X32 #-}
-- | Fold the elements of a vector to a single value
foldInt8X32 :: (Int8 -> Int8 -> Int8) -> Int8X32 -> Int8
foldInt8X32 f' = \ v -> case unpackInt8X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# INLINE plusInt8X32 #-}
-- | Add two vectors element-wise.
plusInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
plusInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int8X32 (plusInt8# m1_1 m1_2) (plusInt8# m2_1 m2_2) (plusInt8# m3_1 m3_2) (plusInt8# m4_1 m4_2) (plusInt8# m5_1 m5_2) (plusInt8# m6_1 m6_2) (plusInt8# m7_1 m7_2) (plusInt8# m8_1 m8_2) (plusInt8# m9_1 m9_2) (plusInt8# m10_1 m10_2) (plusInt8# m11_1 m11_2) (plusInt8# m12_1 m12_2) (plusInt8# m13_1 m13_2) (plusInt8# m14_1 m14_2) (plusInt8# m15_1 m15_2) (plusInt8# m16_1 m16_2) (plusInt8# m17_1 m17_2) (plusInt8# m18_1 m18_2) (plusInt8# m19_1 m19_2) (plusInt8# m20_1 m20_2) (plusInt8# m21_1 m21_2) (plusInt8# m22_1 m22_2) (plusInt8# m23_1 m23_2) (plusInt8# m24_1 m24_2) (plusInt8# m25_1 m25_2) (plusInt8# m26_1 m26_2) (plusInt8# m27_1 m27_2) (plusInt8# m28_1 m28_2) (plusInt8# m29_1 m29_2) (plusInt8# m30_1 m30_2) (plusInt8# m31_1 m31_2) (plusInt8# m32_1 m32_2)

{-# INLINE minusInt8X32 #-}
-- | Subtract two vectors element-wise.
minusInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
minusInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int8X32 (minusInt8# m1_1 m1_2) (minusInt8# m2_1 m2_2) (minusInt8# m3_1 m3_2) (minusInt8# m4_1 m4_2) (minusInt8# m5_1 m5_2) (minusInt8# m6_1 m6_2) (minusInt8# m7_1 m7_2) (minusInt8# m8_1 m8_2) (minusInt8# m9_1 m9_2) (minusInt8# m10_1 m10_2) (minusInt8# m11_1 m11_2) (minusInt8# m12_1 m12_2) (minusInt8# m13_1 m13_2) (minusInt8# m14_1 m14_2) (minusInt8# m15_1 m15_2) (minusInt8# m16_1 m16_2) (minusInt8# m17_1 m17_2) (minusInt8# m18_1 m18_2) (minusInt8# m19_1 m19_2) (minusInt8# m20_1 m20_2) (minusInt8# m21_1 m21_2) (minusInt8# m22_1 m22_2) (minusInt8# m23_1 m23_2) (minusInt8# m24_1 m24_2) (minusInt8# m25_1 m25_2) (minusInt8# m26_1 m26_2) (minusInt8# m27_1 m27_2) (minusInt8# m28_1 m28_2) (minusInt8# m29_1 m29_2) (minusInt8# m30_1 m30_2) (minusInt8# m31_1 m31_2) (minusInt8# m32_1 m32_2)

{-# INLINE timesInt8X32 #-}
-- | Multiply two vectors element-wise.
timesInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
timesInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int8X32 (timesInt8# m1_1 m1_2) (timesInt8# m2_1 m2_2) (timesInt8# m3_1 m3_2) (timesInt8# m4_1 m4_2) (timesInt8# m5_1 m5_2) (timesInt8# m6_1 m6_2) (timesInt8# m7_1 m7_2) (timesInt8# m8_1 m8_2) (timesInt8# m9_1 m9_2) (timesInt8# m10_1 m10_2) (timesInt8# m11_1 m11_2) (timesInt8# m12_1 m12_2) (timesInt8# m13_1 m13_2) (timesInt8# m14_1 m14_2) (timesInt8# m15_1 m15_2) (timesInt8# m16_1 m16_2) (timesInt8# m17_1 m17_2) (timesInt8# m18_1 m18_2) (timesInt8# m19_1 m19_2) (timesInt8# m20_1 m20_2) (timesInt8# m21_1 m21_2) (timesInt8# m22_1 m22_2) (timesInt8# m23_1 m23_2) (timesInt8# m24_1 m24_2) (timesInt8# m25_1 m25_2) (timesInt8# m26_1 m26_2) (timesInt8# m27_1 m27_2) (timesInt8# m28_1 m28_2) (timesInt8# m29_1 m29_2) (timesInt8# m30_1 m30_2) (timesInt8# m31_1 m31_2) (timesInt8# m32_1 m32_2)

{-# INLINE quotInt8X32 #-}
-- | Rounds towards zero element-wise.
quotInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
quotInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int8X32 (quotInt8# m1_1 m1_2) (quotInt8# m2_1 m2_2) (quotInt8# m3_1 m3_2) (quotInt8# m4_1 m4_2) (quotInt8# m5_1 m5_2) (quotInt8# m6_1 m6_2) (quotInt8# m7_1 m7_2) (quotInt8# m8_1 m8_2) (quotInt8# m9_1 m9_2) (quotInt8# m10_1 m10_2) (quotInt8# m11_1 m11_2) (quotInt8# m12_1 m12_2) (quotInt8# m13_1 m13_2) (quotInt8# m14_1 m14_2) (quotInt8# m15_1 m15_2) (quotInt8# m16_1 m16_2) (quotInt8# m17_1 m17_2) (quotInt8# m18_1 m18_2) (quotInt8# m19_1 m19_2) (quotInt8# m20_1 m20_2) (quotInt8# m21_1 m21_2) (quotInt8# m22_1 m22_2) (quotInt8# m23_1 m23_2) (quotInt8# m24_1 m24_2) (quotInt8# m25_1 m25_2) (quotInt8# m26_1 m26_2) (quotInt8# m27_1 m27_2) (quotInt8# m28_1 m28_2) (quotInt8# m29_1 m29_2) (quotInt8# m30_1 m30_2) (quotInt8# m31_1 m31_2) (quotInt8# m32_1 m32_2)

{-# INLINE remInt8X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
remInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Int8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Int8X32 (remInt8# m1_1 m1_2) (remInt8# m2_1 m2_2) (remInt8# m3_1 m3_2) (remInt8# m4_1 m4_2) (remInt8# m5_1 m5_2) (remInt8# m6_1 m6_2) (remInt8# m7_1 m7_2) (remInt8# m8_1 m8_2) (remInt8# m9_1 m9_2) (remInt8# m10_1 m10_2) (remInt8# m11_1 m11_2) (remInt8# m12_1 m12_2) (remInt8# m13_1 m13_2) (remInt8# m14_1 m14_2) (remInt8# m15_1 m15_2) (remInt8# m16_1 m16_2) (remInt8# m17_1 m17_2) (remInt8# m18_1 m18_2) (remInt8# m19_1 m19_2) (remInt8# m20_1 m20_2) (remInt8# m21_1 m21_2) (remInt8# m22_1 m22_2) (remInt8# m23_1 m23_2) (remInt8# m24_1 m24_2) (remInt8# m25_1 m25_2) (remInt8# m26_1 m26_2) (remInt8# m27_1 m27_2) (remInt8# m28_1 m28_2) (remInt8# m29_1 m29_2) (remInt8# m30_1 m30_2) (remInt8# m31_1 m31_2) (remInt8# m32_1 m32_2)

{-# INLINE negateInt8X32 #-}
-- | Negate element-wise.
negateInt8X32 :: Int8X32 -> Int8X32
negateInt8X32 (Int8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) = Int8X32 (negateInt8# m1_1) (negateInt8# m2_1) (negateInt8# m3_1) (negateInt8# m4_1) (negateInt8# m5_1) (negateInt8# m6_1) (negateInt8# m7_1) (negateInt8# m8_1) (negateInt8# m9_1) (negateInt8# m10_1) (negateInt8# m11_1) (negateInt8# m12_1) (negateInt8# m13_1) (negateInt8# m14_1) (negateInt8# m15_1) (negateInt8# m16_1) (negateInt8# m17_1) (negateInt8# m18_1) (negateInt8# m19_1) (negateInt8# m20_1) (negateInt8# m21_1) (negateInt8# m22_1) (negateInt8# m23_1) (negateInt8# m24_1) (negateInt8# m25_1) (negateInt8# m26_1) (negateInt8# m27_1) (negateInt8# m28_1) (negateInt8# m29_1) (negateInt8# m30_1) (negateInt8# m31_1) (negateInt8# m32_1)

{-# INLINE indexInt8X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X32Array :: ByteArray -> Int -> Int8X32
indexInt8X32Array (ByteArray a) (I# i) = Int8X32 (indexInt8Array# a ((i *# 32#) +# 0#)) (indexInt8Array# a ((i *# 32#) +# 1#)) (indexInt8Array# a ((i *# 32#) +# 2#)) (indexInt8Array# a ((i *# 32#) +# 3#)) (indexInt8Array# a ((i *# 32#) +# 4#)) (indexInt8Array# a ((i *# 32#) +# 5#)) (indexInt8Array# a ((i *# 32#) +# 6#)) (indexInt8Array# a ((i *# 32#) +# 7#)) (indexInt8Array# a ((i *# 32#) +# 8#)) (indexInt8Array# a ((i *# 32#) +# 9#)) (indexInt8Array# a ((i *# 32#) +# 10#)) (indexInt8Array# a ((i *# 32#) +# 11#)) (indexInt8Array# a ((i *# 32#) +# 12#)) (indexInt8Array# a ((i *# 32#) +# 13#)) (indexInt8Array# a ((i *# 32#) +# 14#)) (indexInt8Array# a ((i *# 32#) +# 15#)) (indexInt8Array# a ((i *# 32#) +# 16#)) (indexInt8Array# a ((i *# 32#) +# 17#)) (indexInt8Array# a ((i *# 32#) +# 18#)) (indexInt8Array# a ((i *# 32#) +# 19#)) (indexInt8Array# a ((i *# 32#) +# 20#)) (indexInt8Array# a ((i *# 32#) +# 21#)) (indexInt8Array# a ((i *# 32#) +# 22#)) (indexInt8Array# a ((i *# 32#) +# 23#)) (indexInt8Array# a ((i *# 32#) +# 24#)) (indexInt8Array# a ((i *# 32#) +# 25#)) (indexInt8Array# a ((i *# 32#) +# 26#)) (indexInt8Array# a ((i *# 32#) +# 27#)) (indexInt8Array# a ((i *# 32#) +# 28#)) (indexInt8Array# a ((i *# 32#) +# 29#)) (indexInt8Array# a ((i *# 32#) +# 30#)) (indexInt8Array# a ((i *# 32#) +# 31#))

{-# INLINE readInt8X32Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X32
readInt8X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8Array# a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt8Array# a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt8Array# a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt8Array# a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt8Array# a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt8Array# a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt8Array# a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt8Array# a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case readInt8Array# a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readInt8Array# a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readInt8Array# a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readInt8Array# a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readInt8Array# a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readInt8Array# a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readInt8Array# a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readInt8Array# a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case readInt8Array# a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case readInt8Array# a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case readInt8Array# a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case readInt8Array# a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case readInt8Array# a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case readInt8Array# a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case readInt8Array# a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case readInt8Array# a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case readInt8Array# a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case readInt8Array# a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case readInt8Array# a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case readInt8Array# a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case readInt8Array# a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case readInt8Array# a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case readInt8Array# a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case readInt8Array# a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeInt8X32Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X32 -> m ()
writeInt8X32Array (MutableByteArray a) (I# i) (Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeInt8Array# a ((i *# 32#) +# 0#) m1) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 1#) m2) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 2#) m3) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 3#) m4) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 4#) m5) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 5#) m6) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 6#) m7) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 7#) m8) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 8#) m9) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 9#) m10) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 10#) m11) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 11#) m12) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 12#) m13) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 13#) m14) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 14#) m15) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 15#) m16) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 16#) m17) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 17#) m18) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 18#) m19) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 19#) m20) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 20#) m21) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 21#) m22) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 22#) m23) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 23#) m24) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 24#) m25) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 25#) m26) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 26#) m27) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 27#) m28) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 28#) m29) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 29#) m30) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 30#) m31) >> primitive_ (writeInt8Array# a ((i *# 32#) +# 31#) m32)

{-# INLINE indexInt8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X32OffAddr :: Addr -> Int -> Int8X32
indexInt8X32OffAddr (Addr a) (I# i) = Int8X32 (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 1#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 3#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 5#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 7#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 9#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 11#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 13#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 15#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 17#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 19#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 21#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 23#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 25#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 27#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 29#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 32#) +# 31#)) 0#)

{-# INLINE readInt8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X32OffAddr :: PrimMonad m => Addr -> Int -> m Int8X32
readInt8X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeInt8X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X32OffAddr :: PrimMonad m => Addr -> Int -> Int8X32 -> m ()
writeInt8X32OffAddr (Addr a) (I# i) (Int8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 1#)) 0# m2) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0# m3) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 3#)) 0# m4) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m5) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 5#)) 0# m6) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0# m7) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 7#)) 0# m8) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m9) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 9#)) 0# m10) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0# m11) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 11#)) 0# m12) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m13) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 13#)) 0# m14) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0# m15) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 15#)) 0# m16) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m17) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 17#)) 0# m18) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0# m19) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 19#)) 0# m20) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m21) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 21#)) 0# m22) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0# m23) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 23#)) 0# m24) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m25) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 25#)) 0# m26) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0# m27) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 27#)) 0# m28) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m29) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 29#)) 0# m30) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0# m31) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 32#) +# 31#)) 0# m32)


