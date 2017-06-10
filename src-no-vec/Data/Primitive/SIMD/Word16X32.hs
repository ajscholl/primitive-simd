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
data Word16X32 = Word16X32 Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord16# :: Word# -> Word#
broadcastWord16# v = v

packWord16# :: (# Word# #) -> Word#
packWord16# (# v #) = v

unpackWord16# :: Word# -> (# Word# #)
unpackWord16# v = (# v #)

insertWord16# :: Word# -> Word# -> Int# -> Word#
insertWord16# _ v _ = v

plusWord16# :: Word# -> Word# -> Word#
plusWord16# a b = case W16# a + W16# b of W16# c -> c

minusWord16# :: Word# -> Word# -> Word#
minusWord16# a b = case W16# a - W16# b of W16# c -> c

timesWord16# :: Word# -> Word# -> Word#
timesWord16# a b = case W16# a * W16# b of W16# c -> c

quotWord16# :: Word# -> Word# -> Word#
quotWord16# a b = case W16# a `quot` W16# b of W16# c -> c

remWord16# :: Word# -> Word# -> Word#
remWord16# a b = case W16# a `rem` W16# b of W16# c -> c

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
broadcastWord16X32 (W16# x) = case broadcastWord16# x of
    v -> Word16X32 v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

{-# INLINE[1] generateWord16X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord16X32 :: (Int -> Word16) -> Word16X32
generateWord16X32 f = packWord16X32 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15, f 16, f 17, f 18, f 19, f 20, f 21, f 22, f 23, f 24, f 25, f 26, f 27, f 28, f 29, f 30, f 31)

{-# INLINE packWord16X32 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X32 :: (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16) -> Word16X32
packWord16X32 (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16, W16# x17, W16# x18, W16# x19, W16# x20, W16# x21, W16# x22, W16# x23, W16# x24, W16# x25, W16# x26, W16# x27, W16# x28, W16# x29, W16# x30, W16# x31, W16# x32) = Word16X32 (packWord16# (# x1 #)) (packWord16# (# x2 #)) (packWord16# (# x3 #)) (packWord16# (# x4 #)) (packWord16# (# x5 #)) (packWord16# (# x6 #)) (packWord16# (# x7 #)) (packWord16# (# x8 #)) (packWord16# (# x9 #)) (packWord16# (# x10 #)) (packWord16# (# x11 #)) (packWord16# (# x12 #)) (packWord16# (# x13 #)) (packWord16# (# x14 #)) (packWord16# (# x15 #)) (packWord16# (# x16 #)) (packWord16# (# x17 #)) (packWord16# (# x18 #)) (packWord16# (# x19 #)) (packWord16# (# x20 #)) (packWord16# (# x21 #)) (packWord16# (# x22 #)) (packWord16# (# x23 #)) (packWord16# (# x24 #)) (packWord16# (# x25 #)) (packWord16# (# x26 #)) (packWord16# (# x27 #)) (packWord16# (# x28 #)) (packWord16# (# x29 #)) (packWord16# (# x30 #)) (packWord16# (# x31 #)) (packWord16# (# x32 #))

{-# INLINE unpackWord16X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X32 :: Word16X32 -> (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
unpackWord16X32 (Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = case unpackWord16# m1 of
    (# x1 #) -> case unpackWord16# m2 of
        (# x2 #) -> case unpackWord16# m3 of
            (# x3 #) -> case unpackWord16# m4 of
                (# x4 #) -> case unpackWord16# m5 of
                    (# x5 #) -> case unpackWord16# m6 of
                        (# x6 #) -> case unpackWord16# m7 of
                            (# x7 #) -> case unpackWord16# m8 of
                                (# x8 #) -> case unpackWord16# m9 of
                                    (# x9 #) -> case unpackWord16# m10 of
                                        (# x10 #) -> case unpackWord16# m11 of
                                            (# x11 #) -> case unpackWord16# m12 of
                                                (# x12 #) -> case unpackWord16# m13 of
                                                    (# x13 #) -> case unpackWord16# m14 of
                                                        (# x14 #) -> case unpackWord16# m15 of
                                                            (# x15 #) -> case unpackWord16# m16 of
                                                                (# x16 #) -> case unpackWord16# m17 of
                                                                    (# x17 #) -> case unpackWord16# m18 of
                                                                        (# x18 #) -> case unpackWord16# m19 of
                                                                            (# x19 #) -> case unpackWord16# m20 of
                                                                                (# x20 #) -> case unpackWord16# m21 of
                                                                                    (# x21 #) -> case unpackWord16# m22 of
                                                                                        (# x22 #) -> case unpackWord16# m23 of
                                                                                            (# x23 #) -> case unpackWord16# m24 of
                                                                                                (# x24 #) -> case unpackWord16# m25 of
                                                                                                    (# x25 #) -> case unpackWord16# m26 of
                                                                                                        (# x26 #) -> case unpackWord16# m27 of
                                                                                                            (# x27 #) -> case unpackWord16# m28 of
                                                                                                                (# x28 #) -> case unpackWord16# m29 of
                                                                                                                    (# x29 #) -> case unpackWord16# m30 of
                                                                                                                        (# x30 #) -> case unpackWord16# m31 of
                                                                                                                            (# x31 #) -> case unpackWord16# m32 of
                                                                                                                                (# x32 #) -> (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16, W16# x17, W16# x18, W16# x19, W16# x20, W16# x21, W16# x22, W16# x23, W16# x24, W16# x25, W16# x26, W16# x27, W16# x28, W16# x29, W16# x30, W16# x31, W16# x32)

{-# INLINE unsafeInsertWord16X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X32 :: Word16X32 -> Word16 -> Int -> Word16X32
unsafeInsertWord16X32 (Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) (W16# y) _i@(I# ip) | _i < 1 = Word16X32 (insertWord16# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 2 = Word16X32 m1 (insertWord16# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 3 = Word16X32 m1 m2 (insertWord16# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 4 = Word16X32 m1 m2 m3 (insertWord16# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 5 = Word16X32 m1 m2 m3 m4 (insertWord16# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 6 = Word16X32 m1 m2 m3 m4 m5 (insertWord16# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 7 = Word16X32 m1 m2 m3 m4 m5 m6 (insertWord16# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 8 = Word16X32 m1 m2 m3 m4 m5 m6 m7 (insertWord16# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 9 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 (insertWord16# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 10 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertWord16# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 11 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertWord16# m11 y (ip -# 10#)) m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 12 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertWord16# m12 y (ip -# 11#)) m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 13 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertWord16# m13 y (ip -# 12#)) m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 14 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertWord16# m14 y (ip -# 13#)) m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 15 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertWord16# m15 y (ip -# 14#)) m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 16 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertWord16# m16 y (ip -# 15#)) m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 17 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 (insertWord16# m17 y (ip -# 16#)) m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 18 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 (insertWord16# m18 y (ip -# 17#)) m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 19 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 (insertWord16# m19 y (ip -# 18#)) m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 20 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 (insertWord16# m20 y (ip -# 19#)) m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 21 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 (insertWord16# m21 y (ip -# 20#)) m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 22 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 (insertWord16# m22 y (ip -# 21#)) m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 23 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 (insertWord16# m23 y (ip -# 22#)) m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 24 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 (insertWord16# m24 y (ip -# 23#)) m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 25 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 (insertWord16# m25 y (ip -# 24#)) m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 26 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 (insertWord16# m26 y (ip -# 25#)) m27 m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 27 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 (insertWord16# m27 y (ip -# 26#)) m28 m29 m30 m31 m32
                                                                                                                                                                             | _i < 28 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 (insertWord16# m28 y (ip -# 27#)) m29 m30 m31 m32
                                                                                                                                                                             | _i < 29 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 (insertWord16# m29 y (ip -# 28#)) m30 m31 m32
                                                                                                                                                                             | _i < 30 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 (insertWord16# m30 y (ip -# 29#)) m31 m32
                                                                                                                                                                             | _i < 31 = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 (insertWord16# m31 y (ip -# 30#)) m32
                                                                                                                                                                             | otherwise = Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 (insertWord16# m32 y (ip -# 31#))

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

{-# INLINE plusWord16X32 #-}
-- | Add two vectors element-wise.
plusWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
plusWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word16X32 (plusWord16# m1_1 m1_2) (plusWord16# m2_1 m2_2) (plusWord16# m3_1 m3_2) (plusWord16# m4_1 m4_2) (plusWord16# m5_1 m5_2) (plusWord16# m6_1 m6_2) (plusWord16# m7_1 m7_2) (plusWord16# m8_1 m8_2) (plusWord16# m9_1 m9_2) (plusWord16# m10_1 m10_2) (plusWord16# m11_1 m11_2) (plusWord16# m12_1 m12_2) (plusWord16# m13_1 m13_2) (plusWord16# m14_1 m14_2) (plusWord16# m15_1 m15_2) (plusWord16# m16_1 m16_2) (plusWord16# m17_1 m17_2) (plusWord16# m18_1 m18_2) (plusWord16# m19_1 m19_2) (plusWord16# m20_1 m20_2) (plusWord16# m21_1 m21_2) (plusWord16# m22_1 m22_2) (plusWord16# m23_1 m23_2) (plusWord16# m24_1 m24_2) (plusWord16# m25_1 m25_2) (plusWord16# m26_1 m26_2) (plusWord16# m27_1 m27_2) (plusWord16# m28_1 m28_2) (plusWord16# m29_1 m29_2) (plusWord16# m30_1 m30_2) (plusWord16# m31_1 m31_2) (plusWord16# m32_1 m32_2)

{-# INLINE minusWord16X32 #-}
-- | Subtract two vectors element-wise.
minusWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
minusWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word16X32 (minusWord16# m1_1 m1_2) (minusWord16# m2_1 m2_2) (minusWord16# m3_1 m3_2) (minusWord16# m4_1 m4_2) (minusWord16# m5_1 m5_2) (minusWord16# m6_1 m6_2) (minusWord16# m7_1 m7_2) (minusWord16# m8_1 m8_2) (minusWord16# m9_1 m9_2) (minusWord16# m10_1 m10_2) (minusWord16# m11_1 m11_2) (minusWord16# m12_1 m12_2) (minusWord16# m13_1 m13_2) (minusWord16# m14_1 m14_2) (minusWord16# m15_1 m15_2) (minusWord16# m16_1 m16_2) (minusWord16# m17_1 m17_2) (minusWord16# m18_1 m18_2) (minusWord16# m19_1 m19_2) (minusWord16# m20_1 m20_2) (minusWord16# m21_1 m21_2) (minusWord16# m22_1 m22_2) (minusWord16# m23_1 m23_2) (minusWord16# m24_1 m24_2) (minusWord16# m25_1 m25_2) (minusWord16# m26_1 m26_2) (minusWord16# m27_1 m27_2) (minusWord16# m28_1 m28_2) (minusWord16# m29_1 m29_2) (minusWord16# m30_1 m30_2) (minusWord16# m31_1 m31_2) (minusWord16# m32_1 m32_2)

{-# INLINE timesWord16X32 #-}
-- | Multiply two vectors element-wise.
timesWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
timesWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word16X32 (timesWord16# m1_1 m1_2) (timesWord16# m2_1 m2_2) (timesWord16# m3_1 m3_2) (timesWord16# m4_1 m4_2) (timesWord16# m5_1 m5_2) (timesWord16# m6_1 m6_2) (timesWord16# m7_1 m7_2) (timesWord16# m8_1 m8_2) (timesWord16# m9_1 m9_2) (timesWord16# m10_1 m10_2) (timesWord16# m11_1 m11_2) (timesWord16# m12_1 m12_2) (timesWord16# m13_1 m13_2) (timesWord16# m14_1 m14_2) (timesWord16# m15_1 m15_2) (timesWord16# m16_1 m16_2) (timesWord16# m17_1 m17_2) (timesWord16# m18_1 m18_2) (timesWord16# m19_1 m19_2) (timesWord16# m20_1 m20_2) (timesWord16# m21_1 m21_2) (timesWord16# m22_1 m22_2) (timesWord16# m23_1 m23_2) (timesWord16# m24_1 m24_2) (timesWord16# m25_1 m25_2) (timesWord16# m26_1 m26_2) (timesWord16# m27_1 m27_2) (timesWord16# m28_1 m28_2) (timesWord16# m29_1 m29_2) (timesWord16# m30_1 m30_2) (timesWord16# m31_1 m31_2) (timesWord16# m32_1 m32_2)

{-# INLINE quotWord16X32 #-}
-- | Rounds towards zero element-wise.
quotWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
quotWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word16X32 (quotWord16# m1_1 m1_2) (quotWord16# m2_1 m2_2) (quotWord16# m3_1 m3_2) (quotWord16# m4_1 m4_2) (quotWord16# m5_1 m5_2) (quotWord16# m6_1 m6_2) (quotWord16# m7_1 m7_2) (quotWord16# m8_1 m8_2) (quotWord16# m9_1 m9_2) (quotWord16# m10_1 m10_2) (quotWord16# m11_1 m11_2) (quotWord16# m12_1 m12_2) (quotWord16# m13_1 m13_2) (quotWord16# m14_1 m14_2) (quotWord16# m15_1 m15_2) (quotWord16# m16_1 m16_2) (quotWord16# m17_1 m17_2) (quotWord16# m18_1 m18_2) (quotWord16# m19_1 m19_2) (quotWord16# m20_1 m20_2) (quotWord16# m21_1 m21_2) (quotWord16# m22_1 m22_2) (quotWord16# m23_1 m23_2) (quotWord16# m24_1 m24_2) (quotWord16# m25_1 m25_2) (quotWord16# m26_1 m26_2) (quotWord16# m27_1 m27_2) (quotWord16# m28_1 m28_2) (quotWord16# m29_1 m29_2) (quotWord16# m30_1 m30_2) (quotWord16# m31_1 m31_2) (quotWord16# m32_1 m32_2)

{-# INLINE remWord16X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X32 :: Word16X32 -> Word16X32 -> Word16X32
remWord16X32 (Word16X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word16X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word16X32 (remWord16# m1_1 m1_2) (remWord16# m2_1 m2_2) (remWord16# m3_1 m3_2) (remWord16# m4_1 m4_2) (remWord16# m5_1 m5_2) (remWord16# m6_1 m6_2) (remWord16# m7_1 m7_2) (remWord16# m8_1 m8_2) (remWord16# m9_1 m9_2) (remWord16# m10_1 m10_2) (remWord16# m11_1 m11_2) (remWord16# m12_1 m12_2) (remWord16# m13_1 m13_2) (remWord16# m14_1 m14_2) (remWord16# m15_1 m15_2) (remWord16# m16_1 m16_2) (remWord16# m17_1 m17_2) (remWord16# m18_1 m18_2) (remWord16# m19_1 m19_2) (remWord16# m20_1 m20_2) (remWord16# m21_1 m21_2) (remWord16# m22_1 m22_2) (remWord16# m23_1 m23_2) (remWord16# m24_1 m24_2) (remWord16# m25_1 m25_2) (remWord16# m26_1 m26_2) (remWord16# m27_1 m27_2) (remWord16# m28_1 m28_2) (remWord16# m29_1 m29_2) (remWord16# m30_1 m30_2) (remWord16# m31_1 m31_2) (remWord16# m32_1 m32_2)

{-# INLINE indexWord16X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X32Array :: ByteArray -> Int -> Word16X32
indexWord16X32Array (ByteArray a) (I# i) = Word16X32 (indexWord16Array# a ((i *# 32#) +# 0#)) (indexWord16Array# a ((i *# 32#) +# 1#)) (indexWord16Array# a ((i *# 32#) +# 2#)) (indexWord16Array# a ((i *# 32#) +# 3#)) (indexWord16Array# a ((i *# 32#) +# 4#)) (indexWord16Array# a ((i *# 32#) +# 5#)) (indexWord16Array# a ((i *# 32#) +# 6#)) (indexWord16Array# a ((i *# 32#) +# 7#)) (indexWord16Array# a ((i *# 32#) +# 8#)) (indexWord16Array# a ((i *# 32#) +# 9#)) (indexWord16Array# a ((i *# 32#) +# 10#)) (indexWord16Array# a ((i *# 32#) +# 11#)) (indexWord16Array# a ((i *# 32#) +# 12#)) (indexWord16Array# a ((i *# 32#) +# 13#)) (indexWord16Array# a ((i *# 32#) +# 14#)) (indexWord16Array# a ((i *# 32#) +# 15#)) (indexWord16Array# a ((i *# 32#) +# 16#)) (indexWord16Array# a ((i *# 32#) +# 17#)) (indexWord16Array# a ((i *# 32#) +# 18#)) (indexWord16Array# a ((i *# 32#) +# 19#)) (indexWord16Array# a ((i *# 32#) +# 20#)) (indexWord16Array# a ((i *# 32#) +# 21#)) (indexWord16Array# a ((i *# 32#) +# 22#)) (indexWord16Array# a ((i *# 32#) +# 23#)) (indexWord16Array# a ((i *# 32#) +# 24#)) (indexWord16Array# a ((i *# 32#) +# 25#)) (indexWord16Array# a ((i *# 32#) +# 26#)) (indexWord16Array# a ((i *# 32#) +# 27#)) (indexWord16Array# a ((i *# 32#) +# 28#)) (indexWord16Array# a ((i *# 32#) +# 29#)) (indexWord16Array# a ((i *# 32#) +# 30#)) (indexWord16Array# a ((i *# 32#) +# 31#))

{-# INLINE readWord16X32Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X32
readWord16X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16Array# a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord16Array# a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord16Array# a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord16Array# a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord16Array# a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord16Array# a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord16Array# a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord16Array# a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case readWord16Array# a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readWord16Array# a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readWord16Array# a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readWord16Array# a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readWord16Array# a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readWord16Array# a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readWord16Array# a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readWord16Array# a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case readWord16Array# a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case readWord16Array# a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case readWord16Array# a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case readWord16Array# a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case readWord16Array# a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case readWord16Array# a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case readWord16Array# a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case readWord16Array# a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case readWord16Array# a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case readWord16Array# a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case readWord16Array# a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case readWord16Array# a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case readWord16Array# a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case readWord16Array# a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case readWord16Array# a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case readWord16Array# a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeWord16X32Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X32 -> m ()
writeWord16X32Array (MutableByteArray a) (I# i) (Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeWord16Array# a ((i *# 32#) +# 0#) m1) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 1#) m2) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 2#) m3) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 3#) m4) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 4#) m5) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 5#) m6) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 6#) m7) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 7#) m8) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 8#) m9) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 9#) m10) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 10#) m11) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 11#) m12) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 12#) m13) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 13#) m14) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 14#) m15) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 15#) m16) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 16#) m17) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 17#) m18) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 18#) m19) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 19#) m20) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 20#) m21) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 21#) m22) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 22#) m23) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 23#) m24) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 24#) m25) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 25#) m26) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 26#) m27) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 27#) m28) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 28#) m29) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 29#) m30) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 30#) m31) >> primitive_ (writeWord16Array# a ((i *# 32#) +# 31#) m32)

{-# INLINE indexWord16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X32OffAddr :: Addr -> Int -> Word16X32
indexWord16X32OffAddr (Addr a) (I# i) = Word16X32 (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 2#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 6#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 10#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 14#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 18#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 22#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 26#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 30#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 34#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 38#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 42#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 46#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 50#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 54#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 58#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 64#) +# 62#)) 0#)

{-# INLINE readWord16X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X32OffAddr :: PrimMonad m => Addr -> Int -> m Word16X32
readWord16X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 6#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 10#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 12#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 14#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 18#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 20#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 22#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 26#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 28#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 30#) s15 of
                                                                (# s16, m16 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s16 of
                                                                    (# s17, m17 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 34#) s17 of
                                                                        (# s18, m18 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 36#) s18 of
                                                                            (# s19, m19 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 38#) s19 of
                                                                                (# s20, m20 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s20 of
                                                                                    (# s21, m21 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 42#) s21 of
                                                                                        (# s22, m22 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 44#) s22 of
                                                                                            (# s23, m23 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 46#) s23 of
                                                                                                (# s24, m24 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s24 of
                                                                                                    (# s25, m25 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 50#) s25 of
                                                                                                        (# s26, m26 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 52#) s26 of
                                                                                                            (# s27, m27 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 54#) s27 of
                                                                                                                (# s28, m28 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s28 of
                                                                                                                    (# s29, m29 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 58#) s29 of
                                                                                                                        (# s30, m30 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 60#) s30 of
                                                                                                                            (# s31, m31 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 62#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeWord16X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X32OffAddr :: PrimMonad m => Addr -> Int -> Word16X32 -> m ()
writeWord16X32OffAddr (Addr a) (I# i) (Word16X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 2#)) 0# m2) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0# m3) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 6#)) 0# m4) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m5) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 10#)) 0# m6) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0# m7) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 14#)) 0# m8) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m9) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 18#)) 0# m10) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0# m11) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 22#)) 0# m12) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m13) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 26#)) 0# m14) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0# m15) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 30#)) 0# m16) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m17) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 34#)) 0# m18) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0# m19) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 38#)) 0# m20) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m21) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 42#)) 0# m22) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0# m23) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 46#)) 0# m24) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m25) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 50#)) 0# m26) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0# m27) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 54#)) 0# m28) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m29) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 58#)) 0# m30) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0# m31) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 64#) +# 62#)) 0# m32)


