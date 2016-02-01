{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X32 (Word8X32) where

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

-- ** Word8X32
data Word8X32 = Word8X32 Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord8# :: Word# -> Word#
broadcastWord8# v = v

packWord8# :: (# Word# #) -> Word#
packWord8# (# v #) = v

unpackWord8# :: Word# -> (# Word# #)
unpackWord8# v = (# v #)

insertWord8# :: Word# -> Word# -> Int# -> Word#
insertWord8# _ v _ = v

plusWord8# :: Word# -> Word# -> Word#
plusWord8# a b = case W8# a + W8# b of W8# c -> c

minusWord8# :: Word# -> Word# -> Word#
minusWord8# a b = case W8# a - W8# b of W8# c -> c

timesWord8# :: Word# -> Word# -> Word#
timesWord8# a b = case W8# a * W8# b of W8# c -> c

quotWord8# :: Word# -> Word# -> Word#
quotWord8# a b = case W8# a `quot` W8# b of W8# c -> c

remWord8# :: Word# -> Word# -> Word#
remWord8# a b = case W8# a `rem` W8# b of W8# c -> c

abs' :: Word8 -> Word8
abs' (W8# x) = W8# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W8# x) of
    W8# y -> y

signum' :: Word8 -> Word8
signum' (W8# x) = W8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W8# x) of
    W8# y -> y

instance Eq Word8X32 where
    a == b = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Word8X32 where
    a `compare` b = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Word8X32 where
    showsPrec _ a s = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Word8X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Word8X32 where
    (+) = plusWord8X32
    (-) = minusWord8X32
    (*) = timesWord8X32
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X32 where
    type Elem Word8X32 = Word8
    type ElemTuple Word8X32 = (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 1
    broadcastVector    = broadcastWord8X32
    unsafeInsertVector = unsafeInsertWord8X32
    packVector         = packWord8X32
    unpackVector       = unpackWord8X32
    mapVector          = mapWord8X32
    zipVector          = zipWord8X32
    foldVector         = foldWord8X32

instance SIMDIntVector Word8X32 where
    quotVector = quotWord8X32
    remVector  = remWord8X32

instance Prim Word8X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X32 = V_Word8X32 (PV.Vector Word8X32)
newtype instance UV.MVector s Word8X32 = MV_Word8X32 (PMV.MVector s Word8X32)

instance Vector UV.Vector Word8X32 where
    basicUnsafeFreeze (MV_Word8X32 v) = V_Word8X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X32 v) = MV_Word8X32 <$> PV.unsafeThaw v
    basicLength (V_Word8X32 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X32 v) = V_Word8X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X32 m) (V_Word8X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X32 where
    basicLength (MV_Word8X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X32 v) = MV_Word8X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X32 v) (MV_Word8X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X32

{-# INLINE broadcastWord8X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X32 :: Word8 -> Word8X32
broadcastWord8X32 (W8# x) = case broadcastWord8# x of
    v -> Word8X32 v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v

{-# INLINE packWord8X32 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X32 :: (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8) -> Word8X32
packWord8X32 (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32) = Word8X32 (packWord8# (# x1 #)) (packWord8# (# x2 #)) (packWord8# (# x3 #)) (packWord8# (# x4 #)) (packWord8# (# x5 #)) (packWord8# (# x6 #)) (packWord8# (# x7 #)) (packWord8# (# x8 #)) (packWord8# (# x9 #)) (packWord8# (# x10 #)) (packWord8# (# x11 #)) (packWord8# (# x12 #)) (packWord8# (# x13 #)) (packWord8# (# x14 #)) (packWord8# (# x15 #)) (packWord8# (# x16 #)) (packWord8# (# x17 #)) (packWord8# (# x18 #)) (packWord8# (# x19 #)) (packWord8# (# x20 #)) (packWord8# (# x21 #)) (packWord8# (# x22 #)) (packWord8# (# x23 #)) (packWord8# (# x24 #)) (packWord8# (# x25 #)) (packWord8# (# x26 #)) (packWord8# (# x27 #)) (packWord8# (# x28 #)) (packWord8# (# x29 #)) (packWord8# (# x30 #)) (packWord8# (# x31 #)) (packWord8# (# x32 #))

{-# INLINE unpackWord8X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X32 :: Word8X32 -> (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
unpackWord8X32 (Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = case unpackWord8# m1 of
    (# x1 #) -> case unpackWord8# m2 of
        (# x2 #) -> case unpackWord8# m3 of
            (# x3 #) -> case unpackWord8# m4 of
                (# x4 #) -> case unpackWord8# m5 of
                    (# x5 #) -> case unpackWord8# m6 of
                        (# x6 #) -> case unpackWord8# m7 of
                            (# x7 #) -> case unpackWord8# m8 of
                                (# x8 #) -> case unpackWord8# m9 of
                                    (# x9 #) -> case unpackWord8# m10 of
                                        (# x10 #) -> case unpackWord8# m11 of
                                            (# x11 #) -> case unpackWord8# m12 of
                                                (# x12 #) -> case unpackWord8# m13 of
                                                    (# x13 #) -> case unpackWord8# m14 of
                                                        (# x14 #) -> case unpackWord8# m15 of
                                                            (# x15 #) -> case unpackWord8# m16 of
                                                                (# x16 #) -> case unpackWord8# m17 of
                                                                    (# x17 #) -> case unpackWord8# m18 of
                                                                        (# x18 #) -> case unpackWord8# m19 of
                                                                            (# x19 #) -> case unpackWord8# m20 of
                                                                                (# x20 #) -> case unpackWord8# m21 of
                                                                                    (# x21 #) -> case unpackWord8# m22 of
                                                                                        (# x22 #) -> case unpackWord8# m23 of
                                                                                            (# x23 #) -> case unpackWord8# m24 of
                                                                                                (# x24 #) -> case unpackWord8# m25 of
                                                                                                    (# x25 #) -> case unpackWord8# m26 of
                                                                                                        (# x26 #) -> case unpackWord8# m27 of
                                                                                                            (# x27 #) -> case unpackWord8# m28 of
                                                                                                                (# x28 #) -> case unpackWord8# m29 of
                                                                                                                    (# x29 #) -> case unpackWord8# m30 of
                                                                                                                        (# x30 #) -> case unpackWord8# m31 of
                                                                                                                            (# x31 #) -> case unpackWord8# m32 of
                                                                                                                                (# x32 #) -> (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32)

{-# INLINE unsafeInsertWord8X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X32 :: Word8X32 -> Word8 -> Int -> Word8X32
unsafeInsertWord8X32 (Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) (W8# y) _i@(I# ip) | _i < 1 = Word8X32 (insertWord8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 2 = Word8X32 m1 (insertWord8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 3 = Word8X32 m1 m2 (insertWord8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 4 = Word8X32 m1 m2 m3 (insertWord8# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 5 = Word8X32 m1 m2 m3 m4 (insertWord8# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 6 = Word8X32 m1 m2 m3 m4 m5 (insertWord8# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 7 = Word8X32 m1 m2 m3 m4 m5 m6 (insertWord8# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 8 = Word8X32 m1 m2 m3 m4 m5 m6 m7 (insertWord8# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 9 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 (insertWord8# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 10 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertWord8# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 11 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertWord8# m11 y (ip -# 10#)) m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 12 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertWord8# m12 y (ip -# 11#)) m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 13 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertWord8# m13 y (ip -# 12#)) m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 14 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertWord8# m14 y (ip -# 13#)) m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 15 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertWord8# m15 y (ip -# 14#)) m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 16 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertWord8# m16 y (ip -# 15#)) m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 17 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 (insertWord8# m17 y (ip -# 16#)) m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 18 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 (insertWord8# m18 y (ip -# 17#)) m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 19 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 (insertWord8# m19 y (ip -# 18#)) m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 20 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 (insertWord8# m20 y (ip -# 19#)) m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 21 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 (insertWord8# m21 y (ip -# 20#)) m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 22 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 (insertWord8# m22 y (ip -# 21#)) m23 m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 23 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 (insertWord8# m23 y (ip -# 22#)) m24 m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 24 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 (insertWord8# m24 y (ip -# 23#)) m25 m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 25 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 (insertWord8# m25 y (ip -# 24#)) m26 m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 26 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 (insertWord8# m26 y (ip -# 25#)) m27 m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 27 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 (insertWord8# m27 y (ip -# 26#)) m28 m29 m30 m31 m32
                                                                                                                                                                          | _i < 28 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 (insertWord8# m28 y (ip -# 27#)) m29 m30 m31 m32
                                                                                                                                                                          | _i < 29 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 (insertWord8# m29 y (ip -# 28#)) m30 m31 m32
                                                                                                                                                                          | _i < 30 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 (insertWord8# m30 y (ip -# 29#)) m31 m32
                                                                                                                                                                          | _i < 31 = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 (insertWord8# m31 y (ip -# 30#)) m32
                                                                                                                                                                          | otherwise = Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 (insertWord8# m32 y (ip -# 31#))

{-# INLINE[1] mapWord8X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X32 :: (Word8 -> Word8) -> Word8X32 -> Word8X32
mapWord8X32 f = mapWord8X32# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X32 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X32 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X32# #-}
-- | Unboxed helper function.
mapWord8X32# :: (Word# -> Word#) -> Word8X32 -> Word8X32
mapWord8X32# f = \ v -> case unpackWord8X32 v of
    (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32) -> packWord8X32 (W8# (f x1), W8# (f x2), W8# (f x3), W8# (f x4), W8# (f x5), W8# (f x6), W8# (f x7), W8# (f x8), W8# (f x9), W8# (f x10), W8# (f x11), W8# (f x12), W8# (f x13), W8# (f x14), W8# (f x15), W8# (f x16), W8# (f x17), W8# (f x18), W8# (f x19), W8# (f x20), W8# (f x21), W8# (f x22), W8# (f x23), W8# (f x24), W8# (f x25), W8# (f x26), W8# (f x27), W8# (f x28), W8# (f x29), W8# (f x30), W8# (f x31), W8# (f x32))

{-# INLINE[1] zipWord8X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X32 :: (Word8 -> Word8 -> Word8) -> Word8X32 -> Word8X32 -> Word8X32
zipWord8X32 f = \ v1 v2 -> case unpackWord8X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packWord8X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipWord8X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X32 #-}
-- | Fold the elements of a vector to a single value
foldWord8X32 :: (Word8 -> Word8 -> Word8) -> Word8X32 -> Word8
foldWord8X32 f' = \ v -> case unpackWord8X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# INLINE plusWord8X32 #-}
-- | Add two vectors element-wise.
plusWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
plusWord8X32 (Word8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word8X32 (plusWord8# m1_1 m1_2) (plusWord8# m2_1 m2_2) (plusWord8# m3_1 m3_2) (plusWord8# m4_1 m4_2) (plusWord8# m5_1 m5_2) (plusWord8# m6_1 m6_2) (plusWord8# m7_1 m7_2) (plusWord8# m8_1 m8_2) (plusWord8# m9_1 m9_2) (plusWord8# m10_1 m10_2) (plusWord8# m11_1 m11_2) (plusWord8# m12_1 m12_2) (plusWord8# m13_1 m13_2) (plusWord8# m14_1 m14_2) (plusWord8# m15_1 m15_2) (plusWord8# m16_1 m16_2) (plusWord8# m17_1 m17_2) (plusWord8# m18_1 m18_2) (plusWord8# m19_1 m19_2) (plusWord8# m20_1 m20_2) (plusWord8# m21_1 m21_2) (plusWord8# m22_1 m22_2) (plusWord8# m23_1 m23_2) (plusWord8# m24_1 m24_2) (plusWord8# m25_1 m25_2) (plusWord8# m26_1 m26_2) (plusWord8# m27_1 m27_2) (plusWord8# m28_1 m28_2) (plusWord8# m29_1 m29_2) (plusWord8# m30_1 m30_2) (plusWord8# m31_1 m31_2) (plusWord8# m32_1 m32_2)

{-# INLINE minusWord8X32 #-}
-- | Subtract two vectors element-wise.
minusWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
minusWord8X32 (Word8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word8X32 (minusWord8# m1_1 m1_2) (minusWord8# m2_1 m2_2) (minusWord8# m3_1 m3_2) (minusWord8# m4_1 m4_2) (minusWord8# m5_1 m5_2) (minusWord8# m6_1 m6_2) (minusWord8# m7_1 m7_2) (minusWord8# m8_1 m8_2) (minusWord8# m9_1 m9_2) (minusWord8# m10_1 m10_2) (minusWord8# m11_1 m11_2) (minusWord8# m12_1 m12_2) (minusWord8# m13_1 m13_2) (minusWord8# m14_1 m14_2) (minusWord8# m15_1 m15_2) (minusWord8# m16_1 m16_2) (minusWord8# m17_1 m17_2) (minusWord8# m18_1 m18_2) (minusWord8# m19_1 m19_2) (minusWord8# m20_1 m20_2) (minusWord8# m21_1 m21_2) (minusWord8# m22_1 m22_2) (minusWord8# m23_1 m23_2) (minusWord8# m24_1 m24_2) (minusWord8# m25_1 m25_2) (minusWord8# m26_1 m26_2) (minusWord8# m27_1 m27_2) (minusWord8# m28_1 m28_2) (minusWord8# m29_1 m29_2) (minusWord8# m30_1 m30_2) (minusWord8# m31_1 m31_2) (minusWord8# m32_1 m32_2)

{-# INLINE timesWord8X32 #-}
-- | Multiply two vectors element-wise.
timesWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
timesWord8X32 (Word8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word8X32 (timesWord8# m1_1 m1_2) (timesWord8# m2_1 m2_2) (timesWord8# m3_1 m3_2) (timesWord8# m4_1 m4_2) (timesWord8# m5_1 m5_2) (timesWord8# m6_1 m6_2) (timesWord8# m7_1 m7_2) (timesWord8# m8_1 m8_2) (timesWord8# m9_1 m9_2) (timesWord8# m10_1 m10_2) (timesWord8# m11_1 m11_2) (timesWord8# m12_1 m12_2) (timesWord8# m13_1 m13_2) (timesWord8# m14_1 m14_2) (timesWord8# m15_1 m15_2) (timesWord8# m16_1 m16_2) (timesWord8# m17_1 m17_2) (timesWord8# m18_1 m18_2) (timesWord8# m19_1 m19_2) (timesWord8# m20_1 m20_2) (timesWord8# m21_1 m21_2) (timesWord8# m22_1 m22_2) (timesWord8# m23_1 m23_2) (timesWord8# m24_1 m24_2) (timesWord8# m25_1 m25_2) (timesWord8# m26_1 m26_2) (timesWord8# m27_1 m27_2) (timesWord8# m28_1 m28_2) (timesWord8# m29_1 m29_2) (timesWord8# m30_1 m30_2) (timesWord8# m31_1 m31_2) (timesWord8# m32_1 m32_2)

{-# INLINE quotWord8X32 #-}
-- | Rounds towards zero element-wise.
quotWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
quotWord8X32 (Word8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word8X32 (quotWord8# m1_1 m1_2) (quotWord8# m2_1 m2_2) (quotWord8# m3_1 m3_2) (quotWord8# m4_1 m4_2) (quotWord8# m5_1 m5_2) (quotWord8# m6_1 m6_2) (quotWord8# m7_1 m7_2) (quotWord8# m8_1 m8_2) (quotWord8# m9_1 m9_2) (quotWord8# m10_1 m10_2) (quotWord8# m11_1 m11_2) (quotWord8# m12_1 m12_2) (quotWord8# m13_1 m13_2) (quotWord8# m14_1 m14_2) (quotWord8# m15_1 m15_2) (quotWord8# m16_1 m16_2) (quotWord8# m17_1 m17_2) (quotWord8# m18_1 m18_2) (quotWord8# m19_1 m19_2) (quotWord8# m20_1 m20_2) (quotWord8# m21_1 m21_2) (quotWord8# m22_1 m22_2) (quotWord8# m23_1 m23_2) (quotWord8# m24_1 m24_2) (quotWord8# m25_1 m25_2) (quotWord8# m26_1 m26_2) (quotWord8# m27_1 m27_2) (quotWord8# m28_1 m28_2) (quotWord8# m29_1 m29_2) (quotWord8# m30_1 m30_2) (quotWord8# m31_1 m31_2) (quotWord8# m32_1 m32_2)

{-# INLINE remWord8X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
remWord8X32 (Word8X32 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1 m17_1 m18_1 m19_1 m20_1 m21_1 m22_1 m23_1 m24_1 m25_1 m26_1 m27_1 m28_1 m29_1 m30_1 m31_1 m32_1) (Word8X32 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2 m17_2 m18_2 m19_2 m20_2 m21_2 m22_2 m23_2 m24_2 m25_2 m26_2 m27_2 m28_2 m29_2 m30_2 m31_2 m32_2) = Word8X32 (remWord8# m1_1 m1_2) (remWord8# m2_1 m2_2) (remWord8# m3_1 m3_2) (remWord8# m4_1 m4_2) (remWord8# m5_1 m5_2) (remWord8# m6_1 m6_2) (remWord8# m7_1 m7_2) (remWord8# m8_1 m8_2) (remWord8# m9_1 m9_2) (remWord8# m10_1 m10_2) (remWord8# m11_1 m11_2) (remWord8# m12_1 m12_2) (remWord8# m13_1 m13_2) (remWord8# m14_1 m14_2) (remWord8# m15_1 m15_2) (remWord8# m16_1 m16_2) (remWord8# m17_1 m17_2) (remWord8# m18_1 m18_2) (remWord8# m19_1 m19_2) (remWord8# m20_1 m20_2) (remWord8# m21_1 m21_2) (remWord8# m22_1 m22_2) (remWord8# m23_1 m23_2) (remWord8# m24_1 m24_2) (remWord8# m25_1 m25_2) (remWord8# m26_1 m26_2) (remWord8# m27_1 m27_2) (remWord8# m28_1 m28_2) (remWord8# m29_1 m29_2) (remWord8# m30_1 m30_2) (remWord8# m31_1 m31_2) (remWord8# m32_1 m32_2)

{-# INLINE indexWord8X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X32Array :: ByteArray -> Int -> Word8X32
indexWord8X32Array (ByteArray a) (I# i) = Word8X32 (indexWord8Array# a ((i *# 32#) +# 0#)) (indexWord8Array# a ((i *# 32#) +# 1#)) (indexWord8Array# a ((i *# 32#) +# 2#)) (indexWord8Array# a ((i *# 32#) +# 3#)) (indexWord8Array# a ((i *# 32#) +# 4#)) (indexWord8Array# a ((i *# 32#) +# 5#)) (indexWord8Array# a ((i *# 32#) +# 6#)) (indexWord8Array# a ((i *# 32#) +# 7#)) (indexWord8Array# a ((i *# 32#) +# 8#)) (indexWord8Array# a ((i *# 32#) +# 9#)) (indexWord8Array# a ((i *# 32#) +# 10#)) (indexWord8Array# a ((i *# 32#) +# 11#)) (indexWord8Array# a ((i *# 32#) +# 12#)) (indexWord8Array# a ((i *# 32#) +# 13#)) (indexWord8Array# a ((i *# 32#) +# 14#)) (indexWord8Array# a ((i *# 32#) +# 15#)) (indexWord8Array# a ((i *# 32#) +# 16#)) (indexWord8Array# a ((i *# 32#) +# 17#)) (indexWord8Array# a ((i *# 32#) +# 18#)) (indexWord8Array# a ((i *# 32#) +# 19#)) (indexWord8Array# a ((i *# 32#) +# 20#)) (indexWord8Array# a ((i *# 32#) +# 21#)) (indexWord8Array# a ((i *# 32#) +# 22#)) (indexWord8Array# a ((i *# 32#) +# 23#)) (indexWord8Array# a ((i *# 32#) +# 24#)) (indexWord8Array# a ((i *# 32#) +# 25#)) (indexWord8Array# a ((i *# 32#) +# 26#)) (indexWord8Array# a ((i *# 32#) +# 27#)) (indexWord8Array# a ((i *# 32#) +# 28#)) (indexWord8Array# a ((i *# 32#) +# 29#)) (indexWord8Array# a ((i *# 32#) +# 30#)) (indexWord8Array# a ((i *# 32#) +# 31#))

{-# INLINE readWord8X32Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X32
readWord8X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8Array# a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8Array# a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord8Array# a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord8Array# a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord8Array# a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord8Array# a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord8Array# a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord8Array# a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case readWord8Array# a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readWord8Array# a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readWord8Array# a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readWord8Array# a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readWord8Array# a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readWord8Array# a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readWord8Array# a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readWord8Array# a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case readWord8Array# a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case readWord8Array# a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case readWord8Array# a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case readWord8Array# a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case readWord8Array# a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case readWord8Array# a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case readWord8Array# a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case readWord8Array# a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case readWord8Array# a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case readWord8Array# a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case readWord8Array# a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case readWord8Array# a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case readWord8Array# a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case readWord8Array# a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case readWord8Array# a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case readWord8Array# a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeWord8X32Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X32 -> m ()
writeWord8X32Array (MutableByteArray a) (I# i) (Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeWord8Array# a ((i *# 32#) +# 0#) m1) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 1#) m2) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 2#) m3) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 3#) m4) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 4#) m5) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 5#) m6) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 6#) m7) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 7#) m8) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 8#) m9) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 9#) m10) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 10#) m11) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 11#) m12) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 12#) m13) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 13#) m14) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 14#) m15) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 15#) m16) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 16#) m17) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 17#) m18) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 18#) m19) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 19#) m20) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 20#) m21) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 21#) m22) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 22#) m23) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 23#) m24) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 24#) m25) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 25#) m26) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 26#) m27) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 27#) m28) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 28#) m29) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 29#) m30) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 30#) m31) >> primitive_ (writeWord8Array# a ((i *# 32#) +# 31#) m32)

{-# INLINE indexWord8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X32OffAddr :: Addr -> Int -> Word8X32
indexWord8X32OffAddr (Addr a) (I# i) = Word8X32 (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 1#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 3#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 5#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 7#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 9#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 11#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 13#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 15#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 17#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 19#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 21#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 23#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 25#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 27#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 29#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 32#) +# 31#)) 0#)

{-# INLINE readWord8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X32OffAddr :: PrimMonad m => Addr -> Int -> m Word8X32
readWord8X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 7#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 9#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 10#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 11#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 15#) s15 of
                                                                (# s16, m16 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s16 of
                                                                    (# s17, m17 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 17#) s17 of
                                                                        (# s18, m18 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 18#) s18 of
                                                                            (# s19, m19 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 19#) s19 of
                                                                                (# s20, m20 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s20 of
                                                                                    (# s21, m21 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 21#) s21 of
                                                                                        (# s22, m22 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 22#) s22 of
                                                                                            (# s23, m23 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 23#) s23 of
                                                                                                (# s24, m24 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s24 of
                                                                                                    (# s25, m25 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 25#) s25 of
                                                                                                        (# s26, m26 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 26#) s26 of
                                                                                                            (# s27, m27 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 27#) s27 of
                                                                                                                (# s28, m28 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s28 of
                                                                                                                    (# s29, m29 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 29#) s29 of
                                                                                                                        (# s30, m30 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 30#) s30 of
                                                                                                                            (# s31, m31 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 31#) s31 of
                                                                                                                                (# s32, m32 #) -> (# s32, Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 #))

{-# INLINE writeWord8X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X32OffAddr :: PrimMonad m => Addr -> Int -> Word8X32 -> m ()
writeWord8X32OffAddr (Addr a) (I# i) (Word8X32 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32) = primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 1#)) 0# m2) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0# m3) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 3#)) 0# m4) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m5) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 5#)) 0# m6) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0# m7) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 7#)) 0# m8) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m9) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 9#)) 0# m10) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0# m11) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 11#)) 0# m12) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m13) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 13#)) 0# m14) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0# m15) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 15#)) 0# m16) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m17) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 17#)) 0# m18) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0# m19) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 19#)) 0# m20) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m21) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 21#)) 0# m22) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0# m23) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 23#)) 0# m24) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m25) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 25#)) 0# m26) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0# m27) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 27#)) 0# m28) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m29) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 29#)) 0# m30) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0# m31) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 32#) +# 31#)) 0# m32)


