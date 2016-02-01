{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X16 (Word8X16) where

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

-- ** Word8X16
data Word8X16 = Word8X16 Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

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

instance Eq Word8X16 where
    a == b = case unpackWord8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Word8X16 where
    a `compare` b = case unpackWord8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Word8X16 where
    showsPrec _ a s = case unpackWord8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Word8X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Word8X16 where
    (+) = plusWord8X16
    (-) = minusWord8X16
    (*) = timesWord8X16
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X16 where
    type Elem Word8X16 = Word8
    type ElemTuple Word8X16 = (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 1
    broadcastVector    = broadcastWord8X16
    unsafeInsertVector = unsafeInsertWord8X16
    packVector         = packWord8X16
    unpackVector       = unpackWord8X16
    mapVector          = mapWord8X16
    zipVector          = zipWord8X16
    foldVector         = foldWord8X16

instance SIMDIntVector Word8X16 where
    quotVector = quotWord8X16
    remVector  = remWord8X16

instance Prim Word8X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X16 = V_Word8X16 (PV.Vector Word8X16)
newtype instance UV.MVector s Word8X16 = MV_Word8X16 (PMV.MVector s Word8X16)

instance Vector UV.Vector Word8X16 where
    basicUnsafeFreeze (MV_Word8X16 v) = V_Word8X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X16 v) = MV_Word8X16 <$> PV.unsafeThaw v
    basicLength (V_Word8X16 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X16 v) = V_Word8X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X16 m) (V_Word8X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X16 where
    basicLength (MV_Word8X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X16 v) = MV_Word8X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X16 v) (MV_Word8X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X16

{-# INLINE broadcastWord8X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X16 :: Word8 -> Word8X16
broadcastWord8X16 (W8# x) = case broadcastWord8# x of
    v -> Word8X16 v v v v v v v v v v v v v v v v

{-# INLINE packWord8X16 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X16 :: (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8) -> Word8X16
packWord8X16 (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16) = Word8X16 (packWord8# (# x1 #)) (packWord8# (# x2 #)) (packWord8# (# x3 #)) (packWord8# (# x4 #)) (packWord8# (# x5 #)) (packWord8# (# x6 #)) (packWord8# (# x7 #)) (packWord8# (# x8 #)) (packWord8# (# x9 #)) (packWord8# (# x10 #)) (packWord8# (# x11 #)) (packWord8# (# x12 #)) (packWord8# (# x13 #)) (packWord8# (# x14 #)) (packWord8# (# x15 #)) (packWord8# (# x16 #))

{-# INLINE unpackWord8X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X16 :: Word8X16 -> (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
unpackWord8X16 (Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackWord8# m1 of
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
                                                                (# x16 #) -> (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16)

{-# INLINE unsafeInsertWord8X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X16 :: Word8X16 -> Word8 -> Int -> Word8X16
unsafeInsertWord8X16 (Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (W8# y) _i@(I# ip) | _i < 1 = Word8X16 (insertWord8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 2 = Word8X16 m1 (insertWord8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 3 = Word8X16 m1 m2 (insertWord8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 4 = Word8X16 m1 m2 m3 (insertWord8# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 5 = Word8X16 m1 m2 m3 m4 (insertWord8# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 6 = Word8X16 m1 m2 m3 m4 m5 (insertWord8# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 7 = Word8X16 m1 m2 m3 m4 m5 m6 (insertWord8# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 8 = Word8X16 m1 m2 m3 m4 m5 m6 m7 (insertWord8# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 9 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 (insertWord8# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 10 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertWord8# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                          | _i < 11 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertWord8# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                          | _i < 12 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertWord8# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                          | _i < 13 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertWord8# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                          | _i < 14 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertWord8# m14 y (ip -# 13#)) m15 m16
                                                                                                          | _i < 15 = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertWord8# m15 y (ip -# 14#)) m16
                                                                                                          | otherwise = Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertWord8# m16 y (ip -# 15#))

{-# INLINE[1] mapWord8X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X16 :: (Word8 -> Word8) -> Word8X16 -> Word8X16
mapWord8X16 f = mapWord8X16# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X16 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X16 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X16# #-}
-- | Unboxed helper function.
mapWord8X16# :: (Word# -> Word#) -> Word8X16 -> Word8X16
mapWord8X16# f = \ v -> case unpackWord8X16 v of
    (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16) -> packWord8X16 (W8# (f x1), W8# (f x2), W8# (f x3), W8# (f x4), W8# (f x5), W8# (f x6), W8# (f x7), W8# (f x8), W8# (f x9), W8# (f x10), W8# (f x11), W8# (f x12), W8# (f x13), W8# (f x14), W8# (f x15), W8# (f x16))

{-# INLINE[1] zipWord8X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X16 :: (Word8 -> Word8 -> Word8) -> Word8X16 -> Word8X16 -> Word8X16
zipWord8X16 f = \ v1 v2 -> case unpackWord8X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord8X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packWord8X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipWord8X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X16 #-}
-- | Fold the elements of a vector to a single value
foldWord8X16 :: (Word8 -> Word8 -> Word8) -> Word8X16 -> Word8
foldWord8X16 f' = \ v -> case unpackWord8X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusWord8X16 #-}
-- | Add two vectors element-wise.
plusWord8X16 :: Word8X16 -> Word8X16 -> Word8X16
plusWord8X16 (Word8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word8X16 (plusWord8# m1_1 m1_2) (plusWord8# m2_1 m2_2) (plusWord8# m3_1 m3_2) (plusWord8# m4_1 m4_2) (plusWord8# m5_1 m5_2) (plusWord8# m6_1 m6_2) (plusWord8# m7_1 m7_2) (plusWord8# m8_1 m8_2) (plusWord8# m9_1 m9_2) (plusWord8# m10_1 m10_2) (plusWord8# m11_1 m11_2) (plusWord8# m12_1 m12_2) (plusWord8# m13_1 m13_2) (plusWord8# m14_1 m14_2) (plusWord8# m15_1 m15_2) (plusWord8# m16_1 m16_2)

{-# INLINE minusWord8X16 #-}
-- | Subtract two vectors element-wise.
minusWord8X16 :: Word8X16 -> Word8X16 -> Word8X16
minusWord8X16 (Word8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word8X16 (minusWord8# m1_1 m1_2) (minusWord8# m2_1 m2_2) (minusWord8# m3_1 m3_2) (minusWord8# m4_1 m4_2) (minusWord8# m5_1 m5_2) (minusWord8# m6_1 m6_2) (minusWord8# m7_1 m7_2) (minusWord8# m8_1 m8_2) (minusWord8# m9_1 m9_2) (minusWord8# m10_1 m10_2) (minusWord8# m11_1 m11_2) (minusWord8# m12_1 m12_2) (minusWord8# m13_1 m13_2) (minusWord8# m14_1 m14_2) (minusWord8# m15_1 m15_2) (minusWord8# m16_1 m16_2)

{-# INLINE timesWord8X16 #-}
-- | Multiply two vectors element-wise.
timesWord8X16 :: Word8X16 -> Word8X16 -> Word8X16
timesWord8X16 (Word8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word8X16 (timesWord8# m1_1 m1_2) (timesWord8# m2_1 m2_2) (timesWord8# m3_1 m3_2) (timesWord8# m4_1 m4_2) (timesWord8# m5_1 m5_2) (timesWord8# m6_1 m6_2) (timesWord8# m7_1 m7_2) (timesWord8# m8_1 m8_2) (timesWord8# m9_1 m9_2) (timesWord8# m10_1 m10_2) (timesWord8# m11_1 m11_2) (timesWord8# m12_1 m12_2) (timesWord8# m13_1 m13_2) (timesWord8# m14_1 m14_2) (timesWord8# m15_1 m15_2) (timesWord8# m16_1 m16_2)

{-# INLINE quotWord8X16 #-}
-- | Rounds towards zero element-wise.
quotWord8X16 :: Word8X16 -> Word8X16 -> Word8X16
quotWord8X16 (Word8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word8X16 (quotWord8# m1_1 m1_2) (quotWord8# m2_1 m2_2) (quotWord8# m3_1 m3_2) (quotWord8# m4_1 m4_2) (quotWord8# m5_1 m5_2) (quotWord8# m6_1 m6_2) (quotWord8# m7_1 m7_2) (quotWord8# m8_1 m8_2) (quotWord8# m9_1 m9_2) (quotWord8# m10_1 m10_2) (quotWord8# m11_1 m11_2) (quotWord8# m12_1 m12_2) (quotWord8# m13_1 m13_2) (quotWord8# m14_1 m14_2) (quotWord8# m15_1 m15_2) (quotWord8# m16_1 m16_2)

{-# INLINE remWord8X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X16 :: Word8X16 -> Word8X16 -> Word8X16
remWord8X16 (Word8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Word8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Word8X16 (remWord8# m1_1 m1_2) (remWord8# m2_1 m2_2) (remWord8# m3_1 m3_2) (remWord8# m4_1 m4_2) (remWord8# m5_1 m5_2) (remWord8# m6_1 m6_2) (remWord8# m7_1 m7_2) (remWord8# m8_1 m8_2) (remWord8# m9_1 m9_2) (remWord8# m10_1 m10_2) (remWord8# m11_1 m11_2) (remWord8# m12_1 m12_2) (remWord8# m13_1 m13_2) (remWord8# m14_1 m14_2) (remWord8# m15_1 m15_2) (remWord8# m16_1 m16_2)

{-# INLINE indexWord8X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X16Array :: ByteArray -> Int -> Word8X16
indexWord8X16Array (ByteArray a) (I# i) = Word8X16 (indexWord8Array# a ((i *# 16#) +# 0#)) (indexWord8Array# a ((i *# 16#) +# 1#)) (indexWord8Array# a ((i *# 16#) +# 2#)) (indexWord8Array# a ((i *# 16#) +# 3#)) (indexWord8Array# a ((i *# 16#) +# 4#)) (indexWord8Array# a ((i *# 16#) +# 5#)) (indexWord8Array# a ((i *# 16#) +# 6#)) (indexWord8Array# a ((i *# 16#) +# 7#)) (indexWord8Array# a ((i *# 16#) +# 8#)) (indexWord8Array# a ((i *# 16#) +# 9#)) (indexWord8Array# a ((i *# 16#) +# 10#)) (indexWord8Array# a ((i *# 16#) +# 11#)) (indexWord8Array# a ((i *# 16#) +# 12#)) (indexWord8Array# a ((i *# 16#) +# 13#)) (indexWord8Array# a ((i *# 16#) +# 14#)) (indexWord8Array# a ((i *# 16#) +# 15#))

{-# INLINE readWord8X16Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X16
readWord8X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8Array# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8Array# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord8Array# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord8Array# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord8Array# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord8Array# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord8Array# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord8Array# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readWord8Array# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readWord8Array# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readWord8Array# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readWord8Array# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readWord8Array# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readWord8Array# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readWord8Array# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readWord8Array# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeWord8X16Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X16 -> m ()
writeWord8X16Array (MutableByteArray a) (I# i) (Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeWord8Array# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeWord8Array# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexWord8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X16OffAddr :: Addr -> Int -> Word8X16
indexWord8X16OffAddr (Addr a) (I# i) = Word8X16 (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 1#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 3#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 5#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 7#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 9#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 11#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 13#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 16#) +# 15#)) 0#)

{-# INLINE readWord8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X16OffAddr :: PrimMonad m => Addr -> Int -> m Word8X16
readWord8X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeWord8X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X16OffAddr :: PrimMonad m => Addr -> Int -> Word8X16 -> m ()
writeWord8X16OffAddr (Addr a) (I# i) (Word8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 1#)) 0# m2) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0# m3) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 3#)) 0# m4) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m5) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 5#)) 0# m6) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0# m7) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 7#)) 0# m8) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m9) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 9#)) 0# m10) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0# m11) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 11#)) 0# m12) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m13) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 13#)) 0# m14) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0# m15) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 16#) +# 15#)) 0# m16)


