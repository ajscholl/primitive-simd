{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int32X16 (Int32X16) where

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

-- ** Int32X16
data Int32X16 = Int32X16 Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt32# :: Int# -> Int#
broadcastInt32# v = v

packInt32# :: (# Int# #) -> Int#
packInt32# (# v #) = v

unpackInt32# :: Int# -> (# Int# #)
unpackInt32# v = (# v #)

insertInt32# :: Int# -> Int# -> Int# -> Int#
insertInt32# _ v _ = v

negateInt32# :: Int# -> Int#
negateInt32# a = case negate (I32# a) of I32# b -> b

plusInt32# :: Int# -> Int# -> Int#
plusInt32# a b = case I32# a + I32# b of I32# c -> c

minusInt32# :: Int# -> Int# -> Int#
minusInt32# a b = case I32# a - I32# b of I32# c -> c

timesInt32# :: Int# -> Int# -> Int#
timesInt32# a b = case I32# a * I32# b of I32# c -> c

quotInt32# :: Int# -> Int# -> Int#
quotInt32# a b = case I32# a `quot` I32# b of I32# c -> c

remInt32# :: Int# -> Int# -> Int#
remInt32# a b = case I32# a `rem` I32# b of I32# c -> c

abs' :: Int32 -> Int32
abs' (I32# x) = I32# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I32# x) of
    I32# y -> y

signum' :: Int32 -> Int32
signum' (I32# x) = I32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I32# x) of
    I32# y -> y

instance Eq Int32X16 where
    a == b = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int32X16 where
    a `compare` b = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int32X16 where
    showsPrec _ a s = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int32X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int32X16 where
    (+) = plusInt32X16
    (-) = minusInt32X16
    (*) = timesInt32X16
    negate = negateInt32X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int32X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int32X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int32X16 where
    type Elem Int32X16 = Int32
    type ElemTuple Int32X16 = (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 4
    broadcastVector    = broadcastInt32X16
    generateVector     = generateInt32X16
    unsafeInsertVector = unsafeInsertInt32X16
    packVector         = packInt32X16
    unpackVector       = unpackInt32X16
    mapVector          = mapInt32X16
    zipVector          = zipInt32X16
    foldVector         = foldInt32X16

instance SIMDIntVector Int32X16 where
    quotVector = quotInt32X16
    remVector  = remInt32X16

instance Prim Int32X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt32X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt32X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt32X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt32X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt32X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt32X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int32X16 = V_Int32X16 (PV.Vector Int32X16)
newtype instance UV.MVector s Int32X16 = MV_Int32X16 (PMV.MVector s Int32X16)

instance Vector UV.Vector Int32X16 where
    basicUnsafeFreeze (MV_Int32X16 v) = V_Int32X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int32X16 v) = MV_Int32X16 <$> PV.unsafeThaw v
    basicLength (V_Int32X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int32X16 v) = V_Int32X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int32X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int32X16 m) (V_Int32X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int32X16 where
    basicLength (MV_Int32X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int32X16 v) = MV_Int32X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int32X16 v) (MV_Int32X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int32X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int32X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int32X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int32X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int32X16

{-# INLINE broadcastInt32X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt32X16 :: Int32 -> Int32X16
broadcastInt32X16 (I32# x) = case broadcastInt32# x of
    v -> Int32X16 v v v v v v v v v v v v v v v v

{-# INLINE[1] generateInt32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt32X16 :: (Int -> Int32) -> Int32X16
generateInt32X16 f = packInt32X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packInt32X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X16 :: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) -> Int32X16
packInt32X16 (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16) = Int32X16 (packInt32# (# x1 #)) (packInt32# (# x2 #)) (packInt32# (# x3 #)) (packInt32# (# x4 #)) (packInt32# (# x5 #)) (packInt32# (# x6 #)) (packInt32# (# x7 #)) (packInt32# (# x8 #)) (packInt32# (# x9 #)) (packInt32# (# x10 #)) (packInt32# (# x11 #)) (packInt32# (# x12 #)) (packInt32# (# x13 #)) (packInt32# (# x14 #)) (packInt32# (# x15 #)) (packInt32# (# x16 #))

{-# INLINE unpackInt32X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X16 :: Int32X16 -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
unpackInt32X16 (Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackInt32# m1 of
    (# x1 #) -> case unpackInt32# m2 of
        (# x2 #) -> case unpackInt32# m3 of
            (# x3 #) -> case unpackInt32# m4 of
                (# x4 #) -> case unpackInt32# m5 of
                    (# x5 #) -> case unpackInt32# m6 of
                        (# x6 #) -> case unpackInt32# m7 of
                            (# x7 #) -> case unpackInt32# m8 of
                                (# x8 #) -> case unpackInt32# m9 of
                                    (# x9 #) -> case unpackInt32# m10 of
                                        (# x10 #) -> case unpackInt32# m11 of
                                            (# x11 #) -> case unpackInt32# m12 of
                                                (# x12 #) -> case unpackInt32# m13 of
                                                    (# x13 #) -> case unpackInt32# m14 of
                                                        (# x14 #) -> case unpackInt32# m15 of
                                                            (# x15 #) -> case unpackInt32# m16 of
                                                                (# x16 #) -> (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16)

{-# INLINE unsafeInsertInt32X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X16 :: Int32X16 -> Int32 -> Int -> Int32X16
unsafeInsertInt32X16 (Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (I32# y) _i@(I# ip) | _i < 1 = Int32X16 (insertInt32# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 2 = Int32X16 m1 (insertInt32# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 3 = Int32X16 m1 m2 (insertInt32# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 4 = Int32X16 m1 m2 m3 (insertInt32# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 5 = Int32X16 m1 m2 m3 m4 (insertInt32# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 6 = Int32X16 m1 m2 m3 m4 m5 (insertInt32# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 7 = Int32X16 m1 m2 m3 m4 m5 m6 (insertInt32# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 8 = Int32X16 m1 m2 m3 m4 m5 m6 m7 (insertInt32# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 9 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 (insertInt32# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 10 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertInt32# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 11 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertInt32# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                           | _i < 12 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertInt32# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                           | _i < 13 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertInt32# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                           | _i < 14 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertInt32# m14 y (ip -# 13#)) m15 m16
                                                                                                           | _i < 15 = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertInt32# m15 y (ip -# 14#)) m16
                                                                                                           | otherwise = Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertInt32# m16 y (ip -# 15#))

{-# INLINE[1] mapInt32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt32X16 :: (Int32 -> Int32) -> Int32X16 -> Int32X16
mapInt32X16 f = mapInt32X16# (\ x -> case f (I32# x) of { I32# y -> y})

{-# RULES "mapVector abs" mapInt32X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt32X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt32X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt32X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt32X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt32X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt32X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt32X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt32X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt32X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt32X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt32X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt32X16# #-}
-- | Unboxed helper function.
mapInt32X16# :: (Int# -> Int#) -> Int32X16 -> Int32X16
mapInt32X16# f = \ v -> case unpackInt32X16 v of
    (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16) -> packInt32X16 (I32# (f x1), I32# (f x2), I32# (f x3), I32# (f x4), I32# (f x5), I32# (f x6), I32# (f x7), I32# (f x8), I32# (f x9), I32# (f x10), I32# (f x11), I32# (f x12), I32# (f x13), I32# (f x14), I32# (f x15), I32# (f x16))

{-# INLINE[1] zipInt32X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt32X16 :: (Int32 -> Int32 -> Int32) -> Int32X16 -> Int32X16 -> Int32X16
zipInt32X16 f = \ v1 v2 -> case unpackInt32X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt32X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt32X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt32X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt32X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt32X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt32X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt32X16 #-}
-- | Fold the elements of a vector to a single value
foldInt32X16 :: (Int32 -> Int32 -> Int32) -> Int32X16 -> Int32
foldInt32X16 f' = \ v -> case unpackInt32X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusInt32X16 #-}
-- | Add two vectors element-wise.
plusInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
plusInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int32X16 (plusInt32# m1_1 m1_2) (plusInt32# m2_1 m2_2) (plusInt32# m3_1 m3_2) (plusInt32# m4_1 m4_2) (plusInt32# m5_1 m5_2) (plusInt32# m6_1 m6_2) (plusInt32# m7_1 m7_2) (plusInt32# m8_1 m8_2) (plusInt32# m9_1 m9_2) (plusInt32# m10_1 m10_2) (plusInt32# m11_1 m11_2) (plusInt32# m12_1 m12_2) (plusInt32# m13_1 m13_2) (plusInt32# m14_1 m14_2) (plusInt32# m15_1 m15_2) (plusInt32# m16_1 m16_2)

{-# INLINE minusInt32X16 #-}
-- | Subtract two vectors element-wise.
minusInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
minusInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int32X16 (minusInt32# m1_1 m1_2) (minusInt32# m2_1 m2_2) (minusInt32# m3_1 m3_2) (minusInt32# m4_1 m4_2) (minusInt32# m5_1 m5_2) (minusInt32# m6_1 m6_2) (minusInt32# m7_1 m7_2) (minusInt32# m8_1 m8_2) (minusInt32# m9_1 m9_2) (minusInt32# m10_1 m10_2) (minusInt32# m11_1 m11_2) (minusInt32# m12_1 m12_2) (minusInt32# m13_1 m13_2) (minusInt32# m14_1 m14_2) (minusInt32# m15_1 m15_2) (minusInt32# m16_1 m16_2)

{-# INLINE timesInt32X16 #-}
-- | Multiply two vectors element-wise.
timesInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
timesInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int32X16 (timesInt32# m1_1 m1_2) (timesInt32# m2_1 m2_2) (timesInt32# m3_1 m3_2) (timesInt32# m4_1 m4_2) (timesInt32# m5_1 m5_2) (timesInt32# m6_1 m6_2) (timesInt32# m7_1 m7_2) (timesInt32# m8_1 m8_2) (timesInt32# m9_1 m9_2) (timesInt32# m10_1 m10_2) (timesInt32# m11_1 m11_2) (timesInt32# m12_1 m12_2) (timesInt32# m13_1 m13_2) (timesInt32# m14_1 m14_2) (timesInt32# m15_1 m15_2) (timesInt32# m16_1 m16_2)

{-# INLINE quotInt32X16 #-}
-- | Rounds towards zero element-wise.
quotInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
quotInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int32X16 (quotInt32# m1_1 m1_2) (quotInt32# m2_1 m2_2) (quotInt32# m3_1 m3_2) (quotInt32# m4_1 m4_2) (quotInt32# m5_1 m5_2) (quotInt32# m6_1 m6_2) (quotInt32# m7_1 m7_2) (quotInt32# m8_1 m8_2) (quotInt32# m9_1 m9_2) (quotInt32# m10_1 m10_2) (quotInt32# m11_1 m11_2) (quotInt32# m12_1 m12_2) (quotInt32# m13_1 m13_2) (quotInt32# m14_1 m14_2) (quotInt32# m15_1 m15_2) (quotInt32# m16_1 m16_2)

{-# INLINE remInt32X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
remInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int32X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int32X16 (remInt32# m1_1 m1_2) (remInt32# m2_1 m2_2) (remInt32# m3_1 m3_2) (remInt32# m4_1 m4_2) (remInt32# m5_1 m5_2) (remInt32# m6_1 m6_2) (remInt32# m7_1 m7_2) (remInt32# m8_1 m8_2) (remInt32# m9_1 m9_2) (remInt32# m10_1 m10_2) (remInt32# m11_1 m11_2) (remInt32# m12_1 m12_2) (remInt32# m13_1 m13_2) (remInt32# m14_1 m14_2) (remInt32# m15_1 m15_2) (remInt32# m16_1 m16_2)

{-# INLINE negateInt32X16 #-}
-- | Negate element-wise.
negateInt32X16 :: Int32X16 -> Int32X16
negateInt32X16 (Int32X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) = Int32X16 (negateInt32# m1_1) (negateInt32# m2_1) (negateInt32# m3_1) (negateInt32# m4_1) (negateInt32# m5_1) (negateInt32# m6_1) (negateInt32# m7_1) (negateInt32# m8_1) (negateInt32# m9_1) (negateInt32# m10_1) (negateInt32# m11_1) (negateInt32# m12_1) (negateInt32# m13_1) (negateInt32# m14_1) (negateInt32# m15_1) (negateInt32# m16_1)

{-# INLINE indexInt32X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X16Array :: ByteArray -> Int -> Int32X16
indexInt32X16Array (ByteArray a) (I# i) = Int32X16 (indexInt32Array# a ((i *# 16#) +# 0#)) (indexInt32Array# a ((i *# 16#) +# 1#)) (indexInt32Array# a ((i *# 16#) +# 2#)) (indexInt32Array# a ((i *# 16#) +# 3#)) (indexInt32Array# a ((i *# 16#) +# 4#)) (indexInt32Array# a ((i *# 16#) +# 5#)) (indexInt32Array# a ((i *# 16#) +# 6#)) (indexInt32Array# a ((i *# 16#) +# 7#)) (indexInt32Array# a ((i *# 16#) +# 8#)) (indexInt32Array# a ((i *# 16#) +# 9#)) (indexInt32Array# a ((i *# 16#) +# 10#)) (indexInt32Array# a ((i *# 16#) +# 11#)) (indexInt32Array# a ((i *# 16#) +# 12#)) (indexInt32Array# a ((i *# 16#) +# 13#)) (indexInt32Array# a ((i *# 16#) +# 14#)) (indexInt32Array# a ((i *# 16#) +# 15#))

{-# INLINE readInt32X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X16
readInt32X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32Array# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt32Array# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt32Array# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt32Array# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt32Array# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt32Array# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt32Array# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt32Array# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readInt32Array# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readInt32Array# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readInt32Array# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readInt32Array# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readInt32Array# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readInt32Array# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readInt32Array# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readInt32Array# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt32X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X16 -> m ()
writeInt32X16Array (MutableByteArray a) (I# i) (Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt32Array# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeInt32Array# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexInt32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X16OffAddr :: Addr -> Int -> Int32X16
indexInt32X16OffAddr (Addr a) (I# i) = Int32X16 (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0#)

{-# INLINE readInt32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X16OffAddr :: PrimMonad m => Addr -> Int -> m Int32X16
readInt32X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 28#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 36#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 44#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 52#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 60#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt32X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X16OffAddr :: PrimMonad m => Addr -> Int -> Int32X16 -> m ()
writeInt32X16OffAddr (Addr a) (I# i) (Int32X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0# m2) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m3) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0# m4) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m5) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0# m6) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m7) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0# m8) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m9) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0# m10) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m11) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0# m12) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m13) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0# m14) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m15) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0# m16)


