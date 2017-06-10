{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X16 (Int16X16) where

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

-- ** Int16X16
data Int16X16 = Int16X16 Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

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

instance Eq Int16X16 where
    a == b = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int16X16 where
    a `compare` b = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int16X16 where
    showsPrec _ a s = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int16X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int16X16 where
    (+) = plusInt16X16
    (-) = minusInt16X16
    (*) = timesInt16X16
    negate = negateInt16X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X16 where
    type Elem Int16X16 = Int16
    type ElemTuple Int16X16 = (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 2
    broadcastVector    = broadcastInt16X16
    generateVector     = generateInt16X16
    unsafeInsertVector = unsafeInsertInt16X16
    packVector         = packInt16X16
    unpackVector       = unpackInt16X16
    mapVector          = mapInt16X16
    zipVector          = zipInt16X16
    foldVector         = foldInt16X16

instance SIMDIntVector Int16X16 where
    quotVector = quotInt16X16
    remVector  = remInt16X16

instance Prim Int16X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X16 = V_Int16X16 (PV.Vector Int16X16)
newtype instance UV.MVector s Int16X16 = MV_Int16X16 (PMV.MVector s Int16X16)

instance Vector UV.Vector Int16X16 where
    basicUnsafeFreeze (MV_Int16X16 v) = V_Int16X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X16 v) = MV_Int16X16 <$> PV.unsafeThaw v
    basicLength (V_Int16X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X16 v) = V_Int16X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X16 m) (V_Int16X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X16 where
    basicLength (MV_Int16X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X16 v) = MV_Int16X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X16 v) (MV_Int16X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X16

{-# INLINE broadcastInt16X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X16 :: Int16 -> Int16X16
broadcastInt16X16 (I16# x) = case broadcastInt16# x of
    v -> Int16X16 v v v v v v v v v v v v v v v v

{-# INLINE[1] generateInt16X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt16X16 :: (Int -> Int16) -> Int16X16
generateInt16X16 f = packInt16X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packInt16X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X16 :: (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16) -> Int16X16
packInt16X16 (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16) = Int16X16 (packInt16# (# x1 #)) (packInt16# (# x2 #)) (packInt16# (# x3 #)) (packInt16# (# x4 #)) (packInt16# (# x5 #)) (packInt16# (# x6 #)) (packInt16# (# x7 #)) (packInt16# (# x8 #)) (packInt16# (# x9 #)) (packInt16# (# x10 #)) (packInt16# (# x11 #)) (packInt16# (# x12 #)) (packInt16# (# x13 #)) (packInt16# (# x14 #)) (packInt16# (# x15 #)) (packInt16# (# x16 #))

{-# INLINE unpackInt16X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X16 :: Int16X16 -> (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
unpackInt16X16 (Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackInt16# m1 of
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
                                                                (# x16 #) -> (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16)

{-# INLINE unsafeInsertInt16X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X16 :: Int16X16 -> Int16 -> Int -> Int16X16
unsafeInsertInt16X16 (Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (I16# y) _i@(I# ip) | _i < 1 = Int16X16 (insertInt16# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 2 = Int16X16 m1 (insertInt16# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 3 = Int16X16 m1 m2 (insertInt16# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 4 = Int16X16 m1 m2 m3 (insertInt16# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 5 = Int16X16 m1 m2 m3 m4 (insertInt16# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 6 = Int16X16 m1 m2 m3 m4 m5 (insertInt16# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 7 = Int16X16 m1 m2 m3 m4 m5 m6 (insertInt16# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 8 = Int16X16 m1 m2 m3 m4 m5 m6 m7 (insertInt16# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 9 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 (insertInt16# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 10 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertInt16# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                           | _i < 11 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertInt16# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                           | _i < 12 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertInt16# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                           | _i < 13 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertInt16# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                           | _i < 14 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertInt16# m14 y (ip -# 13#)) m15 m16
                                                                                                           | _i < 15 = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertInt16# m15 y (ip -# 14#)) m16
                                                                                                           | otherwise = Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertInt16# m16 y (ip -# 15#))

{-# INLINE[1] mapInt16X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X16 :: (Int16 -> Int16) -> Int16X16 -> Int16X16
mapInt16X16 f = mapInt16X16# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X16# #-}
-- | Unboxed helper function.
mapInt16X16# :: (Int# -> Int#) -> Int16X16 -> Int16X16
mapInt16X16# f = \ v -> case unpackInt16X16 v of
    (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16) -> packInt16X16 (I16# (f x1), I16# (f x2), I16# (f x3), I16# (f x4), I16# (f x5), I16# (f x6), I16# (f x7), I16# (f x8), I16# (f x9), I16# (f x10), I16# (f x11), I16# (f x12), I16# (f x13), I16# (f x14), I16# (f x15), I16# (f x16))

{-# INLINE[1] zipInt16X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X16 :: (Int16 -> Int16 -> Int16) -> Int16X16 -> Int16X16 -> Int16X16
zipInt16X16 f = \ v1 v2 -> case unpackInt16X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt16X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt16X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X16 #-}
-- | Fold the elements of a vector to a single value
foldInt16X16 :: (Int16 -> Int16 -> Int16) -> Int16X16 -> Int16
foldInt16X16 f' = \ v -> case unpackInt16X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusInt16X16 #-}
-- | Add two vectors element-wise.
plusInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
plusInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int16X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int16X16 (plusInt16# m1_1 m1_2) (plusInt16# m2_1 m2_2) (plusInt16# m3_1 m3_2) (plusInt16# m4_1 m4_2) (plusInt16# m5_1 m5_2) (plusInt16# m6_1 m6_2) (plusInt16# m7_1 m7_2) (plusInt16# m8_1 m8_2) (plusInt16# m9_1 m9_2) (plusInt16# m10_1 m10_2) (plusInt16# m11_1 m11_2) (plusInt16# m12_1 m12_2) (plusInt16# m13_1 m13_2) (plusInt16# m14_1 m14_2) (plusInt16# m15_1 m15_2) (plusInt16# m16_1 m16_2)

{-# INLINE minusInt16X16 #-}
-- | Subtract two vectors element-wise.
minusInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
minusInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int16X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int16X16 (minusInt16# m1_1 m1_2) (minusInt16# m2_1 m2_2) (minusInt16# m3_1 m3_2) (minusInt16# m4_1 m4_2) (minusInt16# m5_1 m5_2) (minusInt16# m6_1 m6_2) (minusInt16# m7_1 m7_2) (minusInt16# m8_1 m8_2) (minusInt16# m9_1 m9_2) (minusInt16# m10_1 m10_2) (minusInt16# m11_1 m11_2) (minusInt16# m12_1 m12_2) (minusInt16# m13_1 m13_2) (minusInt16# m14_1 m14_2) (minusInt16# m15_1 m15_2) (minusInt16# m16_1 m16_2)

{-# INLINE timesInt16X16 #-}
-- | Multiply two vectors element-wise.
timesInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
timesInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int16X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int16X16 (timesInt16# m1_1 m1_2) (timesInt16# m2_1 m2_2) (timesInt16# m3_1 m3_2) (timesInt16# m4_1 m4_2) (timesInt16# m5_1 m5_2) (timesInt16# m6_1 m6_2) (timesInt16# m7_1 m7_2) (timesInt16# m8_1 m8_2) (timesInt16# m9_1 m9_2) (timesInt16# m10_1 m10_2) (timesInt16# m11_1 m11_2) (timesInt16# m12_1 m12_2) (timesInt16# m13_1 m13_2) (timesInt16# m14_1 m14_2) (timesInt16# m15_1 m15_2) (timesInt16# m16_1 m16_2)

{-# INLINE quotInt16X16 #-}
-- | Rounds towards zero element-wise.
quotInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
quotInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int16X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int16X16 (quotInt16# m1_1 m1_2) (quotInt16# m2_1 m2_2) (quotInt16# m3_1 m3_2) (quotInt16# m4_1 m4_2) (quotInt16# m5_1 m5_2) (quotInt16# m6_1 m6_2) (quotInt16# m7_1 m7_2) (quotInt16# m8_1 m8_2) (quotInt16# m9_1 m9_2) (quotInt16# m10_1 m10_2) (quotInt16# m11_1 m11_2) (quotInt16# m12_1 m12_2) (quotInt16# m13_1 m13_2) (quotInt16# m14_1 m14_2) (quotInt16# m15_1 m15_2) (quotInt16# m16_1 m16_2)

{-# INLINE remInt16X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
remInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int16X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int16X16 (remInt16# m1_1 m1_2) (remInt16# m2_1 m2_2) (remInt16# m3_1 m3_2) (remInt16# m4_1 m4_2) (remInt16# m5_1 m5_2) (remInt16# m6_1 m6_2) (remInt16# m7_1 m7_2) (remInt16# m8_1 m8_2) (remInt16# m9_1 m9_2) (remInt16# m10_1 m10_2) (remInt16# m11_1 m11_2) (remInt16# m12_1 m12_2) (remInt16# m13_1 m13_2) (remInt16# m14_1 m14_2) (remInt16# m15_1 m15_2) (remInt16# m16_1 m16_2)

{-# INLINE negateInt16X16 #-}
-- | Negate element-wise.
negateInt16X16 :: Int16X16 -> Int16X16
negateInt16X16 (Int16X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) = Int16X16 (negateInt16# m1_1) (negateInt16# m2_1) (negateInt16# m3_1) (negateInt16# m4_1) (negateInt16# m5_1) (negateInt16# m6_1) (negateInt16# m7_1) (negateInt16# m8_1) (negateInt16# m9_1) (negateInt16# m10_1) (negateInt16# m11_1) (negateInt16# m12_1) (negateInt16# m13_1) (negateInt16# m14_1) (negateInt16# m15_1) (negateInt16# m16_1)

{-# INLINE indexInt16X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X16Array :: ByteArray -> Int -> Int16X16
indexInt16X16Array (ByteArray a) (I# i) = Int16X16 (indexInt16Array# a ((i *# 16#) +# 0#)) (indexInt16Array# a ((i *# 16#) +# 1#)) (indexInt16Array# a ((i *# 16#) +# 2#)) (indexInt16Array# a ((i *# 16#) +# 3#)) (indexInt16Array# a ((i *# 16#) +# 4#)) (indexInt16Array# a ((i *# 16#) +# 5#)) (indexInt16Array# a ((i *# 16#) +# 6#)) (indexInt16Array# a ((i *# 16#) +# 7#)) (indexInt16Array# a ((i *# 16#) +# 8#)) (indexInt16Array# a ((i *# 16#) +# 9#)) (indexInt16Array# a ((i *# 16#) +# 10#)) (indexInt16Array# a ((i *# 16#) +# 11#)) (indexInt16Array# a ((i *# 16#) +# 12#)) (indexInt16Array# a ((i *# 16#) +# 13#)) (indexInt16Array# a ((i *# 16#) +# 14#)) (indexInt16Array# a ((i *# 16#) +# 15#))

{-# INLINE readInt16X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X16
readInt16X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16Array# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt16Array# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt16Array# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt16Array# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt16Array# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt16Array# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt16Array# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt16Array# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readInt16Array# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readInt16Array# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readInt16Array# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readInt16Array# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readInt16Array# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readInt16Array# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readInt16Array# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readInt16Array# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt16X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X16 -> m ()
writeInt16X16Array (MutableByteArray a) (I# i) (Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt16Array# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeInt16Array# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexInt16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X16OffAddr :: Addr -> Int -> Int16X16
indexInt16X16OffAddr (Addr a) (I# i) = Int16X16 (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0#)

{-# INLINE readInt16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X16OffAddr :: PrimMonad m => Addr -> Int -> m Int16X16
readInt16X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 6#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 10#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 14#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 18#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 22#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 26#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 30#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt16X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X16OffAddr :: PrimMonad m => Addr -> Int -> Int16X16 -> m ()
writeInt16X16OffAddr (Addr a) (I# i) (Int16X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 2#)) 0# m2) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m3) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 6#)) 0# m4) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m5) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 10#)) 0# m6) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m7) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 14#)) 0# m8) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m9) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 18#)) 0# m10) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m11) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 22#)) 0# m12) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m13) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 26#)) 0# m14) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m15) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 32#) +# 30#)) 0# m16)


