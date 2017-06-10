{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X16 (Int8X16) where

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

-- ** Int8X16
data Int8X16 = Int8X16 Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

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

instance Eq Int8X16 where
    a == b = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int8X16 where
    a `compare` b = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int8X16 where
    showsPrec _ a s = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int8X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int8X16 where
    (+) = plusInt8X16
    (-) = minusInt8X16
    (*) = timesInt8X16
    negate = negateInt8X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X16 where
    type Elem Int8X16 = Int8
    type ElemTuple Int8X16 = (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 1
    broadcastVector    = broadcastInt8X16
    generateVector     = generateInt8X16
    unsafeInsertVector = unsafeInsertInt8X16
    packVector         = packInt8X16
    unpackVector       = unpackInt8X16
    mapVector          = mapInt8X16
    zipVector          = zipInt8X16
    foldVector         = foldInt8X16

instance SIMDIntVector Int8X16 where
    quotVector = quotInt8X16
    remVector  = remInt8X16

instance Prim Int8X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X16 = V_Int8X16 (PV.Vector Int8X16)
newtype instance UV.MVector s Int8X16 = MV_Int8X16 (PMV.MVector s Int8X16)

instance Vector UV.Vector Int8X16 where
    basicUnsafeFreeze (MV_Int8X16 v) = V_Int8X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X16 v) = MV_Int8X16 <$> PV.unsafeThaw v
    basicLength (V_Int8X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X16 v) = V_Int8X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X16 m) (V_Int8X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X16 where
    basicLength (MV_Int8X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X16 v) = MV_Int8X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X16 v) (MV_Int8X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X16

{-# INLINE broadcastInt8X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X16 :: Int8 -> Int8X16
broadcastInt8X16 (I8# x) = case broadcastInt8# x of
    v -> Int8X16 v v v v v v v v v v v v v v v v

{-# INLINE[1] generateInt8X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt8X16 :: (Int -> Int8) -> Int8X16
generateInt8X16 f = packInt8X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packInt8X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X16 :: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) -> Int8X16
packInt8X16 (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16) = Int8X16 (packInt8# (# x1 #)) (packInt8# (# x2 #)) (packInt8# (# x3 #)) (packInt8# (# x4 #)) (packInt8# (# x5 #)) (packInt8# (# x6 #)) (packInt8# (# x7 #)) (packInt8# (# x8 #)) (packInt8# (# x9 #)) (packInt8# (# x10 #)) (packInt8# (# x11 #)) (packInt8# (# x12 #)) (packInt8# (# x13 #)) (packInt8# (# x14 #)) (packInt8# (# x15 #)) (packInt8# (# x16 #))

{-# INLINE unpackInt8X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X16 :: Int8X16 -> (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
unpackInt8X16 (Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackInt8# m1 of
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
                                                                (# x16 #) -> (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16)

{-# INLINE unsafeInsertInt8X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X16 :: Int8X16 -> Int8 -> Int -> Int8X16
unsafeInsertInt8X16 (Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (I8# y) _i@(I# ip) | _i < 1 = Int8X16 (insertInt8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 2 = Int8X16 m1 (insertInt8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 3 = Int8X16 m1 m2 (insertInt8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 4 = Int8X16 m1 m2 m3 (insertInt8# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 5 = Int8X16 m1 m2 m3 m4 (insertInt8# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 6 = Int8X16 m1 m2 m3 m4 m5 (insertInt8# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 7 = Int8X16 m1 m2 m3 m4 m5 m6 (insertInt8# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 8 = Int8X16 m1 m2 m3 m4 m5 m6 m7 (insertInt8# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 9 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 (insertInt8# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 10 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertInt8# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                        | _i < 11 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertInt8# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                        | _i < 12 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertInt8# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                        | _i < 13 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertInt8# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                        | _i < 14 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertInt8# m14 y (ip -# 13#)) m15 m16
                                                                                                        | _i < 15 = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertInt8# m15 y (ip -# 14#)) m16
                                                                                                        | otherwise = Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertInt8# m16 y (ip -# 15#))

{-# INLINE[1] mapInt8X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X16 :: (Int8 -> Int8) -> Int8X16 -> Int8X16
mapInt8X16 f = mapInt8X16# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X16# #-}
-- | Unboxed helper function.
mapInt8X16# :: (Int# -> Int#) -> Int8X16 -> Int8X16
mapInt8X16# f = \ v -> case unpackInt8X16 v of
    (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16) -> packInt8X16 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4), I8# (f x5), I8# (f x6), I8# (f x7), I8# (f x8), I8# (f x9), I8# (f x10), I8# (f x11), I8# (f x12), I8# (f x13), I8# (f x14), I8# (f x15), I8# (f x16))

{-# INLINE[1] zipInt8X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X16 :: (Int8 -> Int8 -> Int8) -> Int8X16 -> Int8X16 -> Int8X16
zipInt8X16 f = \ v1 v2 -> case unpackInt8X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt8X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt8X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X16 #-}
-- | Fold the elements of a vector to a single value
foldInt8X16 :: (Int8 -> Int8 -> Int8) -> Int8X16 -> Int8
foldInt8X16 f' = \ v -> case unpackInt8X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusInt8X16 #-}
-- | Add two vectors element-wise.
plusInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
plusInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int8X16 (plusInt8# m1_1 m1_2) (plusInt8# m2_1 m2_2) (plusInt8# m3_1 m3_2) (plusInt8# m4_1 m4_2) (plusInt8# m5_1 m5_2) (plusInt8# m6_1 m6_2) (plusInt8# m7_1 m7_2) (plusInt8# m8_1 m8_2) (plusInt8# m9_1 m9_2) (plusInt8# m10_1 m10_2) (plusInt8# m11_1 m11_2) (plusInt8# m12_1 m12_2) (plusInt8# m13_1 m13_2) (plusInt8# m14_1 m14_2) (plusInt8# m15_1 m15_2) (plusInt8# m16_1 m16_2)

{-# INLINE minusInt8X16 #-}
-- | Subtract two vectors element-wise.
minusInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
minusInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int8X16 (minusInt8# m1_1 m1_2) (minusInt8# m2_1 m2_2) (minusInt8# m3_1 m3_2) (minusInt8# m4_1 m4_2) (minusInt8# m5_1 m5_2) (minusInt8# m6_1 m6_2) (minusInt8# m7_1 m7_2) (minusInt8# m8_1 m8_2) (minusInt8# m9_1 m9_2) (minusInt8# m10_1 m10_2) (minusInt8# m11_1 m11_2) (minusInt8# m12_1 m12_2) (minusInt8# m13_1 m13_2) (minusInt8# m14_1 m14_2) (minusInt8# m15_1 m15_2) (minusInt8# m16_1 m16_2)

{-# INLINE timesInt8X16 #-}
-- | Multiply two vectors element-wise.
timesInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
timesInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int8X16 (timesInt8# m1_1 m1_2) (timesInt8# m2_1 m2_2) (timesInt8# m3_1 m3_2) (timesInt8# m4_1 m4_2) (timesInt8# m5_1 m5_2) (timesInt8# m6_1 m6_2) (timesInt8# m7_1 m7_2) (timesInt8# m8_1 m8_2) (timesInt8# m9_1 m9_2) (timesInt8# m10_1 m10_2) (timesInt8# m11_1 m11_2) (timesInt8# m12_1 m12_2) (timesInt8# m13_1 m13_2) (timesInt8# m14_1 m14_2) (timesInt8# m15_1 m15_2) (timesInt8# m16_1 m16_2)

{-# INLINE quotInt8X16 #-}
-- | Rounds towards zero element-wise.
quotInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
quotInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int8X16 (quotInt8# m1_1 m1_2) (quotInt8# m2_1 m2_2) (quotInt8# m3_1 m3_2) (quotInt8# m4_1 m4_2) (quotInt8# m5_1 m5_2) (quotInt8# m6_1 m6_2) (quotInt8# m7_1 m7_2) (quotInt8# m8_1 m8_2) (quotInt8# m9_1 m9_2) (quotInt8# m10_1 m10_2) (quotInt8# m11_1 m11_2) (quotInt8# m12_1 m12_2) (quotInt8# m13_1 m13_2) (quotInt8# m14_1 m14_2) (quotInt8# m15_1 m15_2) (quotInt8# m16_1 m16_2)

{-# INLINE remInt8X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
remInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (Int8X16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = Int8X16 (remInt8# m1_1 m1_2) (remInt8# m2_1 m2_2) (remInt8# m3_1 m3_2) (remInt8# m4_1 m4_2) (remInt8# m5_1 m5_2) (remInt8# m6_1 m6_2) (remInt8# m7_1 m7_2) (remInt8# m8_1 m8_2) (remInt8# m9_1 m9_2) (remInt8# m10_1 m10_2) (remInt8# m11_1 m11_2) (remInt8# m12_1 m12_2) (remInt8# m13_1 m13_2) (remInt8# m14_1 m14_2) (remInt8# m15_1 m15_2) (remInt8# m16_1 m16_2)

{-# INLINE negateInt8X16 #-}
-- | Negate element-wise.
negateInt8X16 :: Int8X16 -> Int8X16
negateInt8X16 (Int8X16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) = Int8X16 (negateInt8# m1_1) (negateInt8# m2_1) (negateInt8# m3_1) (negateInt8# m4_1) (negateInt8# m5_1) (negateInt8# m6_1) (negateInt8# m7_1) (negateInt8# m8_1) (negateInt8# m9_1) (negateInt8# m10_1) (negateInt8# m11_1) (negateInt8# m12_1) (negateInt8# m13_1) (negateInt8# m14_1) (negateInt8# m15_1) (negateInt8# m16_1)

{-# INLINE indexInt8X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X16Array :: ByteArray -> Int -> Int8X16
indexInt8X16Array (ByteArray a) (I# i) = Int8X16 (indexInt8Array# a ((i *# 16#) +# 0#)) (indexInt8Array# a ((i *# 16#) +# 1#)) (indexInt8Array# a ((i *# 16#) +# 2#)) (indexInt8Array# a ((i *# 16#) +# 3#)) (indexInt8Array# a ((i *# 16#) +# 4#)) (indexInt8Array# a ((i *# 16#) +# 5#)) (indexInt8Array# a ((i *# 16#) +# 6#)) (indexInt8Array# a ((i *# 16#) +# 7#)) (indexInt8Array# a ((i *# 16#) +# 8#)) (indexInt8Array# a ((i *# 16#) +# 9#)) (indexInt8Array# a ((i *# 16#) +# 10#)) (indexInt8Array# a ((i *# 16#) +# 11#)) (indexInt8Array# a ((i *# 16#) +# 12#)) (indexInt8Array# a ((i *# 16#) +# 13#)) (indexInt8Array# a ((i *# 16#) +# 14#)) (indexInt8Array# a ((i *# 16#) +# 15#))

{-# INLINE readInt8X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X16
readInt8X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8Array# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt8Array# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt8Array# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt8Array# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt8Array# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt8Array# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt8Array# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt8Array# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readInt8Array# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readInt8Array# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readInt8Array# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readInt8Array# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readInt8Array# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readInt8Array# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readInt8Array# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readInt8Array# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt8X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X16 -> m ()
writeInt8X16Array (MutableByteArray a) (I# i) (Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt8Array# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeInt8Array# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexInt8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X16OffAddr :: Addr -> Int -> Int8X16
indexInt8X16OffAddr (Addr a) (I# i) = Int8X16 (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 1#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 3#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 5#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 7#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 9#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 11#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 13#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 16#) +# 15#)) 0#)

{-# INLINE readInt8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X16OffAddr :: PrimMonad m => Addr -> Int -> m Int8X16
readInt8X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeInt8X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X16OffAddr :: PrimMonad m => Addr -> Int -> Int8X16 -> m ()
writeInt8X16OffAddr (Addr a) (I# i) (Int8X16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 1#)) 0# m2) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0# m3) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 3#)) 0# m4) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m5) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 5#)) 0# m6) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0# m7) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 7#)) 0# m8) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m9) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 9#)) 0# m10) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0# m11) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 11#)) 0# m12) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m13) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 13#)) 0# m14) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0# m15) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 16#) +# 15#)) 0# m16)


