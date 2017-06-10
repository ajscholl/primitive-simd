{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.FloatX16 (FloatX16) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

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

-- ** FloatX16
data FloatX16 = FloatX16 Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# Float# deriving Typeable

broadcastFloat# :: Float# -> Float#
broadcastFloat# v = v

packFloat# :: (# Float# #) -> Float#
packFloat# (# v #) = v

unpackFloat# :: Float# -> (# Float# #)
unpackFloat# v = (# v #)

insertFloat# :: Float# -> Float# -> Int# -> Float#
insertFloat# _ v _ = v

abs' :: Float -> Float
abs' (F# x) = F# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Float# -> Float#
abs# x = case abs (F# x) of
    F# y -> y

signum' :: Float -> Float
signum' (F# x) = F# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Float# -> Float#
signum# x = case signum (F# x) of
    F# y -> y

instance Eq FloatX16 where
    a == b = case unpackFloatX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackFloatX16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord FloatX16 where
    a `compare` b = case unpackFloatX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackFloatX16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show FloatX16 where
    showsPrec _ a s = case unpackFloatX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "FloatX16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num FloatX16 where
    (+) = plusFloatX16
    (-) = minusFloatX16
    (*) = timesFloatX16
    negate = negateFloatX16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional FloatX16 where
    (/)          = divideFloatX16
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating FloatX16 where
    pi           = broadcastVector pi
    exp          = mapVector exp
    sqrt         = mapVector sqrt
    log          = mapVector log
    (**)         = zipVector (**)
    logBase      = zipVector (**)
    sin          = mapVector sin 
    tan          = mapVector tan
    cos          = mapVector cos 
    asin         = mapVector asin
    atan         = mapVector atan 
    acos         = mapVector acos
    sinh         = mapVector sinh 
    tanh         = mapVector tanh
    cosh         = mapVector cosh
    asinh        = mapVector asinh
    atanh        = mapVector atanh
    acosh        = mapVector acosh

instance Storable FloatX16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector FloatX16 where
    type Elem FloatX16 = Float
    type ElemTuple FloatX16 = (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 4
    broadcastVector    = broadcastFloatX16
    generateVector     = generateFloatX16
    unsafeInsertVector = unsafeInsertFloatX16
    packVector         = packFloatX16
    unpackVector       = unpackFloatX16
    mapVector          = mapFloatX16
    zipVector          = zipFloatX16
    foldVector         = foldFloatX16

instance Prim FloatX16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexFloatX16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readFloatX16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeFloatX16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexFloatX16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readFloatX16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeFloatX16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector FloatX16 = V_FloatX16 (PV.Vector FloatX16)
newtype instance UV.MVector s FloatX16 = MV_FloatX16 (PMV.MVector s FloatX16)

instance Vector UV.Vector FloatX16 where
    basicUnsafeFreeze (MV_FloatX16 v) = V_FloatX16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_FloatX16 v) = MV_FloatX16 <$> PV.unsafeThaw v
    basicLength (V_FloatX16 v) = PV.length v
    basicUnsafeSlice start len (V_FloatX16 v) = V_FloatX16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_FloatX16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_FloatX16 m) (V_FloatX16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector FloatX16 where
    basicLength (MV_FloatX16 v) = PMV.length v
    basicUnsafeSlice start len (MV_FloatX16 v) = MV_FloatX16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_FloatX16 v) (MV_FloatX16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_FloatX16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_FloatX16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_FloatX16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_FloatX16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox FloatX16

{-# INLINE broadcastFloatX16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastFloatX16 :: Float -> FloatX16
broadcastFloatX16 (F# x) = case broadcastFloat# x of
    v -> FloatX16 v v v v v v v v v v v v v v v v

{-# INLINE[1] generateFloatX16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateFloatX16 :: (Int -> Float) -> FloatX16
generateFloatX16 f = packFloatX16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packFloatX16 #-}
-- | Pack the elements of a tuple into a vector.
packFloatX16 :: (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float) -> FloatX16
packFloatX16 (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8, F# x9, F# x10, F# x11, F# x12, F# x13, F# x14, F# x15, F# x16) = FloatX16 (packFloat# (# x1 #)) (packFloat# (# x2 #)) (packFloat# (# x3 #)) (packFloat# (# x4 #)) (packFloat# (# x5 #)) (packFloat# (# x6 #)) (packFloat# (# x7 #)) (packFloat# (# x8 #)) (packFloat# (# x9 #)) (packFloat# (# x10 #)) (packFloat# (# x11 #)) (packFloat# (# x12 #)) (packFloat# (# x13 #)) (packFloat# (# x14 #)) (packFloat# (# x15 #)) (packFloat# (# x16 #))

{-# INLINE unpackFloatX16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackFloatX16 :: FloatX16 -> (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float)
unpackFloatX16 (FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = case unpackFloat# m1 of
    (# x1 #) -> case unpackFloat# m2 of
        (# x2 #) -> case unpackFloat# m3 of
            (# x3 #) -> case unpackFloat# m4 of
                (# x4 #) -> case unpackFloat# m5 of
                    (# x5 #) -> case unpackFloat# m6 of
                        (# x6 #) -> case unpackFloat# m7 of
                            (# x7 #) -> case unpackFloat# m8 of
                                (# x8 #) -> case unpackFloat# m9 of
                                    (# x9 #) -> case unpackFloat# m10 of
                                        (# x10 #) -> case unpackFloat# m11 of
                                            (# x11 #) -> case unpackFloat# m12 of
                                                (# x12 #) -> case unpackFloat# m13 of
                                                    (# x13 #) -> case unpackFloat# m14 of
                                                        (# x14 #) -> case unpackFloat# m15 of
                                                            (# x15 #) -> case unpackFloat# m16 of
                                                                (# x16 #) -> (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8, F# x9, F# x10, F# x11, F# x12, F# x13, F# x14, F# x15, F# x16)

{-# INLINE unsafeInsertFloatX16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertFloatX16 :: FloatX16 -> Float -> Int -> FloatX16
unsafeInsertFloatX16 (FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) (F# y) _i@(I# ip) | _i < 1 = FloatX16 (insertFloat# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 2 = FloatX16 m1 (insertFloat# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 3 = FloatX16 m1 m2 (insertFloat# m3 y (ip -# 2#)) m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 4 = FloatX16 m1 m2 m3 (insertFloat# m4 y (ip -# 3#)) m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 5 = FloatX16 m1 m2 m3 m4 (insertFloat# m5 y (ip -# 4#)) m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 6 = FloatX16 m1 m2 m3 m4 m5 (insertFloat# m6 y (ip -# 5#)) m7 m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 7 = FloatX16 m1 m2 m3 m4 m5 m6 (insertFloat# m7 y (ip -# 6#)) m8 m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 8 = FloatX16 m1 m2 m3 m4 m5 m6 m7 (insertFloat# m8 y (ip -# 7#)) m9 m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 9 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 (insertFloat# m9 y (ip -# 8#)) m10 m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 10 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 (insertFloat# m10 y (ip -# 9#)) m11 m12 m13 m14 m15 m16
                                                                                                         | _i < 11 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 (insertFloat# m11 y (ip -# 10#)) m12 m13 m14 m15 m16
                                                                                                         | _i < 12 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 (insertFloat# m12 y (ip -# 11#)) m13 m14 m15 m16
                                                                                                         | _i < 13 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 (insertFloat# m13 y (ip -# 12#)) m14 m15 m16
                                                                                                         | _i < 14 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 (insertFloat# m14 y (ip -# 13#)) m15 m16
                                                                                                         | _i < 15 = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 (insertFloat# m15 y (ip -# 14#)) m16
                                                                                                         | otherwise = FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 (insertFloat# m16 y (ip -# 15#))

{-# INLINE[1] mapFloatX16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapFloatX16 :: (Float -> Float) -> FloatX16 -> FloatX16
mapFloatX16 f = mapFloatX16# (\ x -> case f (F# x) of { F# y -> y})

{-# RULES "mapVector abs" mapFloatX16 abs = abs #-}
{-# RULES "mapVector signum" mapFloatX16 signum = signum #-}
{-# RULES "mapVector negate" mapFloatX16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapFloatX16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapFloatX16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapFloatX16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapFloatX16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapFloatX16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapFloatX16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapFloatX16 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapFloatX16 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapFloatX16 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapFloatX16# #-}
-- | Unboxed helper function.
mapFloatX16# :: (Float# -> Float#) -> FloatX16 -> FloatX16
mapFloatX16# f = \ v -> case unpackFloatX16 v of
    (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8, F# x9, F# x10, F# x11, F# x12, F# x13, F# x14, F# x15, F# x16) -> packFloatX16 (F# (f x1), F# (f x2), F# (f x3), F# (f x4), F# (f x5), F# (f x6), F# (f x7), F# (f x8), F# (f x9), F# (f x10), F# (f x11), F# (f x12), F# (f x13), F# (f x14), F# (f x15), F# (f x16))

{-# INLINE[1] zipFloatX16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipFloatX16 :: (Float -> Float -> Float) -> FloatX16 -> FloatX16 -> FloatX16
zipFloatX16 f = \ v1 v2 -> case unpackFloatX16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackFloatX16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packFloatX16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipFloatX16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipFloatX16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipFloatX16 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipFloatX16 (/) a b = a / b #-}

{-# INLINE[1] foldFloatX16 #-}
-- | Fold the elements of a vector to a single value
foldFloatX16 :: (Float -> Float -> Float) -> FloatX16 -> Float
foldFloatX16 f' = \ v -> case unpackFloatX16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusFloatX16 #-}
-- | Add two vectors element-wise.
plusFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
plusFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (FloatX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = FloatX16 (plusFloat# m1_1 m1_2) (plusFloat# m2_1 m2_2) (plusFloat# m3_1 m3_2) (plusFloat# m4_1 m4_2) (plusFloat# m5_1 m5_2) (plusFloat# m6_1 m6_2) (plusFloat# m7_1 m7_2) (plusFloat# m8_1 m8_2) (plusFloat# m9_1 m9_2) (plusFloat# m10_1 m10_2) (plusFloat# m11_1 m11_2) (plusFloat# m12_1 m12_2) (plusFloat# m13_1 m13_2) (plusFloat# m14_1 m14_2) (plusFloat# m15_1 m15_2) (plusFloat# m16_1 m16_2)

{-# INLINE minusFloatX16 #-}
-- | Subtract two vectors element-wise.
minusFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
minusFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (FloatX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = FloatX16 (minusFloat# m1_1 m1_2) (minusFloat# m2_1 m2_2) (minusFloat# m3_1 m3_2) (minusFloat# m4_1 m4_2) (minusFloat# m5_1 m5_2) (minusFloat# m6_1 m6_2) (minusFloat# m7_1 m7_2) (minusFloat# m8_1 m8_2) (minusFloat# m9_1 m9_2) (minusFloat# m10_1 m10_2) (minusFloat# m11_1 m11_2) (minusFloat# m12_1 m12_2) (minusFloat# m13_1 m13_2) (minusFloat# m14_1 m14_2) (minusFloat# m15_1 m15_2) (minusFloat# m16_1 m16_2)

{-# INLINE timesFloatX16 #-}
-- | Multiply two vectors element-wise.
timesFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
timesFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (FloatX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = FloatX16 (timesFloat# m1_1 m1_2) (timesFloat# m2_1 m2_2) (timesFloat# m3_1 m3_2) (timesFloat# m4_1 m4_2) (timesFloat# m5_1 m5_2) (timesFloat# m6_1 m6_2) (timesFloat# m7_1 m7_2) (timesFloat# m8_1 m8_2) (timesFloat# m9_1 m9_2) (timesFloat# m10_1 m10_2) (timesFloat# m11_1 m11_2) (timesFloat# m12_1 m12_2) (timesFloat# m13_1 m13_2) (timesFloat# m14_1 m14_2) (timesFloat# m15_1 m15_2) (timesFloat# m16_1 m16_2)

{-# INLINE divideFloatX16 #-}
-- | Divide two vectors element-wise.
divideFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
divideFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) (FloatX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2 m9_2 m10_2 m11_2 m12_2 m13_2 m14_2 m15_2 m16_2) = FloatX16 (divideFloat# m1_1 m1_2) (divideFloat# m2_1 m2_2) (divideFloat# m3_1 m3_2) (divideFloat# m4_1 m4_2) (divideFloat# m5_1 m5_2) (divideFloat# m6_1 m6_2) (divideFloat# m7_1 m7_2) (divideFloat# m8_1 m8_2) (divideFloat# m9_1 m9_2) (divideFloat# m10_1 m10_2) (divideFloat# m11_1 m11_2) (divideFloat# m12_1 m12_2) (divideFloat# m13_1 m13_2) (divideFloat# m14_1 m14_2) (divideFloat# m15_1 m15_2) (divideFloat# m16_1 m16_2)

{-# INLINE negateFloatX16 #-}
-- | Negate element-wise.
negateFloatX16 :: FloatX16 -> FloatX16
negateFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1 m9_1 m10_1 m11_1 m12_1 m13_1 m14_1 m15_1 m16_1) = FloatX16 (negateFloat# m1_1) (negateFloat# m2_1) (negateFloat# m3_1) (negateFloat# m4_1) (negateFloat# m5_1) (negateFloat# m6_1) (negateFloat# m7_1) (negateFloat# m8_1) (negateFloat# m9_1) (negateFloat# m10_1) (negateFloat# m11_1) (negateFloat# m12_1) (negateFloat# m13_1) (negateFloat# m14_1) (negateFloat# m15_1) (negateFloat# m16_1)

{-# INLINE indexFloatX16Array #-}
-- | Read a vector from specified index of the immutable array.
indexFloatX16Array :: ByteArray -> Int -> FloatX16
indexFloatX16Array (ByteArray a) (I# i) = FloatX16 (indexFloatArray# a ((i *# 16#) +# 0#)) (indexFloatArray# a ((i *# 16#) +# 1#)) (indexFloatArray# a ((i *# 16#) +# 2#)) (indexFloatArray# a ((i *# 16#) +# 3#)) (indexFloatArray# a ((i *# 16#) +# 4#)) (indexFloatArray# a ((i *# 16#) +# 5#)) (indexFloatArray# a ((i *# 16#) +# 6#)) (indexFloatArray# a ((i *# 16#) +# 7#)) (indexFloatArray# a ((i *# 16#) +# 8#)) (indexFloatArray# a ((i *# 16#) +# 9#)) (indexFloatArray# a ((i *# 16#) +# 10#)) (indexFloatArray# a ((i *# 16#) +# 11#)) (indexFloatArray# a ((i *# 16#) +# 12#)) (indexFloatArray# a ((i *# 16#) +# 13#)) (indexFloatArray# a ((i *# 16#) +# 14#)) (indexFloatArray# a ((i *# 16#) +# 15#))

{-# INLINE readFloatX16Array #-}
-- | Read a vector from specified index of the mutable array.
readFloatX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m FloatX16
readFloatX16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readFloatArray# a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case readFloatArray# a ((i *# 16#) +# 1#) s1 of
        (# s2, m2 #) -> case readFloatArray# a ((i *# 16#) +# 2#) s2 of
            (# s3, m3 #) -> case readFloatArray# a ((i *# 16#) +# 3#) s3 of
                (# s4, m4 #) -> case readFloatArray# a ((i *# 16#) +# 4#) s4 of
                    (# s5, m5 #) -> case readFloatArray# a ((i *# 16#) +# 5#) s5 of
                        (# s6, m6 #) -> case readFloatArray# a ((i *# 16#) +# 6#) s6 of
                            (# s7, m7 #) -> case readFloatArray# a ((i *# 16#) +# 7#) s7 of
                                (# s8, m8 #) -> case readFloatArray# a ((i *# 16#) +# 8#) s8 of
                                    (# s9, m9 #) -> case readFloatArray# a ((i *# 16#) +# 9#) s9 of
                                        (# s10, m10 #) -> case readFloatArray# a ((i *# 16#) +# 10#) s10 of
                                            (# s11, m11 #) -> case readFloatArray# a ((i *# 16#) +# 11#) s11 of
                                                (# s12, m12 #) -> case readFloatArray# a ((i *# 16#) +# 12#) s12 of
                                                    (# s13, m13 #) -> case readFloatArray# a ((i *# 16#) +# 13#) s13 of
                                                        (# s14, m14 #) -> case readFloatArray# a ((i *# 16#) +# 14#) s14 of
                                                            (# s15, m15 #) -> case readFloatArray# a ((i *# 16#) +# 15#) s15 of
                                                                (# s16, m16 #) -> (# s16, FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeFloatX16Array #-}
-- | Write a vector to specified index of mutable array.
writeFloatX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> FloatX16 -> m ()
writeFloatX16Array (MutableByteArray a) (I# i) (FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeFloatArray# a ((i *# 16#) +# 0#) m1) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 1#) m2) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 2#) m3) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 3#) m4) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 4#) m5) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 5#) m6) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 6#) m7) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 7#) m8) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 8#) m9) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 9#) m10) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 10#) m11) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 11#) m12) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 12#) m13) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 13#) m14) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 14#) m15) >> primitive_ (writeFloatArray# a ((i *# 16#) +# 15#) m16)

{-# INLINE indexFloatX16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexFloatX16OffAddr :: Addr -> Int -> FloatX16
indexFloatX16OffAddr (Addr a) (I# i) = FloatX16 (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0#)

{-# INLINE readFloatX16OffAddr #-}
-- | Reads vector from the specified index of the address.
readFloatX16OffAddr :: PrimMonad m => Addr -> Int -> m FloatX16
readFloatX16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 28#) s7 of
                                (# s8, m8 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s8 of
                                    (# s9, m9 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 36#) s9 of
                                        (# s10, m10 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 40#) s10 of
                                            (# s11, m11 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 44#) s11 of
                                                (# s12, m12 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s12 of
                                                    (# s13, m13 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 52#) s13 of
                                                        (# s14, m14 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 56#) s14 of
                                                            (# s15, m15 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 60#) s15 of
                                                                (# s16, m16 #) -> (# s16, FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 #))

{-# INLINE writeFloatX16OffAddr #-}
-- | Write vector to the specified index of the address.
writeFloatX16OffAddr :: PrimMonad m => Addr -> Int -> FloatX16 -> m ()
writeFloatX16OffAddr (Addr a) (I# i) (FloatX16 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16) = primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 4#)) 0# m2) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 8#)) 0# m3) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 12#)) 0# m4) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m5) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 20#)) 0# m6) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 24#)) 0# m7) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 28#)) 0# m8) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m9) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 36#)) 0# m10) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 40#)) 0# m11) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 44#)) 0# m12) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m13) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 52#)) 0# m14) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 56#)) 0# m15) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 64#) +# 60#)) 0# m16)


