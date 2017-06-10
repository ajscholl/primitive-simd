{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.DoubleX16 (DoubleX16) where

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

-- ** DoubleX16
data DoubleX16 = DoubleX16 DoubleX2# DoubleX2# DoubleX2# DoubleX2# DoubleX2# DoubleX2# DoubleX2# DoubleX2# deriving Typeable

abs' :: Double -> Double
abs' (D# x) = D# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Double# -> Double#
abs# x = case abs (D# x) of
    D# y -> y

signum' :: Double -> Double
signum' (D# x) = D# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Double# -> Double#
signum# x = case signum (D# x) of
    D# y -> y

instance Eq DoubleX16 where
    a == b = case unpackDoubleX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackDoubleX16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord DoubleX16 where
    a `compare` b = case unpackDoubleX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackDoubleX16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show DoubleX16 where
    showsPrec _ a s = case unpackDoubleX16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "DoubleX16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num DoubleX16 where
    (+) = plusDoubleX16
    (-) = minusDoubleX16
    (*) = timesDoubleX16
    negate = negateDoubleX16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional DoubleX16 where
    (/)          = divideDoubleX16
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating DoubleX16 where
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

instance Storable DoubleX16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector DoubleX16 where
    type Elem DoubleX16 = Double
    type ElemTuple DoubleX16 = (Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 8
    broadcastVector    = broadcastDoubleX16
    generateVector     = generateDoubleX16
    unsafeInsertVector = unsafeInsertDoubleX16
    packVector         = packDoubleX16
    unpackVector       = unpackDoubleX16
    mapVector          = mapDoubleX16
    zipVector          = zipDoubleX16
    foldVector         = foldDoubleX16
    sumVector          = sumDoubleX16

instance Prim DoubleX16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexDoubleX16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readDoubleX16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeDoubleX16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexDoubleX16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readDoubleX16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeDoubleX16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector DoubleX16 = V_DoubleX16 (PV.Vector DoubleX16)
newtype instance UV.MVector s DoubleX16 = MV_DoubleX16 (PMV.MVector s DoubleX16)

instance Vector UV.Vector DoubleX16 where
    basicUnsafeFreeze (MV_DoubleX16 v) = V_DoubleX16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_DoubleX16 v) = MV_DoubleX16 <$> PV.unsafeThaw v
    basicLength (V_DoubleX16 v) = PV.length v
    basicUnsafeSlice start len (V_DoubleX16 v) = V_DoubleX16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_DoubleX16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_DoubleX16 m) (V_DoubleX16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector DoubleX16 where
    basicLength (MV_DoubleX16 v) = PMV.length v
    basicUnsafeSlice start len (MV_DoubleX16 v) = MV_DoubleX16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_DoubleX16 v) (MV_DoubleX16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_DoubleX16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_DoubleX16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_DoubleX16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_DoubleX16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox DoubleX16

{-# INLINE broadcastDoubleX16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastDoubleX16 :: Double -> DoubleX16
broadcastDoubleX16 (D# x) = case broadcastDoubleX2# x of
    v -> DoubleX16 v v v v v v v v

{-# INLINE[1] generateDoubleX16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateDoubleX16 :: (Int -> Double) -> DoubleX16
generateDoubleX16 f = packDoubleX16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packDoubleX16 #-}
-- | Pack the elements of a tuple into a vector.
packDoubleX16 :: (Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double) -> DoubleX16
packDoubleX16 (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8, D# x9, D# x10, D# x11, D# x12, D# x13, D# x14, D# x15, D# x16) = DoubleX16 (packDoubleX2# (# x1, x2 #)) (packDoubleX2# (# x3, x4 #)) (packDoubleX2# (# x5, x6 #)) (packDoubleX2# (# x7, x8 #)) (packDoubleX2# (# x9, x10 #)) (packDoubleX2# (# x11, x12 #)) (packDoubleX2# (# x13, x14 #)) (packDoubleX2# (# x15, x16 #))

{-# INLINE unpackDoubleX16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackDoubleX16 :: DoubleX16 -> (Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double)
unpackDoubleX16 (DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackDoubleX2# m1 of
    (# x1, x2 #) -> case unpackDoubleX2# m2 of
        (# x3, x4 #) -> case unpackDoubleX2# m3 of
            (# x5, x6 #) -> case unpackDoubleX2# m4 of
                (# x7, x8 #) -> case unpackDoubleX2# m5 of
                    (# x9, x10 #) -> case unpackDoubleX2# m6 of
                        (# x11, x12 #) -> case unpackDoubleX2# m7 of
                            (# x13, x14 #) -> case unpackDoubleX2# m8 of
                                (# x15, x16 #) -> (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8, D# x9, D# x10, D# x11, D# x12, D# x13, D# x14, D# x15, D# x16)

{-# INLINE unsafeInsertDoubleX16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertDoubleX16 :: DoubleX16 -> Double -> Int -> DoubleX16
unsafeInsertDoubleX16 (DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8) (D# y) _i@(I# ip) | _i < 2 = DoubleX16 (insertDoubleX2# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                            | _i < 4 = DoubleX16 m1 (insertDoubleX2# m2 y (ip -# 2#)) m3 m4 m5 m6 m7 m8
                                                                            | _i < 6 = DoubleX16 m1 m2 (insertDoubleX2# m3 y (ip -# 4#)) m4 m5 m6 m7 m8
                                                                            | _i < 8 = DoubleX16 m1 m2 m3 (insertDoubleX2# m4 y (ip -# 6#)) m5 m6 m7 m8
                                                                            | _i < 10 = DoubleX16 m1 m2 m3 m4 (insertDoubleX2# m5 y (ip -# 8#)) m6 m7 m8
                                                                            | _i < 12 = DoubleX16 m1 m2 m3 m4 m5 (insertDoubleX2# m6 y (ip -# 10#)) m7 m8
                                                                            | _i < 14 = DoubleX16 m1 m2 m3 m4 m5 m6 (insertDoubleX2# m7 y (ip -# 12#)) m8
                                                                            | otherwise = DoubleX16 m1 m2 m3 m4 m5 m6 m7 (insertDoubleX2# m8 y (ip -# 14#))

{-# INLINE[1] mapDoubleX16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapDoubleX16 :: (Double -> Double) -> DoubleX16 -> DoubleX16
mapDoubleX16 f = mapDoubleX16# (\ x -> case f (D# x) of { D# y -> y})

{-# RULES "mapVector abs" mapDoubleX16 abs = abs #-}
{-# RULES "mapVector signum" mapDoubleX16 signum = signum #-}
{-# RULES "mapVector negate" mapDoubleX16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapDoubleX16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapDoubleX16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapDoubleX16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapDoubleX16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapDoubleX16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapDoubleX16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapDoubleX16 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapDoubleX16 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapDoubleX16 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapDoubleX16# #-}
-- | Unboxed helper function.
mapDoubleX16# :: (Double# -> Double#) -> DoubleX16 -> DoubleX16
mapDoubleX16# f = \ v -> case unpackDoubleX16 v of
    (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8, D# x9, D# x10, D# x11, D# x12, D# x13, D# x14, D# x15, D# x16) -> packDoubleX16 (D# (f x1), D# (f x2), D# (f x3), D# (f x4), D# (f x5), D# (f x6), D# (f x7), D# (f x8), D# (f x9), D# (f x10), D# (f x11), D# (f x12), D# (f x13), D# (f x14), D# (f x15), D# (f x16))

{-# INLINE[1] zipDoubleX16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipDoubleX16 :: (Double -> Double -> Double) -> DoubleX16 -> DoubleX16 -> DoubleX16
zipDoubleX16 f = \ v1 v2 -> case unpackDoubleX16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackDoubleX16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packDoubleX16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipDoubleX16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipDoubleX16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipDoubleX16 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipDoubleX16 (/) a b = a / b #-}

{-# INLINE[1] foldDoubleX16 #-}
-- | Fold the elements of a vector to a single value
foldDoubleX16 :: (Double -> Double -> Double) -> DoubleX16 -> Double
foldDoubleX16 f' = \ v -> case unpackDoubleX16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldDoubleX16 (+) = sumVector #-}

{-# INLINE sumDoubleX16 #-}
-- | Sum up the elements of a vector to a single value.
sumDoubleX16 :: DoubleX16 -> Double
sumDoubleX16 (DoubleX16 x1 x2 x3 x4 x5 x6 x7 x8) = case unpackDoubleX2# (plusDoubleX2# x1 (plusDoubleX2# x2 (plusDoubleX2# x3 (plusDoubleX2# x4 (plusDoubleX2# x5 (plusDoubleX2# x6 (plusDoubleX2# x7 x8))))))) of
    (# y1, y2 #) -> D# y1 + D# y2

{-# INLINE plusDoubleX16 #-}
-- | Add two vectors element-wise.
plusDoubleX16 :: DoubleX16 -> DoubleX16 -> DoubleX16
plusDoubleX16 (DoubleX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (DoubleX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = DoubleX16 (plusDoubleX2# m1_1 m1_2) (plusDoubleX2# m2_1 m2_2) (plusDoubleX2# m3_1 m3_2) (plusDoubleX2# m4_1 m4_2) (plusDoubleX2# m5_1 m5_2) (plusDoubleX2# m6_1 m6_2) (plusDoubleX2# m7_1 m7_2) (plusDoubleX2# m8_1 m8_2)

{-# INLINE minusDoubleX16 #-}
-- | Subtract two vectors element-wise.
minusDoubleX16 :: DoubleX16 -> DoubleX16 -> DoubleX16
minusDoubleX16 (DoubleX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (DoubleX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = DoubleX16 (minusDoubleX2# m1_1 m1_2) (minusDoubleX2# m2_1 m2_2) (minusDoubleX2# m3_1 m3_2) (minusDoubleX2# m4_1 m4_2) (minusDoubleX2# m5_1 m5_2) (minusDoubleX2# m6_1 m6_2) (minusDoubleX2# m7_1 m7_2) (minusDoubleX2# m8_1 m8_2)

{-# INLINE timesDoubleX16 #-}
-- | Multiply two vectors element-wise.
timesDoubleX16 :: DoubleX16 -> DoubleX16 -> DoubleX16
timesDoubleX16 (DoubleX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (DoubleX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = DoubleX16 (timesDoubleX2# m1_1 m1_2) (timesDoubleX2# m2_1 m2_2) (timesDoubleX2# m3_1 m3_2) (timesDoubleX2# m4_1 m4_2) (timesDoubleX2# m5_1 m5_2) (timesDoubleX2# m6_1 m6_2) (timesDoubleX2# m7_1 m7_2) (timesDoubleX2# m8_1 m8_2)

{-# INLINE divideDoubleX16 #-}
-- | Divide two vectors element-wise.
divideDoubleX16 :: DoubleX16 -> DoubleX16 -> DoubleX16
divideDoubleX16 (DoubleX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (DoubleX16 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = DoubleX16 (divideDoubleX2# m1_1 m1_2) (divideDoubleX2# m2_1 m2_2) (divideDoubleX2# m3_1 m3_2) (divideDoubleX2# m4_1 m4_2) (divideDoubleX2# m5_1 m5_2) (divideDoubleX2# m6_1 m6_2) (divideDoubleX2# m7_1 m7_2) (divideDoubleX2# m8_1 m8_2)

{-# INLINE negateDoubleX16 #-}
-- | Negate element-wise.
negateDoubleX16 :: DoubleX16 -> DoubleX16
negateDoubleX16 (DoubleX16 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) = DoubleX16 (negateDoubleX2# m1_1) (negateDoubleX2# m2_1) (negateDoubleX2# m3_1) (negateDoubleX2# m4_1) (negateDoubleX2# m5_1) (negateDoubleX2# m6_1) (negateDoubleX2# m7_1) (negateDoubleX2# m8_1)

{-# INLINE indexDoubleX16Array #-}
-- | Read a vector from specified index of the immutable array.
indexDoubleX16Array :: ByteArray -> Int -> DoubleX16
indexDoubleX16Array (ByteArray a) (I# i) = DoubleX16 (indexDoubleX2Array# a ((i *# 8#) +# 0#)) (indexDoubleX2Array# a ((i *# 8#) +# 1#)) (indexDoubleX2Array# a ((i *# 8#) +# 2#)) (indexDoubleX2Array# a ((i *# 8#) +# 3#)) (indexDoubleX2Array# a ((i *# 8#) +# 4#)) (indexDoubleX2Array# a ((i *# 8#) +# 5#)) (indexDoubleX2Array# a ((i *# 8#) +# 6#)) (indexDoubleX2Array# a ((i *# 8#) +# 7#))

{-# INLINE readDoubleX16Array #-}
-- | Read a vector from specified index of the mutable array.
readDoubleX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m DoubleX16
readDoubleX16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readDoubleX2Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readDoubleX2Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readDoubleX2Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readDoubleX2Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readDoubleX2Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readDoubleX2Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readDoubleX2Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readDoubleX2Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeDoubleX16Array #-}
-- | Write a vector to specified index of mutable array.
writeDoubleX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> DoubleX16 -> m ()
writeDoubleX16Array (MutableByteArray a) (I# i) (DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeDoubleX2Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexDoubleX16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexDoubleX16OffAddr :: Addr -> Int -> DoubleX16
indexDoubleX16OffAddr (Addr a) (I# i) = DoubleX16 (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 0#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 16#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 32#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 48#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 64#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 80#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 96#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 112#)) 0#)

{-# INLINE readDoubleX16OffAddr #-}
-- | Reads vector from the specified index of the address.
readDoubleX16OffAddr :: PrimMonad m => Addr -> Int -> m DoubleX16
readDoubleX16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 48#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 64#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 80#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 96#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 128#) +# 112#) s7 of
                                (# s8, m8 #) -> (# s8, DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeDoubleX16OffAddr #-}
-- | Write vector to the specified index of the address.
writeDoubleX16OffAddr :: PrimMonad m => Addr -> Int -> DoubleX16 -> m ()
writeDoubleX16OffAddr (Addr a) (I# i) (DoubleX16 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 0#)) 0# m1) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 16#)) 0# m2) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 32#)) 0# m3) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 48#)) 0# m4) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 64#)) 0# m5) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 80#)) 0# m6) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 96#)) 0# m7) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 128#) +# 112#)) 0# m8)


