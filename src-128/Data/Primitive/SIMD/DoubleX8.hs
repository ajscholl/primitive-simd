{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.DoubleX8 (DoubleX8) where

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

-- ** DoubleX8
data DoubleX8 = DoubleX8 DoubleX2# DoubleX2# DoubleX2# DoubleX2# deriving Typeable

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

instance Eq DoubleX8 where
    a == b = case unpackDoubleX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackDoubleX8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord DoubleX8 where
    a `compare` b = case unpackDoubleX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackDoubleX8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show DoubleX8 where
    showsPrec _ a s = case unpackDoubleX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "DoubleX8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num DoubleX8 where
    (+) = plusDoubleX8
    (-) = minusDoubleX8
    (*) = timesDoubleX8
    negate = negateDoubleX8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional DoubleX8 where
    (/)          = divideDoubleX8
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating DoubleX8 where
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

instance Storable DoubleX8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector DoubleX8 where
    type Elem DoubleX8 = Double
    type ElemTuple DoubleX8 = (Double, Double, Double, Double, Double, Double, Double, Double)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 8
    broadcastVector    = broadcastDoubleX8
    generateVector     = generateDoubleX8
    unsafeInsertVector = unsafeInsertDoubleX8
    packVector         = packDoubleX8
    unpackVector       = unpackDoubleX8
    mapVector          = mapDoubleX8
    zipVector          = zipDoubleX8
    foldVector         = foldDoubleX8
    sumVector          = sumDoubleX8

instance Prim DoubleX8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexDoubleX8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readDoubleX8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeDoubleX8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexDoubleX8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readDoubleX8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeDoubleX8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector DoubleX8 = V_DoubleX8 (PV.Vector DoubleX8)
newtype instance UV.MVector s DoubleX8 = MV_DoubleX8 (PMV.MVector s DoubleX8)

instance Vector UV.Vector DoubleX8 where
    basicUnsafeFreeze (MV_DoubleX8 v) = V_DoubleX8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_DoubleX8 v) = MV_DoubleX8 <$> PV.unsafeThaw v
    basicLength (V_DoubleX8 v) = PV.length v
    basicUnsafeSlice start len (V_DoubleX8 v) = V_DoubleX8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_DoubleX8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_DoubleX8 m) (V_DoubleX8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector DoubleX8 where
    basicLength (MV_DoubleX8 v) = PMV.length v
    basicUnsafeSlice start len (MV_DoubleX8 v) = MV_DoubleX8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_DoubleX8 v) (MV_DoubleX8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_DoubleX8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_DoubleX8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_DoubleX8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_DoubleX8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox DoubleX8

{-# INLINE broadcastDoubleX8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastDoubleX8 :: Double -> DoubleX8
broadcastDoubleX8 (D# x) = case broadcastDoubleX2# x of
    v -> DoubleX8 v v v v

{-# INLINE[1] generateDoubleX8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateDoubleX8 :: (Int -> Double) -> DoubleX8
generateDoubleX8 f = packDoubleX8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packDoubleX8 #-}
-- | Pack the elements of a tuple into a vector.
packDoubleX8 :: (Double, Double, Double, Double, Double, Double, Double, Double) -> DoubleX8
packDoubleX8 (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8) = DoubleX8 (packDoubleX2# (# x1, x2 #)) (packDoubleX2# (# x3, x4 #)) (packDoubleX2# (# x5, x6 #)) (packDoubleX2# (# x7, x8 #))

{-# INLINE unpackDoubleX8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackDoubleX8 :: DoubleX8 -> (Double, Double, Double, Double, Double, Double, Double, Double)
unpackDoubleX8 (DoubleX8 m1 m2 m3 m4) = case unpackDoubleX2# m1 of
    (# x1, x2 #) -> case unpackDoubleX2# m2 of
        (# x3, x4 #) -> case unpackDoubleX2# m3 of
            (# x5, x6 #) -> case unpackDoubleX2# m4 of
                (# x7, x8 #) -> (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8)

{-# INLINE unsafeInsertDoubleX8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertDoubleX8 :: DoubleX8 -> Double -> Int -> DoubleX8
unsafeInsertDoubleX8 (DoubleX8 m1 m2 m3 m4) (D# y) _i@(I# ip) | _i < 2 = DoubleX8 (insertDoubleX2# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 4 = DoubleX8 m1 (insertDoubleX2# m2 y (ip -# 2#)) m3 m4
                                                              | _i < 6 = DoubleX8 m1 m2 (insertDoubleX2# m3 y (ip -# 4#)) m4
                                                              | otherwise = DoubleX8 m1 m2 m3 (insertDoubleX2# m4 y (ip -# 6#))

{-# INLINE[1] mapDoubleX8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapDoubleX8 :: (Double -> Double) -> DoubleX8 -> DoubleX8
mapDoubleX8 f = mapDoubleX8# (\ x -> case f (D# x) of { D# y -> y})

{-# RULES "mapVector abs" mapDoubleX8 abs = abs #-}
{-# RULES "mapVector signum" mapDoubleX8 signum = signum #-}
{-# RULES "mapVector negate" mapDoubleX8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapDoubleX8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapDoubleX8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapDoubleX8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapDoubleX8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapDoubleX8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapDoubleX8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapDoubleX8 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapDoubleX8 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapDoubleX8 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapDoubleX8# #-}
-- | Unboxed helper function.
mapDoubleX8# :: (Double# -> Double#) -> DoubleX8 -> DoubleX8
mapDoubleX8# f = \ v -> case unpackDoubleX8 v of
    (D# x1, D# x2, D# x3, D# x4, D# x5, D# x6, D# x7, D# x8) -> packDoubleX8 (D# (f x1), D# (f x2), D# (f x3), D# (f x4), D# (f x5), D# (f x6), D# (f x7), D# (f x8))

{-# INLINE[1] zipDoubleX8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipDoubleX8 :: (Double -> Double -> Double) -> DoubleX8 -> DoubleX8 -> DoubleX8
zipDoubleX8 f = \ v1 v2 -> case unpackDoubleX8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackDoubleX8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packDoubleX8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipDoubleX8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipDoubleX8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipDoubleX8 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipDoubleX8 (/) a b = a / b #-}

{-# INLINE[1] foldDoubleX8 #-}
-- | Fold the elements of a vector to a single value
foldDoubleX8 :: (Double -> Double -> Double) -> DoubleX8 -> Double
foldDoubleX8 f' = \ v -> case unpackDoubleX8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldDoubleX8 (+) = sumVector #-}

{-# INLINE sumDoubleX8 #-}
-- | Sum up the elements of a vector to a single value.
sumDoubleX8 :: DoubleX8 -> Double
sumDoubleX8 (DoubleX8 x1 x2 x3 x4) = case unpackDoubleX2# (plusDoubleX2# x1 (plusDoubleX2# x2 (plusDoubleX2# x3 x4))) of
    (# y1, y2 #) -> D# y1 + D# y2

{-# INLINE plusDoubleX8 #-}
-- | Add two vectors element-wise.
plusDoubleX8 :: DoubleX8 -> DoubleX8 -> DoubleX8
plusDoubleX8 (DoubleX8 m1_1 m2_1 m3_1 m4_1) (DoubleX8 m1_2 m2_2 m3_2 m4_2) = DoubleX8 (plusDoubleX2# m1_1 m1_2) (plusDoubleX2# m2_1 m2_2) (plusDoubleX2# m3_1 m3_2) (plusDoubleX2# m4_1 m4_2)

{-# INLINE minusDoubleX8 #-}
-- | Subtract two vectors element-wise.
minusDoubleX8 :: DoubleX8 -> DoubleX8 -> DoubleX8
minusDoubleX8 (DoubleX8 m1_1 m2_1 m3_1 m4_1) (DoubleX8 m1_2 m2_2 m3_2 m4_2) = DoubleX8 (minusDoubleX2# m1_1 m1_2) (minusDoubleX2# m2_1 m2_2) (minusDoubleX2# m3_1 m3_2) (minusDoubleX2# m4_1 m4_2)

{-# INLINE timesDoubleX8 #-}
-- | Multiply two vectors element-wise.
timesDoubleX8 :: DoubleX8 -> DoubleX8 -> DoubleX8
timesDoubleX8 (DoubleX8 m1_1 m2_1 m3_1 m4_1) (DoubleX8 m1_2 m2_2 m3_2 m4_2) = DoubleX8 (timesDoubleX2# m1_1 m1_2) (timesDoubleX2# m2_1 m2_2) (timesDoubleX2# m3_1 m3_2) (timesDoubleX2# m4_1 m4_2)

{-# INLINE divideDoubleX8 #-}
-- | Divide two vectors element-wise.
divideDoubleX8 :: DoubleX8 -> DoubleX8 -> DoubleX8
divideDoubleX8 (DoubleX8 m1_1 m2_1 m3_1 m4_1) (DoubleX8 m1_2 m2_2 m3_2 m4_2) = DoubleX8 (divideDoubleX2# m1_1 m1_2) (divideDoubleX2# m2_1 m2_2) (divideDoubleX2# m3_1 m3_2) (divideDoubleX2# m4_1 m4_2)

{-# INLINE negateDoubleX8 #-}
-- | Negate element-wise.
negateDoubleX8 :: DoubleX8 -> DoubleX8
negateDoubleX8 (DoubleX8 m1_1 m2_1 m3_1 m4_1) = DoubleX8 (negateDoubleX2# m1_1) (negateDoubleX2# m2_1) (negateDoubleX2# m3_1) (negateDoubleX2# m4_1)

{-# INLINE indexDoubleX8Array #-}
-- | Read a vector from specified index of the immutable array.
indexDoubleX8Array :: ByteArray -> Int -> DoubleX8
indexDoubleX8Array (ByteArray a) (I# i) = DoubleX8 (indexDoubleX2Array# a ((i *# 4#) +# 0#)) (indexDoubleX2Array# a ((i *# 4#) +# 1#)) (indexDoubleX2Array# a ((i *# 4#) +# 2#)) (indexDoubleX2Array# a ((i *# 4#) +# 3#))

{-# INLINE readDoubleX8Array #-}
-- | Read a vector from specified index of the mutable array.
readDoubleX8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m DoubleX8
readDoubleX8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readDoubleX2Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readDoubleX2Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readDoubleX2Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readDoubleX2Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, DoubleX8 m1 m2 m3 m4 #))

{-# INLINE writeDoubleX8Array #-}
-- | Write a vector to specified index of mutable array.
writeDoubleX8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> DoubleX8 -> m ()
writeDoubleX8Array (MutableByteArray a) (I# i) (DoubleX8 m1 m2 m3 m4) = primitive_ (writeDoubleX2Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeDoubleX2Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeDoubleX2Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeDoubleX2Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexDoubleX8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexDoubleX8OffAddr :: Addr -> Int -> DoubleX8
indexDoubleX8OffAddr (Addr a) (I# i) = DoubleX8 (indexDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#)

{-# INLINE readDoubleX8OffAddr #-}
-- | Reads vector from the specified index of the address.
readDoubleX8OffAddr :: PrimMonad m => Addr -> Int -> m DoubleX8
readDoubleX8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readDoubleX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s3 of
                (# s4, m4 #) -> (# s4, DoubleX8 m1 m2 m3 m4 #))

{-# INLINE writeDoubleX8OffAddr #-}
-- | Write vector to the specified index of the address.
writeDoubleX8OffAddr :: PrimMonad m => Addr -> Int -> DoubleX8 -> m ()
writeDoubleX8OffAddr (Addr a) (I# i) (DoubleX8 m1 m2 m3 m4) = primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m2) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m3) >> primitive_ (writeDoubleX2OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m4)


