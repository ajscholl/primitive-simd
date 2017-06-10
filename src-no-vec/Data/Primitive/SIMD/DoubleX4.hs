{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.DoubleX4 (DoubleX4) where

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

-- ** DoubleX4
data DoubleX4 = DoubleX4 Double# Double# Double# Double# deriving Typeable

broadcastDouble# :: Double# -> Double#
broadcastDouble# v = v

packDouble# :: (# Double# #) -> Double#
packDouble# (# v #) = v

unpackDouble# :: Double# -> (# Double# #)
unpackDouble# v = (# v #)

insertDouble# :: Double# -> Double# -> Int# -> Double#
insertDouble# _ v _ = v

plusDouble# :: Double# -> Double# -> Double#
plusDouble# a b = case D# a + D# b of D# c -> c

minusDouble# :: Double# -> Double# -> Double#
minusDouble# a b = case D# a - D# b of D# c -> c

timesDouble# :: Double# -> Double# -> Double#
timesDouble# a b = case D# a * D# b of D# c -> c

divideDouble# :: Double# -> Double# -> Double#
divideDouble# a b = case D# a / D# b of D# c -> c

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

instance Eq DoubleX4 where
    a == b = case unpackDoubleX4 a of
        (x1, x2, x3, x4) -> case unpackDoubleX4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord DoubleX4 where
    a `compare` b = case unpackDoubleX4 a of
        (x1, x2, x3, x4) -> case unpackDoubleX4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show DoubleX4 where
    showsPrec _ a s = case unpackDoubleX4 a of
        (x1, x2, x3, x4) -> "DoubleX4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num DoubleX4 where
    (+) = plusDoubleX4
    (-) = minusDoubleX4
    (*) = timesDoubleX4
    negate = negateDoubleX4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional DoubleX4 where
    (/)          = divideDoubleX4
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating DoubleX4 where
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

instance Storable DoubleX4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector DoubleX4 where
    type Elem DoubleX4 = Double
    type ElemTuple DoubleX4 = (Double, Double, Double, Double)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 8
    broadcastVector    = broadcastDoubleX4
    generateVector     = generateDoubleX4
    unsafeInsertVector = unsafeInsertDoubleX4
    packVector         = packDoubleX4
    unpackVector       = unpackDoubleX4
    mapVector          = mapDoubleX4
    zipVector          = zipDoubleX4
    foldVector         = foldDoubleX4

instance Prim DoubleX4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexDoubleX4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readDoubleX4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeDoubleX4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexDoubleX4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readDoubleX4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeDoubleX4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector DoubleX4 = V_DoubleX4 (PV.Vector DoubleX4)
newtype instance UV.MVector s DoubleX4 = MV_DoubleX4 (PMV.MVector s DoubleX4)

instance Vector UV.Vector DoubleX4 where
    basicUnsafeFreeze (MV_DoubleX4 v) = V_DoubleX4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_DoubleX4 v) = MV_DoubleX4 <$> PV.unsafeThaw v
    basicLength (V_DoubleX4 v) = PV.length v
    basicUnsafeSlice start len (V_DoubleX4 v) = V_DoubleX4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_DoubleX4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_DoubleX4 m) (V_DoubleX4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector DoubleX4 where
    basicLength (MV_DoubleX4 v) = PMV.length v
    basicUnsafeSlice start len (MV_DoubleX4 v) = MV_DoubleX4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_DoubleX4 v) (MV_DoubleX4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_DoubleX4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_DoubleX4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_DoubleX4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_DoubleX4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox DoubleX4

{-# INLINE broadcastDoubleX4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastDoubleX4 :: Double -> DoubleX4
broadcastDoubleX4 (D# x) = case broadcastDouble# x of
    v -> DoubleX4 v v v v

{-# INLINE[1] generateDoubleX4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateDoubleX4 :: (Int -> Double) -> DoubleX4
generateDoubleX4 f = packDoubleX4 (f 0, f 1, f 2, f 3)

{-# INLINE packDoubleX4 #-}
-- | Pack the elements of a tuple into a vector.
packDoubleX4 :: (Double, Double, Double, Double) -> DoubleX4
packDoubleX4 (D# x1, D# x2, D# x3, D# x4) = DoubleX4 (packDouble# (# x1 #)) (packDouble# (# x2 #)) (packDouble# (# x3 #)) (packDouble# (# x4 #))

{-# INLINE unpackDoubleX4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackDoubleX4 :: DoubleX4 -> (Double, Double, Double, Double)
unpackDoubleX4 (DoubleX4 m1 m2 m3 m4) = case unpackDouble# m1 of
    (# x1 #) -> case unpackDouble# m2 of
        (# x2 #) -> case unpackDouble# m3 of
            (# x3 #) -> case unpackDouble# m4 of
                (# x4 #) -> (D# x1, D# x2, D# x3, D# x4)

{-# INLINE unsafeInsertDoubleX4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertDoubleX4 :: DoubleX4 -> Double -> Int -> DoubleX4
unsafeInsertDoubleX4 (DoubleX4 m1 m2 m3 m4) (D# y) _i@(I# ip) | _i < 1 = DoubleX4 (insertDouble# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 2 = DoubleX4 m1 (insertDouble# m2 y (ip -# 1#)) m3 m4
                                                              | _i < 3 = DoubleX4 m1 m2 (insertDouble# m3 y (ip -# 2#)) m4
                                                              | otherwise = DoubleX4 m1 m2 m3 (insertDouble# m4 y (ip -# 3#))

{-# INLINE[1] mapDoubleX4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapDoubleX4 :: (Double -> Double) -> DoubleX4 -> DoubleX4
mapDoubleX4 f = mapDoubleX4# (\ x -> case f (D# x) of { D# y -> y})

{-# RULES "mapVector abs" mapDoubleX4 abs = abs #-}
{-# RULES "mapVector signum" mapDoubleX4 signum = signum #-}
{-# RULES "mapVector negate" mapDoubleX4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapDoubleX4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapDoubleX4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapDoubleX4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapDoubleX4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapDoubleX4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapDoubleX4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapDoubleX4 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapDoubleX4 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapDoubleX4 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapDoubleX4# #-}
-- | Unboxed helper function.
mapDoubleX4# :: (Double# -> Double#) -> DoubleX4 -> DoubleX4
mapDoubleX4# f = \ v -> case unpackDoubleX4 v of
    (D# x1, D# x2, D# x3, D# x4) -> packDoubleX4 (D# (f x1), D# (f x2), D# (f x3), D# (f x4))

{-# INLINE[1] zipDoubleX4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipDoubleX4 :: (Double -> Double -> Double) -> DoubleX4 -> DoubleX4 -> DoubleX4
zipDoubleX4 f = \ v1 v2 -> case unpackDoubleX4 v1 of
    (x1, x2, x3, x4) -> case unpackDoubleX4 v2 of
        (y1, y2, y3, y4) -> packDoubleX4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipDoubleX4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipDoubleX4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipDoubleX4 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipDoubleX4 (/) a b = a / b #-}

{-# INLINE[1] foldDoubleX4 #-}
-- | Fold the elements of a vector to a single value
foldDoubleX4 :: (Double -> Double -> Double) -> DoubleX4 -> Double
foldDoubleX4 f' = \ v -> case unpackDoubleX4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusDoubleX4 #-}
-- | Add two vectors element-wise.
plusDoubleX4 :: DoubleX4 -> DoubleX4 -> DoubleX4
plusDoubleX4 (DoubleX4 m1_1 m2_1 m3_1 m4_1) (DoubleX4 m1_2 m2_2 m3_2 m4_2) = DoubleX4 (plusDouble# m1_1 m1_2) (plusDouble# m2_1 m2_2) (plusDouble# m3_1 m3_2) (plusDouble# m4_1 m4_2)

{-# INLINE minusDoubleX4 #-}
-- | Subtract two vectors element-wise.
minusDoubleX4 :: DoubleX4 -> DoubleX4 -> DoubleX4
minusDoubleX4 (DoubleX4 m1_1 m2_1 m3_1 m4_1) (DoubleX4 m1_2 m2_2 m3_2 m4_2) = DoubleX4 (minusDouble# m1_1 m1_2) (minusDouble# m2_1 m2_2) (minusDouble# m3_1 m3_2) (minusDouble# m4_1 m4_2)

{-# INLINE timesDoubleX4 #-}
-- | Multiply two vectors element-wise.
timesDoubleX4 :: DoubleX4 -> DoubleX4 -> DoubleX4
timesDoubleX4 (DoubleX4 m1_1 m2_1 m3_1 m4_1) (DoubleX4 m1_2 m2_2 m3_2 m4_2) = DoubleX4 (timesDouble# m1_1 m1_2) (timesDouble# m2_1 m2_2) (timesDouble# m3_1 m3_2) (timesDouble# m4_1 m4_2)

{-# INLINE divideDoubleX4 #-}
-- | Divide two vectors element-wise.
divideDoubleX4 :: DoubleX4 -> DoubleX4 -> DoubleX4
divideDoubleX4 (DoubleX4 m1_1 m2_1 m3_1 m4_1) (DoubleX4 m1_2 m2_2 m3_2 m4_2) = DoubleX4 (divideDouble# m1_1 m1_2) (divideDouble# m2_1 m2_2) (divideDouble# m3_1 m3_2) (divideDouble# m4_1 m4_2)

{-# INLINE negateDoubleX4 #-}
-- | Negate element-wise.
negateDoubleX4 :: DoubleX4 -> DoubleX4
negateDoubleX4 (DoubleX4 m1_1 m2_1 m3_1 m4_1) = DoubleX4 (negateDouble# m1_1) (negateDouble# m2_1) (negateDouble# m3_1) (negateDouble# m4_1)

{-# INLINE indexDoubleX4Array #-}
-- | Read a vector from specified index of the immutable array.
indexDoubleX4Array :: ByteArray -> Int -> DoubleX4
indexDoubleX4Array (ByteArray a) (I# i) = DoubleX4 (indexDoubleArray# a ((i *# 4#) +# 0#)) (indexDoubleArray# a ((i *# 4#) +# 1#)) (indexDoubleArray# a ((i *# 4#) +# 2#)) (indexDoubleArray# a ((i *# 4#) +# 3#))

{-# INLINE readDoubleX4Array #-}
-- | Read a vector from specified index of the mutable array.
readDoubleX4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m DoubleX4
readDoubleX4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readDoubleArray# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readDoubleArray# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readDoubleArray# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readDoubleArray# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, DoubleX4 m1 m2 m3 m4 #))

{-# INLINE writeDoubleX4Array #-}
-- | Write a vector to specified index of mutable array.
writeDoubleX4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> DoubleX4 -> m ()
writeDoubleX4Array (MutableByteArray a) (I# i) (DoubleX4 m1 m2 m3 m4) = primitive_ (writeDoubleArray# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeDoubleArray# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeDoubleArray# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeDoubleArray# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexDoubleX4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexDoubleX4OffAddr :: Addr -> Int -> DoubleX4
indexDoubleX4OffAddr (Addr a) (I# i) = DoubleX4 (indexDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#)

{-# INLINE readDoubleX4OffAddr #-}
-- | Reads vector from the specified index of the address.
readDoubleX4OffAddr :: PrimMonad m => Addr -> Int -> m DoubleX4
readDoubleX4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s3 of
                (# s4, m4 #) -> (# s4, DoubleX4 m1 m2 m3 m4 #))

{-# INLINE writeDoubleX4OffAddr #-}
-- | Write vector to the specified index of the address.
writeDoubleX4OffAddr :: PrimMonad m => Addr -> Int -> DoubleX4 -> m ()
writeDoubleX4OffAddr (Addr a) (I# i) (DoubleX4 m1 m2 m3 m4) = primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m2) >> primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m3) >> primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m4)


