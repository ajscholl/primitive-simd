{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.DoubleX2 (DoubleX2) where

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

-- ** DoubleX2
data DoubleX2 = DoubleX2 Double# Double# deriving Typeable

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

instance Eq DoubleX2 where
    a == b = case unpackDoubleX2 a of
        (x1, x2) -> case unpackDoubleX2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord DoubleX2 where
    a `compare` b = case unpackDoubleX2 a of
        (x1, x2) -> case unpackDoubleX2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show DoubleX2 where
    showsPrec _ a s = case unpackDoubleX2 a of
        (x1, x2) -> "DoubleX2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num DoubleX2 where
    (+) = plusDoubleX2
    (-) = minusDoubleX2
    (*) = timesDoubleX2
    negate = negateDoubleX2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional DoubleX2 where
    (/)          = divideDoubleX2
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating DoubleX2 where
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

instance Storable DoubleX2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector DoubleX2 where
    type Elem DoubleX2 = Double
    type ElemTuple DoubleX2 = (Double, Double)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 8
    broadcastVector    = broadcastDoubleX2
    unsafeInsertVector = unsafeInsertDoubleX2
    packVector         = packDoubleX2
    unpackVector       = unpackDoubleX2
    mapVector          = mapDoubleX2
    zipVector          = zipDoubleX2
    foldVector         = foldDoubleX2

instance Prim DoubleX2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexDoubleX2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readDoubleX2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeDoubleX2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexDoubleX2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readDoubleX2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeDoubleX2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector DoubleX2 = V_DoubleX2 (PV.Vector DoubleX2)
newtype instance UV.MVector s DoubleX2 = MV_DoubleX2 (PMV.MVector s DoubleX2)

instance Vector UV.Vector DoubleX2 where
    basicUnsafeFreeze (MV_DoubleX2 v) = V_DoubleX2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_DoubleX2 v) = MV_DoubleX2 <$> PV.unsafeThaw v
    basicLength (V_DoubleX2 v) = PV.length v
    basicUnsafeSlice start len (V_DoubleX2 v) = V_DoubleX2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_DoubleX2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_DoubleX2 m) (V_DoubleX2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector DoubleX2 where
    basicLength (MV_DoubleX2 v) = PMV.length v
    basicUnsafeSlice start len (MV_DoubleX2 v) = MV_DoubleX2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_DoubleX2 v) (MV_DoubleX2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_DoubleX2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_DoubleX2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_DoubleX2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_DoubleX2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox DoubleX2

{-# INLINE broadcastDoubleX2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastDoubleX2 :: Double -> DoubleX2
broadcastDoubleX2 (D# x) = case broadcastDouble# x of
    v -> DoubleX2 v v

{-# INLINE packDoubleX2 #-}
-- | Pack the elements of a tuple into a vector.
packDoubleX2 :: (Double, Double) -> DoubleX2
packDoubleX2 (D# x1, D# x2) = DoubleX2 (packDouble# (# x1 #)) (packDouble# (# x2 #))

{-# INLINE unpackDoubleX2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackDoubleX2 :: DoubleX2 -> (Double, Double)
unpackDoubleX2 (DoubleX2 m1 m2) = case unpackDouble# m1 of
    (# x1 #) -> case unpackDouble# m2 of
        (# x2 #) -> (D# x1, D# x2)

{-# INLINE unsafeInsertDoubleX2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertDoubleX2 :: DoubleX2 -> Double -> Int -> DoubleX2
unsafeInsertDoubleX2 (DoubleX2 m1 m2) (D# y) _i@(I# ip) | _i < 1 = DoubleX2 (insertDouble# m1 y (ip -# 0#)) m2
                                                        | otherwise = DoubleX2 m1 (insertDouble# m2 y (ip -# 1#))

{-# INLINE[1] mapDoubleX2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapDoubleX2 :: (Double -> Double) -> DoubleX2 -> DoubleX2
mapDoubleX2 f = mapDoubleX2# (\ x -> case f (D# x) of { D# y -> y})

{-# RULES "mapVector abs" mapDoubleX2 abs = abs #-}
{-# RULES "mapVector signum" mapDoubleX2 signum = signum #-}
{-# RULES "mapVector negate" mapDoubleX2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapDoubleX2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapDoubleX2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapDoubleX2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapDoubleX2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapDoubleX2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapDoubleX2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapDoubleX2 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapDoubleX2 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapDoubleX2 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapDoubleX2# #-}
-- | Unboxed helper function.
mapDoubleX2# :: (Double# -> Double#) -> DoubleX2 -> DoubleX2
mapDoubleX2# f = \ v -> case unpackDoubleX2 v of
    (D# x1, D# x2) -> packDoubleX2 (D# (f x1), D# (f x2))

{-# INLINE[1] zipDoubleX2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipDoubleX2 :: (Double -> Double -> Double) -> DoubleX2 -> DoubleX2 -> DoubleX2
zipDoubleX2 f = \ v1 v2 -> case unpackDoubleX2 v1 of
    (x1, x2) -> case unpackDoubleX2 v2 of
        (y1, y2) -> packDoubleX2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipDoubleX2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipDoubleX2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipDoubleX2 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipDoubleX2 (/) a b = a / b #-}

{-# INLINE[1] foldDoubleX2 #-}
-- | Fold the elements of a vector to a single value
foldDoubleX2 :: (Double -> Double -> Double) -> DoubleX2 -> Double
foldDoubleX2 f' = \ v -> case unpackDoubleX2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusDoubleX2 #-}
-- | Add two vectors element-wise.
plusDoubleX2 :: DoubleX2 -> DoubleX2 -> DoubleX2
plusDoubleX2 (DoubleX2 m1_1 m2_1) (DoubleX2 m1_2 m2_2) = DoubleX2 (plusDouble# m1_1 m1_2) (plusDouble# m2_1 m2_2)

{-# INLINE minusDoubleX2 #-}
-- | Subtract two vectors element-wise.
minusDoubleX2 :: DoubleX2 -> DoubleX2 -> DoubleX2
minusDoubleX2 (DoubleX2 m1_1 m2_1) (DoubleX2 m1_2 m2_2) = DoubleX2 (minusDouble# m1_1 m1_2) (minusDouble# m2_1 m2_2)

{-# INLINE timesDoubleX2 #-}
-- | Multiply two vectors element-wise.
timesDoubleX2 :: DoubleX2 -> DoubleX2 -> DoubleX2
timesDoubleX2 (DoubleX2 m1_1 m2_1) (DoubleX2 m1_2 m2_2) = DoubleX2 (timesDouble# m1_1 m1_2) (timesDouble# m2_1 m2_2)

{-# INLINE divideDoubleX2 #-}
-- | Divide two vectors element-wise.
divideDoubleX2 :: DoubleX2 -> DoubleX2 -> DoubleX2
divideDoubleX2 (DoubleX2 m1_1 m2_1) (DoubleX2 m1_2 m2_2) = DoubleX2 (divideDouble# m1_1 m1_2) (divideDouble# m2_1 m2_2)

{-# INLINE negateDoubleX2 #-}
-- | Negate element-wise.
negateDoubleX2 :: DoubleX2 -> DoubleX2
negateDoubleX2 (DoubleX2 m1_1 m2_1) = DoubleX2 (negateDouble# m1_1) (negateDouble# m2_1)

{-# INLINE indexDoubleX2Array #-}
-- | Read a vector from specified index of the immutable array.
indexDoubleX2Array :: ByteArray -> Int -> DoubleX2
indexDoubleX2Array (ByteArray a) (I# i) = DoubleX2 (indexDoubleArray# a ((i *# 2#) +# 0#)) (indexDoubleArray# a ((i *# 2#) +# 1#))

{-# INLINE readDoubleX2Array #-}
-- | Read a vector from specified index of the mutable array.
readDoubleX2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m DoubleX2
readDoubleX2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readDoubleArray# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readDoubleArray# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, DoubleX2 m1 m2 #))

{-# INLINE writeDoubleX2Array #-}
-- | Write a vector to specified index of mutable array.
writeDoubleX2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> DoubleX2 -> m ()
writeDoubleX2Array (MutableByteArray a) (I# i) (DoubleX2 m1 m2) = primitive_ (writeDoubleArray# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeDoubleArray# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexDoubleX2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexDoubleX2OffAddr :: Addr -> Int -> DoubleX2
indexDoubleX2OffAddr (Addr a) (I# i) = DoubleX2 (indexDoubleOffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexDoubleOffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#)

{-# INLINE readDoubleX2OffAddr #-}
-- | Reads vector from the specified index of the address.
readDoubleX2OffAddr :: PrimMonad m => Addr -> Int -> m DoubleX2
readDoubleX2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readDoubleOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s1 of
        (# s2, m2 #) -> (# s2, DoubleX2 m1 m2 #))

{-# INLINE writeDoubleX2OffAddr #-}
-- | Write vector to the specified index of the address.
writeDoubleX2OffAddr :: PrimMonad m => Addr -> Int -> DoubleX2 -> m ()
writeDoubleX2OffAddr (Addr a) (I# i) (DoubleX2 m1 m2) = primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeDoubleOffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m2)


