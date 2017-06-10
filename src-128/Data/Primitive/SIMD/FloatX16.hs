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
data FloatX16 = FloatX16 FloatX4# FloatX4# FloatX4# FloatX4# deriving Typeable

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
    sumVector          = sumFloatX16

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
broadcastFloatX16 (F# x) = case broadcastFloatX4# x of
    v -> FloatX16 v v v v

{-# INLINE[1] generateFloatX16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateFloatX16 :: (Int -> Float) -> FloatX16
generateFloatX16 f = packFloatX16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packFloatX16 #-}
-- | Pack the elements of a tuple into a vector.
packFloatX16 :: (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float) -> FloatX16
packFloatX16 (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8, F# x9, F# x10, F# x11, F# x12, F# x13, F# x14, F# x15, F# x16) = FloatX16 (packFloatX4# (# x1, x2, x3, x4 #)) (packFloatX4# (# x5, x6, x7, x8 #)) (packFloatX4# (# x9, x10, x11, x12 #)) (packFloatX4# (# x13, x14, x15, x16 #))

{-# INLINE unpackFloatX16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackFloatX16 :: FloatX16 -> (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float)
unpackFloatX16 (FloatX16 m1 m2 m3 m4) = case unpackFloatX4# m1 of
    (# x1, x2, x3, x4 #) -> case unpackFloatX4# m2 of
        (# x5, x6, x7, x8 #) -> case unpackFloatX4# m3 of
            (# x9, x10, x11, x12 #) -> case unpackFloatX4# m4 of
                (# x13, x14, x15, x16 #) -> (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8, F# x9, F# x10, F# x11, F# x12, F# x13, F# x14, F# x15, F# x16)

{-# INLINE unsafeInsertFloatX16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertFloatX16 :: FloatX16 -> Float -> Int -> FloatX16
unsafeInsertFloatX16 (FloatX16 m1 m2 m3 m4) (F# y) _i@(I# ip) | _i < 4 = FloatX16 (insertFloatX4# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 8 = FloatX16 m1 (insertFloatX4# m2 y (ip -# 4#)) m3 m4
                                                              | _i < 12 = FloatX16 m1 m2 (insertFloatX4# m3 y (ip -# 8#)) m4
                                                              | otherwise = FloatX16 m1 m2 m3 (insertFloatX4# m4 y (ip -# 12#))

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

{-# RULES "foldVector (+)" foldFloatX16 (+) = sumVector #-}

{-# INLINE sumFloatX16 #-}
-- | Sum up the elements of a vector to a single value.
sumFloatX16 :: FloatX16 -> Float
sumFloatX16 (FloatX16 x1 x2 x3 x4) = case unpackFloatX4# (plusFloatX4# x1 (plusFloatX4# x2 (plusFloatX4# x3 x4))) of
    (# y1, y2, y3, y4 #) -> F# y1 + F# y2 + F# y3 + F# y4

{-# INLINE plusFloatX16 #-}
-- | Add two vectors element-wise.
plusFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
plusFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1) (FloatX16 m1_2 m2_2 m3_2 m4_2) = FloatX16 (plusFloatX4# m1_1 m1_2) (plusFloatX4# m2_1 m2_2) (plusFloatX4# m3_1 m3_2) (plusFloatX4# m4_1 m4_2)

{-# INLINE minusFloatX16 #-}
-- | Subtract two vectors element-wise.
minusFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
minusFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1) (FloatX16 m1_2 m2_2 m3_2 m4_2) = FloatX16 (minusFloatX4# m1_1 m1_2) (minusFloatX4# m2_1 m2_2) (minusFloatX4# m3_1 m3_2) (minusFloatX4# m4_1 m4_2)

{-# INLINE timesFloatX16 #-}
-- | Multiply two vectors element-wise.
timesFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
timesFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1) (FloatX16 m1_2 m2_2 m3_2 m4_2) = FloatX16 (timesFloatX4# m1_1 m1_2) (timesFloatX4# m2_1 m2_2) (timesFloatX4# m3_1 m3_2) (timesFloatX4# m4_1 m4_2)

{-# INLINE divideFloatX16 #-}
-- | Divide two vectors element-wise.
divideFloatX16 :: FloatX16 -> FloatX16 -> FloatX16
divideFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1) (FloatX16 m1_2 m2_2 m3_2 m4_2) = FloatX16 (divideFloatX4# m1_1 m1_2) (divideFloatX4# m2_1 m2_2) (divideFloatX4# m3_1 m3_2) (divideFloatX4# m4_1 m4_2)

{-# INLINE negateFloatX16 #-}
-- | Negate element-wise.
negateFloatX16 :: FloatX16 -> FloatX16
negateFloatX16 (FloatX16 m1_1 m2_1 m3_1 m4_1) = FloatX16 (negateFloatX4# m1_1) (negateFloatX4# m2_1) (negateFloatX4# m3_1) (negateFloatX4# m4_1)

{-# INLINE indexFloatX16Array #-}
-- | Read a vector from specified index of the immutable array.
indexFloatX16Array :: ByteArray -> Int -> FloatX16
indexFloatX16Array (ByteArray a) (I# i) = FloatX16 (indexFloatX4Array# a ((i *# 4#) +# 0#)) (indexFloatX4Array# a ((i *# 4#) +# 1#)) (indexFloatX4Array# a ((i *# 4#) +# 2#)) (indexFloatX4Array# a ((i *# 4#) +# 3#))

{-# INLINE readFloatX16Array #-}
-- | Read a vector from specified index of the mutable array.
readFloatX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m FloatX16
readFloatX16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readFloatX4Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readFloatX4Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readFloatX4Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readFloatX4Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, FloatX16 m1 m2 m3 m4 #))

{-# INLINE writeFloatX16Array #-}
-- | Write a vector to specified index of mutable array.
writeFloatX16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> FloatX16 -> m ()
writeFloatX16Array (MutableByteArray a) (I# i) (FloatX16 m1 m2 m3 m4) = primitive_ (writeFloatX4Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeFloatX4Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeFloatX4Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeFloatX4Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexFloatX16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexFloatX16OffAddr :: Addr -> Int -> FloatX16
indexFloatX16OffAddr (Addr a) (I# i) = FloatX16 (indexFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#)

{-# INLINE readFloatX16OffAddr #-}
-- | Reads vector from the specified index of the address.
readFloatX16OffAddr :: PrimMonad m => Addr -> Int -> m FloatX16
readFloatX16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readFloatX4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readFloatX4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readFloatX4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readFloatX4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s3 of
                (# s4, m4 #) -> (# s4, FloatX16 m1 m2 m3 m4 #))

{-# INLINE writeFloatX16OffAddr #-}
-- | Write vector to the specified index of the address.
writeFloatX16OffAddr :: PrimMonad m => Addr -> Int -> FloatX16 -> m ()
writeFloatX16OffAddr (Addr a) (I# i) (FloatX16 m1 m2 m3 m4) = primitive_ (writeFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m2) >> primitive_ (writeFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m3) >> primitive_ (writeFloatX4OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m4)


