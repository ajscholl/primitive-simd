{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.FloatX4 (FloatX4) where

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

-- ** FloatX4
data FloatX4 = FloatX4 Float# Float# Float# Float# deriving Typeable

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

instance Eq FloatX4 where
    a == b = case unpackFloatX4 a of
        (x1, x2, x3, x4) -> case unpackFloatX4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord FloatX4 where
    a `compare` b = case unpackFloatX4 a of
        (x1, x2, x3, x4) -> case unpackFloatX4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show FloatX4 where
    showsPrec _ a s = case unpackFloatX4 a of
        (x1, x2, x3, x4) -> "FloatX4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num FloatX4 where
    (+) = plusFloatX4
    (-) = minusFloatX4
    (*) = timesFloatX4
    negate = negateFloatX4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional FloatX4 where
    (/)          = divideFloatX4
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating FloatX4 where
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

instance Storable FloatX4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector FloatX4 where
    type Elem FloatX4 = Float
    type ElemTuple FloatX4 = (Float, Float, Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 4
    broadcastVector    = broadcastFloatX4
    unsafeInsertVector = unsafeInsertFloatX4
    packVector         = packFloatX4
    unpackVector       = unpackFloatX4
    mapVector          = mapFloatX4
    zipVector          = zipFloatX4
    foldVector         = foldFloatX4

instance Prim FloatX4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexFloatX4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readFloatX4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeFloatX4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexFloatX4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readFloatX4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeFloatX4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector FloatX4 = V_FloatX4 (PV.Vector FloatX4)
newtype instance UV.MVector s FloatX4 = MV_FloatX4 (PMV.MVector s FloatX4)

instance Vector UV.Vector FloatX4 where
    basicUnsafeFreeze (MV_FloatX4 v) = V_FloatX4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_FloatX4 v) = MV_FloatX4 <$> PV.unsafeThaw v
    basicLength (V_FloatX4 v) = PV.length v
    basicUnsafeSlice start len (V_FloatX4 v) = V_FloatX4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_FloatX4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_FloatX4 m) (V_FloatX4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector FloatX4 where
    basicLength (MV_FloatX4 v) = PMV.length v
    basicUnsafeSlice start len (MV_FloatX4 v) = MV_FloatX4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_FloatX4 v) (MV_FloatX4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_FloatX4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_FloatX4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_FloatX4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_FloatX4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox FloatX4

{-# INLINE broadcastFloatX4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastFloatX4 :: Float -> FloatX4
broadcastFloatX4 (F# x) = case broadcastFloat# x of
    v -> FloatX4 v v v v

{-# INLINE packFloatX4 #-}
-- | Pack the elements of a tuple into a vector.
packFloatX4 :: (Float, Float, Float, Float) -> FloatX4
packFloatX4 (F# x1, F# x2, F# x3, F# x4) = FloatX4 (packFloat# (# x1 #)) (packFloat# (# x2 #)) (packFloat# (# x3 #)) (packFloat# (# x4 #))

{-# INLINE unpackFloatX4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackFloatX4 :: FloatX4 -> (Float, Float, Float, Float)
unpackFloatX4 (FloatX4 m1 m2 m3 m4) = case unpackFloat# m1 of
    (# x1 #) -> case unpackFloat# m2 of
        (# x2 #) -> case unpackFloat# m3 of
            (# x3 #) -> case unpackFloat# m4 of
                (# x4 #) -> (F# x1, F# x2, F# x3, F# x4)

{-# INLINE unsafeInsertFloatX4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertFloatX4 :: FloatX4 -> Float -> Int -> FloatX4
unsafeInsertFloatX4 (FloatX4 m1 m2 m3 m4) (F# y) _i@(I# ip) | _i < 1 = FloatX4 (insertFloat# m1 y (ip -# 0#)) m2 m3 m4
                                                            | _i < 2 = FloatX4 m1 (insertFloat# m2 y (ip -# 1#)) m3 m4
                                                            | _i < 3 = FloatX4 m1 m2 (insertFloat# m3 y (ip -# 2#)) m4
                                                            | otherwise = FloatX4 m1 m2 m3 (insertFloat# m4 y (ip -# 3#))

{-# INLINE[1] mapFloatX4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapFloatX4 :: (Float -> Float) -> FloatX4 -> FloatX4
mapFloatX4 f = mapFloatX4# (\ x -> case f (F# x) of { F# y -> y})

{-# RULES "mapVector abs" mapFloatX4 abs = abs #-}
{-# RULES "mapVector signum" mapFloatX4 signum = signum #-}
{-# RULES "mapVector negate" mapFloatX4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapFloatX4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapFloatX4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapFloatX4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapFloatX4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapFloatX4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapFloatX4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapFloatX4 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapFloatX4 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapFloatX4 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapFloatX4# #-}
-- | Unboxed helper function.
mapFloatX4# :: (Float# -> Float#) -> FloatX4 -> FloatX4
mapFloatX4# f = \ v -> case unpackFloatX4 v of
    (F# x1, F# x2, F# x3, F# x4) -> packFloatX4 (F# (f x1), F# (f x2), F# (f x3), F# (f x4))

{-# INLINE[1] zipFloatX4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipFloatX4 :: (Float -> Float -> Float) -> FloatX4 -> FloatX4 -> FloatX4
zipFloatX4 f = \ v1 v2 -> case unpackFloatX4 v1 of
    (x1, x2, x3, x4) -> case unpackFloatX4 v2 of
        (y1, y2, y3, y4) -> packFloatX4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipFloatX4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipFloatX4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipFloatX4 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipFloatX4 (/) a b = a / b #-}

{-# INLINE[1] foldFloatX4 #-}
-- | Fold the elements of a vector to a single value
foldFloatX4 :: (Float -> Float -> Float) -> FloatX4 -> Float
foldFloatX4 f' = \ v -> case unpackFloatX4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusFloatX4 #-}
-- | Add two vectors element-wise.
plusFloatX4 :: FloatX4 -> FloatX4 -> FloatX4
plusFloatX4 (FloatX4 m1_1 m2_1 m3_1 m4_1) (FloatX4 m1_2 m2_2 m3_2 m4_2) = FloatX4 (plusFloat# m1_1 m1_2) (plusFloat# m2_1 m2_2) (plusFloat# m3_1 m3_2) (plusFloat# m4_1 m4_2)

{-# INLINE minusFloatX4 #-}
-- | Subtract two vectors element-wise.
minusFloatX4 :: FloatX4 -> FloatX4 -> FloatX4
minusFloatX4 (FloatX4 m1_1 m2_1 m3_1 m4_1) (FloatX4 m1_2 m2_2 m3_2 m4_2) = FloatX4 (minusFloat# m1_1 m1_2) (minusFloat# m2_1 m2_2) (minusFloat# m3_1 m3_2) (minusFloat# m4_1 m4_2)

{-# INLINE timesFloatX4 #-}
-- | Multiply two vectors element-wise.
timesFloatX4 :: FloatX4 -> FloatX4 -> FloatX4
timesFloatX4 (FloatX4 m1_1 m2_1 m3_1 m4_1) (FloatX4 m1_2 m2_2 m3_2 m4_2) = FloatX4 (timesFloat# m1_1 m1_2) (timesFloat# m2_1 m2_2) (timesFloat# m3_1 m3_2) (timesFloat# m4_1 m4_2)

{-# INLINE divideFloatX4 #-}
-- | Divide two vectors element-wise.
divideFloatX4 :: FloatX4 -> FloatX4 -> FloatX4
divideFloatX4 (FloatX4 m1_1 m2_1 m3_1 m4_1) (FloatX4 m1_2 m2_2 m3_2 m4_2) = FloatX4 (divideFloat# m1_1 m1_2) (divideFloat# m2_1 m2_2) (divideFloat# m3_1 m3_2) (divideFloat# m4_1 m4_2)

{-# INLINE negateFloatX4 #-}
-- | Negate element-wise.
negateFloatX4 :: FloatX4 -> FloatX4
negateFloatX4 (FloatX4 m1_1 m2_1 m3_1 m4_1) = FloatX4 (negateFloat# m1_1) (negateFloat# m2_1) (negateFloat# m3_1) (negateFloat# m4_1)

{-# INLINE indexFloatX4Array #-}
-- | Read a vector from specified index of the immutable array.
indexFloatX4Array :: ByteArray -> Int -> FloatX4
indexFloatX4Array (ByteArray a) (I# i) = FloatX4 (indexFloatArray# a ((i *# 4#) +# 0#)) (indexFloatArray# a ((i *# 4#) +# 1#)) (indexFloatArray# a ((i *# 4#) +# 2#)) (indexFloatArray# a ((i *# 4#) +# 3#))

{-# INLINE readFloatX4Array #-}
-- | Read a vector from specified index of the mutable array.
readFloatX4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m FloatX4
readFloatX4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readFloatArray# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readFloatArray# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readFloatArray# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readFloatArray# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, FloatX4 m1 m2 m3 m4 #))

{-# INLINE writeFloatX4Array #-}
-- | Write a vector to specified index of mutable array.
writeFloatX4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> FloatX4 -> m ()
writeFloatX4Array (MutableByteArray a) (I# i) (FloatX4 m1 m2 m3 m4) = primitive_ (writeFloatArray# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeFloatArray# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeFloatArray# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeFloatArray# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexFloatX4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexFloatX4OffAddr :: Addr -> Int -> FloatX4
indexFloatX4OffAddr (Addr a) (I# i) = FloatX4 (indexFloatOffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#)

{-# INLINE readFloatX4OffAddr #-}
-- | Reads vector from the specified index of the address.
readFloatX4OffAddr :: PrimMonad m => Addr -> Int -> m FloatX4
readFloatX4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s3 of
                (# s4, m4 #) -> (# s4, FloatX4 m1 m2 m3 m4 #))

{-# INLINE writeFloatX4OffAddr #-}
-- | Write vector to the specified index of the address.
writeFloatX4OffAddr :: PrimMonad m => Addr -> Int -> FloatX4 -> m ()
writeFloatX4OffAddr (Addr a) (I# i) (FloatX4 m1 m2 m3 m4) = primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m2) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m3) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m4)


