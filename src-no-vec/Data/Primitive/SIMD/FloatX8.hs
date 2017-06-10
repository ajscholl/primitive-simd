{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.FloatX8 (FloatX8) where

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

-- ** FloatX8
data FloatX8 = FloatX8 Float# Float# Float# Float# Float# Float# Float# Float# deriving Typeable

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

instance Eq FloatX8 where
    a == b = case unpackFloatX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackFloatX8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord FloatX8 where
    a `compare` b = case unpackFloatX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackFloatX8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show FloatX8 where
    showsPrec _ a s = case unpackFloatX8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "FloatX8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num FloatX8 where
    (+) = plusFloatX8
    (-) = minusFloatX8
    (*) = timesFloatX8
    negate = negateFloatX8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional FloatX8 where
    (/)          = divideFloatX8
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating FloatX8 where
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

instance Storable FloatX8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector FloatX8 where
    type Elem FloatX8 = Float
    type ElemTuple FloatX8 = (Float, Float, Float, Float, Float, Float, Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 4
    broadcastVector    = broadcastFloatX8
    generateVector     = generateFloatX8
    unsafeInsertVector = unsafeInsertFloatX8
    packVector         = packFloatX8
    unpackVector       = unpackFloatX8
    mapVector          = mapFloatX8
    zipVector          = zipFloatX8
    foldVector         = foldFloatX8

instance Prim FloatX8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexFloatX8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readFloatX8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeFloatX8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexFloatX8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readFloatX8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeFloatX8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector FloatX8 = V_FloatX8 (PV.Vector FloatX8)
newtype instance UV.MVector s FloatX8 = MV_FloatX8 (PMV.MVector s FloatX8)

instance Vector UV.Vector FloatX8 where
    basicUnsafeFreeze (MV_FloatX8 v) = V_FloatX8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_FloatX8 v) = MV_FloatX8 <$> PV.unsafeThaw v
    basicLength (V_FloatX8 v) = PV.length v
    basicUnsafeSlice start len (V_FloatX8 v) = V_FloatX8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_FloatX8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_FloatX8 m) (V_FloatX8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector FloatX8 where
    basicLength (MV_FloatX8 v) = PMV.length v
    basicUnsafeSlice start len (MV_FloatX8 v) = MV_FloatX8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_FloatX8 v) (MV_FloatX8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_FloatX8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_FloatX8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_FloatX8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_FloatX8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox FloatX8

{-# INLINE broadcastFloatX8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastFloatX8 :: Float -> FloatX8
broadcastFloatX8 (F# x) = case broadcastFloat# x of
    v -> FloatX8 v v v v v v v v

{-# INLINE[1] generateFloatX8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateFloatX8 :: (Int -> Float) -> FloatX8
generateFloatX8 f = packFloatX8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packFloatX8 #-}
-- | Pack the elements of a tuple into a vector.
packFloatX8 :: (Float, Float, Float, Float, Float, Float, Float, Float) -> FloatX8
packFloatX8 (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8) = FloatX8 (packFloat# (# x1 #)) (packFloat# (# x2 #)) (packFloat# (# x3 #)) (packFloat# (# x4 #)) (packFloat# (# x5 #)) (packFloat# (# x6 #)) (packFloat# (# x7 #)) (packFloat# (# x8 #))

{-# INLINE unpackFloatX8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackFloatX8 :: FloatX8 -> (Float, Float, Float, Float, Float, Float, Float, Float)
unpackFloatX8 (FloatX8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackFloat# m1 of
    (# x1 #) -> case unpackFloat# m2 of
        (# x2 #) -> case unpackFloat# m3 of
            (# x3 #) -> case unpackFloat# m4 of
                (# x4 #) -> case unpackFloat# m5 of
                    (# x5 #) -> case unpackFloat# m6 of
                        (# x6 #) -> case unpackFloat# m7 of
                            (# x7 #) -> case unpackFloat# m8 of
                                (# x8 #) -> (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8)

{-# INLINE unsafeInsertFloatX8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertFloatX8 :: FloatX8 -> Float -> Int -> FloatX8
unsafeInsertFloatX8 (FloatX8 m1 m2 m3 m4 m5 m6 m7 m8) (F# y) _i@(I# ip) | _i < 1 = FloatX8 (insertFloat# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                        | _i < 2 = FloatX8 m1 (insertFloat# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                        | _i < 3 = FloatX8 m1 m2 (insertFloat# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                        | _i < 4 = FloatX8 m1 m2 m3 (insertFloat# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                        | _i < 5 = FloatX8 m1 m2 m3 m4 (insertFloat# m5 y (ip -# 4#)) m6 m7 m8
                                                                        | _i < 6 = FloatX8 m1 m2 m3 m4 m5 (insertFloat# m6 y (ip -# 5#)) m7 m8
                                                                        | _i < 7 = FloatX8 m1 m2 m3 m4 m5 m6 (insertFloat# m7 y (ip -# 6#)) m8
                                                                        | otherwise = FloatX8 m1 m2 m3 m4 m5 m6 m7 (insertFloat# m8 y (ip -# 7#))

{-# INLINE[1] mapFloatX8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapFloatX8 :: (Float -> Float) -> FloatX8 -> FloatX8
mapFloatX8 f = mapFloatX8# (\ x -> case f (F# x) of { F# y -> y})

{-# RULES "mapVector abs" mapFloatX8 abs = abs #-}
{-# RULES "mapVector signum" mapFloatX8 signum = signum #-}
{-# RULES "mapVector negate" mapFloatX8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapFloatX8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapFloatX8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapFloatX8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapFloatX8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapFloatX8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapFloatX8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapFloatX8 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapFloatX8 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapFloatX8 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapFloatX8# #-}
-- | Unboxed helper function.
mapFloatX8# :: (Float# -> Float#) -> FloatX8 -> FloatX8
mapFloatX8# f = \ v -> case unpackFloatX8 v of
    (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8) -> packFloatX8 (F# (f x1), F# (f x2), F# (f x3), F# (f x4), F# (f x5), F# (f x6), F# (f x7), F# (f x8))

{-# INLINE[1] zipFloatX8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipFloatX8 :: (Float -> Float -> Float) -> FloatX8 -> FloatX8 -> FloatX8
zipFloatX8 f = \ v1 v2 -> case unpackFloatX8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackFloatX8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packFloatX8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipFloatX8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipFloatX8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipFloatX8 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipFloatX8 (/) a b = a / b #-}

{-# INLINE[1] foldFloatX8 #-}
-- | Fold the elements of a vector to a single value
foldFloatX8 :: (Float -> Float -> Float) -> FloatX8 -> Float
foldFloatX8 f' = \ v -> case unpackFloatX8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusFloatX8 #-}
-- | Add two vectors element-wise.
plusFloatX8 :: FloatX8 -> FloatX8 -> FloatX8
plusFloatX8 (FloatX8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (FloatX8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = FloatX8 (plusFloat# m1_1 m1_2) (plusFloat# m2_1 m2_2) (plusFloat# m3_1 m3_2) (plusFloat# m4_1 m4_2) (plusFloat# m5_1 m5_2) (plusFloat# m6_1 m6_2) (plusFloat# m7_1 m7_2) (plusFloat# m8_1 m8_2)

{-# INLINE minusFloatX8 #-}
-- | Subtract two vectors element-wise.
minusFloatX8 :: FloatX8 -> FloatX8 -> FloatX8
minusFloatX8 (FloatX8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (FloatX8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = FloatX8 (minusFloat# m1_1 m1_2) (minusFloat# m2_1 m2_2) (minusFloat# m3_1 m3_2) (minusFloat# m4_1 m4_2) (minusFloat# m5_1 m5_2) (minusFloat# m6_1 m6_2) (minusFloat# m7_1 m7_2) (minusFloat# m8_1 m8_2)

{-# INLINE timesFloatX8 #-}
-- | Multiply two vectors element-wise.
timesFloatX8 :: FloatX8 -> FloatX8 -> FloatX8
timesFloatX8 (FloatX8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (FloatX8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = FloatX8 (timesFloat# m1_1 m1_2) (timesFloat# m2_1 m2_2) (timesFloat# m3_1 m3_2) (timesFloat# m4_1 m4_2) (timesFloat# m5_1 m5_2) (timesFloat# m6_1 m6_2) (timesFloat# m7_1 m7_2) (timesFloat# m8_1 m8_2)

{-# INLINE divideFloatX8 #-}
-- | Divide two vectors element-wise.
divideFloatX8 :: FloatX8 -> FloatX8 -> FloatX8
divideFloatX8 (FloatX8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (FloatX8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = FloatX8 (divideFloat# m1_1 m1_2) (divideFloat# m2_1 m2_2) (divideFloat# m3_1 m3_2) (divideFloat# m4_1 m4_2) (divideFloat# m5_1 m5_2) (divideFloat# m6_1 m6_2) (divideFloat# m7_1 m7_2) (divideFloat# m8_1 m8_2)

{-# INLINE negateFloatX8 #-}
-- | Negate element-wise.
negateFloatX8 :: FloatX8 -> FloatX8
negateFloatX8 (FloatX8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) = FloatX8 (negateFloat# m1_1) (negateFloat# m2_1) (negateFloat# m3_1) (negateFloat# m4_1) (negateFloat# m5_1) (negateFloat# m6_1) (negateFloat# m7_1) (negateFloat# m8_1)

{-# INLINE indexFloatX8Array #-}
-- | Read a vector from specified index of the immutable array.
indexFloatX8Array :: ByteArray -> Int -> FloatX8
indexFloatX8Array (ByteArray a) (I# i) = FloatX8 (indexFloatArray# a ((i *# 8#) +# 0#)) (indexFloatArray# a ((i *# 8#) +# 1#)) (indexFloatArray# a ((i *# 8#) +# 2#)) (indexFloatArray# a ((i *# 8#) +# 3#)) (indexFloatArray# a ((i *# 8#) +# 4#)) (indexFloatArray# a ((i *# 8#) +# 5#)) (indexFloatArray# a ((i *# 8#) +# 6#)) (indexFloatArray# a ((i *# 8#) +# 7#))

{-# INLINE readFloatX8Array #-}
-- | Read a vector from specified index of the mutable array.
readFloatX8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m FloatX8
readFloatX8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readFloatArray# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readFloatArray# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readFloatArray# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readFloatArray# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readFloatArray# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readFloatArray# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readFloatArray# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readFloatArray# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, FloatX8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeFloatX8Array #-}
-- | Write a vector to specified index of mutable array.
writeFloatX8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> FloatX8 -> m ()
writeFloatX8Array (MutableByteArray a) (I# i) (FloatX8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeFloatArray# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeFloatArray# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexFloatX8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexFloatX8OffAddr :: Addr -> Int -> FloatX8
indexFloatX8OffAddr (Addr a) (I# i) = FloatX8 (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexFloatOffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#)

{-# INLINE readFloatX8OffAddr #-}
-- | Reads vector from the specified index of the address.
readFloatX8OffAddr :: PrimMonad m => Addr -> Int -> m FloatX8
readFloatX8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readFloatOffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s7 of
                                (# s8, m8 #) -> (# s8, FloatX8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeFloatX8OffAddr #-}
-- | Write vector to the specified index of the address.
writeFloatX8OffAddr :: PrimMonad m => Addr -> Int -> FloatX8 -> m ()
writeFloatX8OffAddr (Addr a) (I# i) (FloatX8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m2) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m3) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m4) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m5) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m6) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m7) >> primitive_ (writeFloatOffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m8)


