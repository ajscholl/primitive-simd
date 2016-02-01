{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int32X8 (Int32X8) where

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

-- ** Int32X8
data Int32X8 = Int32X8 Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt32# :: Int# -> Int#
broadcastInt32# v = v

packInt32# :: (# Int# #) -> Int#
packInt32# (# v #) = v

unpackInt32# :: Int# -> (# Int# #)
unpackInt32# v = (# v #)

insertInt32# :: Int# -> Int# -> Int# -> Int#
insertInt32# _ v _ = v

negateInt32# :: Int# -> Int#
negateInt32# a = case negate (I32# a) of I32# b -> b

plusInt32# :: Int# -> Int# -> Int#
plusInt32# a b = case I32# a + I32# b of I32# c -> c

minusInt32# :: Int# -> Int# -> Int#
minusInt32# a b = case I32# a - I32# b of I32# c -> c

timesInt32# :: Int# -> Int# -> Int#
timesInt32# a b = case I32# a * I32# b of I32# c -> c

quotInt32# :: Int# -> Int# -> Int#
quotInt32# a b = case I32# a `quot` I32# b of I32# c -> c

remInt32# :: Int# -> Int# -> Int#
remInt32# a b = case I32# a `rem` I32# b of I32# c -> c

abs' :: Int32 -> Int32
abs' (I32# x) = I32# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I32# x) of
    I32# y -> y

signum' :: Int32 -> Int32
signum' (I32# x) = I32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I32# x) of
    I32# y -> y

instance Eq Int32X8 where
    a == b = case unpackInt32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt32X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Int32X8 where
    a `compare` b = case unpackInt32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt32X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Int32X8 where
    showsPrec _ a s = case unpackInt32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Int32X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Int32X8 where
    (+) = plusInt32X8
    (-) = minusInt32X8
    (*) = timesInt32X8
    negate = negateInt32X8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int32X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int32X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int32X8 where
    type Elem Int32X8 = Int32
    type ElemTuple Int32X8 = (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 4
    broadcastVector    = broadcastInt32X8
    unsafeInsertVector = unsafeInsertInt32X8
    packVector         = packInt32X8
    unpackVector       = unpackInt32X8
    mapVector          = mapInt32X8
    zipVector          = zipInt32X8
    foldVector         = foldInt32X8

instance SIMDIntVector Int32X8 where
    quotVector = quotInt32X8
    remVector  = remInt32X8

instance Prim Int32X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt32X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt32X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt32X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt32X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt32X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt32X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int32X8 = V_Int32X8 (PV.Vector Int32X8)
newtype instance UV.MVector s Int32X8 = MV_Int32X8 (PMV.MVector s Int32X8)

instance Vector UV.Vector Int32X8 where
    basicUnsafeFreeze (MV_Int32X8 v) = V_Int32X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int32X8 v) = MV_Int32X8 <$> PV.unsafeThaw v
    basicLength (V_Int32X8 v) = PV.length v
    basicUnsafeSlice start len (V_Int32X8 v) = V_Int32X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int32X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int32X8 m) (V_Int32X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int32X8 where
    basicLength (MV_Int32X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int32X8 v) = MV_Int32X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int32X8 v) (MV_Int32X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int32X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int32X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int32X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int32X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int32X8

{-# INLINE broadcastInt32X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt32X8 :: Int32 -> Int32X8
broadcastInt32X8 (I32# x) = case broadcastInt32# x of
    v -> Int32X8 v v v v v v v v

{-# INLINE packInt32X8 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X8 :: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) -> Int32X8
packInt32X8 (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8) = Int32X8 (packInt32# (# x1 #)) (packInt32# (# x2 #)) (packInt32# (# x3 #)) (packInt32# (# x4 #)) (packInt32# (# x5 #)) (packInt32# (# x6 #)) (packInt32# (# x7 #)) (packInt32# (# x8 #))

{-# INLINE unpackInt32X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X8 :: Int32X8 -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
unpackInt32X8 (Int32X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackInt32# m1 of
    (# x1 #) -> case unpackInt32# m2 of
        (# x2 #) -> case unpackInt32# m3 of
            (# x3 #) -> case unpackInt32# m4 of
                (# x4 #) -> case unpackInt32# m5 of
                    (# x5 #) -> case unpackInt32# m6 of
                        (# x6 #) -> case unpackInt32# m7 of
                            (# x7 #) -> case unpackInt32# m8 of
                                (# x8 #) -> (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8)

{-# INLINE unsafeInsertInt32X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X8 :: Int32X8 -> Int32 -> Int -> Int32X8
unsafeInsertInt32X8 (Int32X8 m1 m2 m3 m4 m5 m6 m7 m8) (I32# y) _i@(I# ip) | _i < 1 = Int32X8 (insertInt32# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                          | _i < 2 = Int32X8 m1 (insertInt32# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                          | _i < 3 = Int32X8 m1 m2 (insertInt32# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                          | _i < 4 = Int32X8 m1 m2 m3 (insertInt32# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                          | _i < 5 = Int32X8 m1 m2 m3 m4 (insertInt32# m5 y (ip -# 4#)) m6 m7 m8
                                                                          | _i < 6 = Int32X8 m1 m2 m3 m4 m5 (insertInt32# m6 y (ip -# 5#)) m7 m8
                                                                          | _i < 7 = Int32X8 m1 m2 m3 m4 m5 m6 (insertInt32# m7 y (ip -# 6#)) m8
                                                                          | otherwise = Int32X8 m1 m2 m3 m4 m5 m6 m7 (insertInt32# m8 y (ip -# 7#))

{-# INLINE[1] mapInt32X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt32X8 :: (Int32 -> Int32) -> Int32X8 -> Int32X8
mapInt32X8 f = mapInt32X8# (\ x -> case f (I32# x) of { I32# y -> y})

{-# RULES "mapVector abs" mapInt32X8 abs = abs #-}
{-# RULES "mapVector signum" mapInt32X8 signum = signum #-}
{-# RULES "mapVector negate" mapInt32X8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt32X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt32X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt32X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt32X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt32X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt32X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt32X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt32X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt32X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt32X8# #-}
-- | Unboxed helper function.
mapInt32X8# :: (Int# -> Int#) -> Int32X8 -> Int32X8
mapInt32X8# f = \ v -> case unpackInt32X8 v of
    (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8) -> packInt32X8 (I32# (f x1), I32# (f x2), I32# (f x3), I32# (f x4), I32# (f x5), I32# (f x6), I32# (f x7), I32# (f x8))

{-# INLINE[1] zipInt32X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt32X8 :: (Int32 -> Int32 -> Int32) -> Int32X8 -> Int32X8 -> Int32X8
zipInt32X8 f = \ v1 v2 -> case unpackInt32X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt32X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packInt32X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipInt32X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt32X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt32X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt32X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt32X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt32X8 #-}
-- | Fold the elements of a vector to a single value
foldInt32X8 :: (Int32 -> Int32 -> Int32) -> Int32X8 -> Int32
foldInt32X8 f' = \ v -> case unpackInt32X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusInt32X8 #-}
-- | Add two vectors element-wise.
plusInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
plusInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int32X8 (plusInt32# m1_1 m1_2) (plusInt32# m2_1 m2_2) (plusInt32# m3_1 m3_2) (plusInt32# m4_1 m4_2) (plusInt32# m5_1 m5_2) (plusInt32# m6_1 m6_2) (plusInt32# m7_1 m7_2) (plusInt32# m8_1 m8_2)

{-# INLINE minusInt32X8 #-}
-- | Subtract two vectors element-wise.
minusInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
minusInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int32X8 (minusInt32# m1_1 m1_2) (minusInt32# m2_1 m2_2) (minusInt32# m3_1 m3_2) (minusInt32# m4_1 m4_2) (minusInt32# m5_1 m5_2) (minusInt32# m6_1 m6_2) (minusInt32# m7_1 m7_2) (minusInt32# m8_1 m8_2)

{-# INLINE timesInt32X8 #-}
-- | Multiply two vectors element-wise.
timesInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
timesInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int32X8 (timesInt32# m1_1 m1_2) (timesInt32# m2_1 m2_2) (timesInt32# m3_1 m3_2) (timesInt32# m4_1 m4_2) (timesInt32# m5_1 m5_2) (timesInt32# m6_1 m6_2) (timesInt32# m7_1 m7_2) (timesInt32# m8_1 m8_2)

{-# INLINE quotInt32X8 #-}
-- | Rounds towards zero element-wise.
quotInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
quotInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int32X8 (quotInt32# m1_1 m1_2) (quotInt32# m2_1 m2_2) (quotInt32# m3_1 m3_2) (quotInt32# m4_1 m4_2) (quotInt32# m5_1 m5_2) (quotInt32# m6_1 m6_2) (quotInt32# m7_1 m7_2) (quotInt32# m8_1 m8_2)

{-# INLINE remInt32X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
remInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int32X8 (remInt32# m1_1 m1_2) (remInt32# m2_1 m2_2) (remInt32# m3_1 m3_2) (remInt32# m4_1 m4_2) (remInt32# m5_1 m5_2) (remInt32# m6_1 m6_2) (remInt32# m7_1 m7_2) (remInt32# m8_1 m8_2)

{-# INLINE negateInt32X8 #-}
-- | Negate element-wise.
negateInt32X8 :: Int32X8 -> Int32X8
negateInt32X8 (Int32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) = Int32X8 (negateInt32# m1_1) (negateInt32# m2_1) (negateInt32# m3_1) (negateInt32# m4_1) (negateInt32# m5_1) (negateInt32# m6_1) (negateInt32# m7_1) (negateInt32# m8_1)

{-# INLINE indexInt32X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X8Array :: ByteArray -> Int -> Int32X8
indexInt32X8Array (ByteArray a) (I# i) = Int32X8 (indexInt32Array# a ((i *# 8#) +# 0#)) (indexInt32Array# a ((i *# 8#) +# 1#)) (indexInt32Array# a ((i *# 8#) +# 2#)) (indexInt32Array# a ((i *# 8#) +# 3#)) (indexInt32Array# a ((i *# 8#) +# 4#)) (indexInt32Array# a ((i *# 8#) +# 5#)) (indexInt32Array# a ((i *# 8#) +# 6#)) (indexInt32Array# a ((i *# 8#) +# 7#))

{-# INLINE readInt32X8Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X8
readInt32X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt32Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt32Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt32Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt32Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt32Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt32Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt32Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Int32X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt32X8Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X8 -> m ()
writeInt32X8Array (MutableByteArray a) (I# i) (Int32X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt32Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeInt32Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexInt32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X8OffAddr :: Addr -> Int -> Int32X8
indexInt32X8OffAddr (Addr a) (I# i) = Int32X8 (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#)

{-# INLINE readInt32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X8OffAddr :: PrimMonad m => Addr -> Int -> m Int32X8
readInt32X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s7 of
                                (# s8, m8 #) -> (# s8, Int32X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt32X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X8OffAddr :: PrimMonad m => Addr -> Int -> Int32X8 -> m ()
writeInt32X8OffAddr (Addr a) (I# i) (Int32X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m2) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m3) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m4) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m5) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m6) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m7) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m8)


