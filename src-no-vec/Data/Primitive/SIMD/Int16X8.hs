{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X8 (Int16X8) where

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

-- ** Int16X8
data Int16X8 = Int16X8 Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt16# :: Int# -> Int#
broadcastInt16# v = v

packInt16# :: (# Int# #) -> Int#
packInt16# (# v #) = v

unpackInt16# :: Int# -> (# Int# #)
unpackInt16# v = (# v #)

insertInt16# :: Int# -> Int# -> Int# -> Int#
insertInt16# _ v _ = v

negateInt16# :: Int# -> Int#
negateInt16# a = case negate (I16# a) of I16# b -> b

plusInt16# :: Int# -> Int# -> Int#
plusInt16# a b = case I16# a + I16# b of I16# c -> c

minusInt16# :: Int# -> Int# -> Int#
minusInt16# a b = case I16# a - I16# b of I16# c -> c

timesInt16# :: Int# -> Int# -> Int#
timesInt16# a b = case I16# a * I16# b of I16# c -> c

quotInt16# :: Int# -> Int# -> Int#
quotInt16# a b = case I16# a `quot` I16# b of I16# c -> c

remInt16# :: Int# -> Int# -> Int#
remInt16# a b = case I16# a `rem` I16# b of I16# c -> c

abs' :: Int16 -> Int16
abs' (I16# x) = I16# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I16# x) of
    I16# y -> y

signum' :: Int16 -> Int16
signum' (I16# x) = I16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I16# x) of
    I16# y -> y

instance Eq Int16X8 where
    a == b = case unpackInt16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt16X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Int16X8 where
    a `compare` b = case unpackInt16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt16X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Int16X8 where
    showsPrec _ a s = case unpackInt16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Int16X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Int16X8 where
    (+) = plusInt16X8
    (-) = minusInt16X8
    (*) = timesInt16X8
    negate = negateInt16X8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X8 where
    type Elem Int16X8 = Int16
    type ElemTuple Int16X8 = (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 2
    broadcastVector    = broadcastInt16X8
    unsafeInsertVector = unsafeInsertInt16X8
    packVector         = packInt16X8
    unpackVector       = unpackInt16X8
    mapVector          = mapInt16X8
    zipVector          = zipInt16X8
    foldVector         = foldInt16X8

instance SIMDIntVector Int16X8 where
    quotVector = quotInt16X8
    remVector  = remInt16X8

instance Prim Int16X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X8 = V_Int16X8 (PV.Vector Int16X8)
newtype instance UV.MVector s Int16X8 = MV_Int16X8 (PMV.MVector s Int16X8)

instance Vector UV.Vector Int16X8 where
    basicUnsafeFreeze (MV_Int16X8 v) = V_Int16X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X8 v) = MV_Int16X8 <$> PV.unsafeThaw v
    basicLength (V_Int16X8 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X8 v) = V_Int16X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X8 m) (V_Int16X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X8 where
    basicLength (MV_Int16X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X8 v) = MV_Int16X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X8 v) (MV_Int16X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X8

{-# INLINE broadcastInt16X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X8 :: Int16 -> Int16X8
broadcastInt16X8 (I16# x) = case broadcastInt16# x of
    v -> Int16X8 v v v v v v v v

{-# INLINE packInt16X8 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X8 :: (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16) -> Int16X8
packInt16X8 (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8) = Int16X8 (packInt16# (# x1 #)) (packInt16# (# x2 #)) (packInt16# (# x3 #)) (packInt16# (# x4 #)) (packInt16# (# x5 #)) (packInt16# (# x6 #)) (packInt16# (# x7 #)) (packInt16# (# x8 #))

{-# INLINE unpackInt16X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X8 :: Int16X8 -> (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
unpackInt16X8 (Int16X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackInt16# m1 of
    (# x1 #) -> case unpackInt16# m2 of
        (# x2 #) -> case unpackInt16# m3 of
            (# x3 #) -> case unpackInt16# m4 of
                (# x4 #) -> case unpackInt16# m5 of
                    (# x5 #) -> case unpackInt16# m6 of
                        (# x6 #) -> case unpackInt16# m7 of
                            (# x7 #) -> case unpackInt16# m8 of
                                (# x8 #) -> (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8)

{-# INLINE unsafeInsertInt16X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X8 :: Int16X8 -> Int16 -> Int -> Int16X8
unsafeInsertInt16X8 (Int16X8 m1 m2 m3 m4 m5 m6 m7 m8) (I16# y) _i@(I# ip) | _i < 1 = Int16X8 (insertInt16# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                          | _i < 2 = Int16X8 m1 (insertInt16# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                          | _i < 3 = Int16X8 m1 m2 (insertInt16# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                          | _i < 4 = Int16X8 m1 m2 m3 (insertInt16# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                          | _i < 5 = Int16X8 m1 m2 m3 m4 (insertInt16# m5 y (ip -# 4#)) m6 m7 m8
                                                                          | _i < 6 = Int16X8 m1 m2 m3 m4 m5 (insertInt16# m6 y (ip -# 5#)) m7 m8
                                                                          | _i < 7 = Int16X8 m1 m2 m3 m4 m5 m6 (insertInt16# m7 y (ip -# 6#)) m8
                                                                          | otherwise = Int16X8 m1 m2 m3 m4 m5 m6 m7 (insertInt16# m8 y (ip -# 7#))

{-# INLINE[1] mapInt16X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X8 :: (Int16 -> Int16) -> Int16X8 -> Int16X8
mapInt16X8 f = mapInt16X8# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X8 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X8 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X8# #-}
-- | Unboxed helper function.
mapInt16X8# :: (Int# -> Int#) -> Int16X8 -> Int16X8
mapInt16X8# f = \ v -> case unpackInt16X8 v of
    (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8) -> packInt16X8 (I16# (f x1), I16# (f x2), I16# (f x3), I16# (f x4), I16# (f x5), I16# (f x6), I16# (f x7), I16# (f x8))

{-# INLINE[1] zipInt16X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X8 :: (Int16 -> Int16 -> Int16) -> Int16X8 -> Int16X8 -> Int16X8
zipInt16X8 f = \ v1 v2 -> case unpackInt16X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt16X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packInt16X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipInt16X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X8 #-}
-- | Fold the elements of a vector to a single value
foldInt16X8 :: (Int16 -> Int16 -> Int16) -> Int16X8 -> Int16
foldInt16X8 f' = \ v -> case unpackInt16X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusInt16X8 #-}
-- | Add two vectors element-wise.
plusInt16X8 :: Int16X8 -> Int16X8 -> Int16X8
plusInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int16X8 (plusInt16# m1_1 m1_2) (plusInt16# m2_1 m2_2) (plusInt16# m3_1 m3_2) (plusInt16# m4_1 m4_2) (plusInt16# m5_1 m5_2) (plusInt16# m6_1 m6_2) (plusInt16# m7_1 m7_2) (plusInt16# m8_1 m8_2)

{-# INLINE minusInt16X8 #-}
-- | Subtract two vectors element-wise.
minusInt16X8 :: Int16X8 -> Int16X8 -> Int16X8
minusInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int16X8 (minusInt16# m1_1 m1_2) (minusInt16# m2_1 m2_2) (minusInt16# m3_1 m3_2) (minusInt16# m4_1 m4_2) (minusInt16# m5_1 m5_2) (minusInt16# m6_1 m6_2) (minusInt16# m7_1 m7_2) (minusInt16# m8_1 m8_2)

{-# INLINE timesInt16X8 #-}
-- | Multiply two vectors element-wise.
timesInt16X8 :: Int16X8 -> Int16X8 -> Int16X8
timesInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int16X8 (timesInt16# m1_1 m1_2) (timesInt16# m2_1 m2_2) (timesInt16# m3_1 m3_2) (timesInt16# m4_1 m4_2) (timesInt16# m5_1 m5_2) (timesInt16# m6_1 m6_2) (timesInt16# m7_1 m7_2) (timesInt16# m8_1 m8_2)

{-# INLINE quotInt16X8 #-}
-- | Rounds towards zero element-wise.
quotInt16X8 :: Int16X8 -> Int16X8 -> Int16X8
quotInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int16X8 (quotInt16# m1_1 m1_2) (quotInt16# m2_1 m2_2) (quotInt16# m3_1 m3_2) (quotInt16# m4_1 m4_2) (quotInt16# m5_1 m5_2) (quotInt16# m6_1 m6_2) (quotInt16# m7_1 m7_2) (quotInt16# m8_1 m8_2)

{-# INLINE remInt16X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X8 :: Int16X8 -> Int16X8 -> Int16X8
remInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int16X8 (remInt16# m1_1 m1_2) (remInt16# m2_1 m2_2) (remInt16# m3_1 m3_2) (remInt16# m4_1 m4_2) (remInt16# m5_1 m5_2) (remInt16# m6_1 m6_2) (remInt16# m7_1 m7_2) (remInt16# m8_1 m8_2)

{-# INLINE negateInt16X8 #-}
-- | Negate element-wise.
negateInt16X8 :: Int16X8 -> Int16X8
negateInt16X8 (Int16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) = Int16X8 (negateInt16# m1_1) (negateInt16# m2_1) (negateInt16# m3_1) (negateInt16# m4_1) (negateInt16# m5_1) (negateInt16# m6_1) (negateInt16# m7_1) (negateInt16# m8_1)

{-# INLINE indexInt16X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X8Array :: ByteArray -> Int -> Int16X8
indexInt16X8Array (ByteArray a) (I# i) = Int16X8 (indexInt16Array# a ((i *# 8#) +# 0#)) (indexInt16Array# a ((i *# 8#) +# 1#)) (indexInt16Array# a ((i *# 8#) +# 2#)) (indexInt16Array# a ((i *# 8#) +# 3#)) (indexInt16Array# a ((i *# 8#) +# 4#)) (indexInt16Array# a ((i *# 8#) +# 5#)) (indexInt16Array# a ((i *# 8#) +# 6#)) (indexInt16Array# a ((i *# 8#) +# 7#))

{-# INLINE readInt16X8Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X8
readInt16X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt16Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt16Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt16Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt16Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt16Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt16Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt16Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Int16X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt16X8Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X8 -> m ()
writeInt16X8Array (MutableByteArray a) (I# i) (Int16X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt16Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeInt16Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexInt16X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X8OffAddr :: Addr -> Int -> Int16X8
indexInt16X8OffAddr (Addr a) (I# i) = Int16X8 (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0#)

{-# INLINE readInt16X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X8OffAddr :: PrimMonad m => Addr -> Int -> m Int16X8
readInt16X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 6#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 10#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 14#) s7 of
                                (# s8, m8 #) -> (# s8, Int16X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt16X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X8OffAddr :: PrimMonad m => Addr -> Int -> Int16X8 -> m ()
writeInt16X8OffAddr (Addr a) (I# i) (Int16X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0# m2) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m3) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0# m4) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m5) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0# m6) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m7) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0# m8)


