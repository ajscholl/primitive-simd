{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int32X16 (Int32X16) where

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

-- ** Int32X16
data Int32X16 = Int32X16 Int32X8# Int32X8# deriving Typeable

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

instance Eq Int32X16 where
    a == b = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int32X16 where
    a `compare` b = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int32X16 where
    showsPrec _ a s = case unpackInt32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int32X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int32X16 where
    (+) = plusInt32X16
    (-) = minusInt32X16
    (*) = timesInt32X16
    negate = negateInt32X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int32X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int32X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int32X16 where
    type Elem Int32X16 = Int32
    type ElemTuple Int32X16 = (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 4
    broadcastVector    = broadcastInt32X16
    unsafeInsertVector = unsafeInsertInt32X16
    packVector         = packInt32X16
    unpackVector       = unpackInt32X16
    mapVector          = mapInt32X16
    zipVector          = zipInt32X16
    foldVector         = foldInt32X16
    sumVector          = sumInt32X16

instance SIMDIntVector Int32X16 where
    quotVector = quotInt32X16
    remVector  = remInt32X16

instance Prim Int32X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt32X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt32X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt32X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt32X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt32X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt32X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int32X16 = V_Int32X16 (PV.Vector Int32X16)
newtype instance UV.MVector s Int32X16 = MV_Int32X16 (PMV.MVector s Int32X16)

instance Vector UV.Vector Int32X16 where
    basicUnsafeFreeze (MV_Int32X16 v) = V_Int32X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int32X16 v) = MV_Int32X16 <$> PV.unsafeThaw v
    basicLength (V_Int32X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int32X16 v) = V_Int32X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int32X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int32X16 m) (V_Int32X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int32X16 where
    basicLength (MV_Int32X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int32X16 v) = MV_Int32X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int32X16 v) (MV_Int32X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int32X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int32X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int32X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int32X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int32X16

{-# INLINE broadcastInt32X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt32X16 :: Int32 -> Int32X16
broadcastInt32X16 (I32# x) = case broadcastInt32X8# x of
    v -> Int32X16 v v

{-# INLINE packInt32X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X16 :: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) -> Int32X16
packInt32X16 (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16) = Int32X16 (packInt32X8# (# x1, x2, x3, x4, x5, x6, x7, x8 #)) (packInt32X8# (# x9, x10, x11, x12, x13, x14, x15, x16 #))

{-# INLINE unpackInt32X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X16 :: Int32X16 -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
unpackInt32X16 (Int32X16 m1 m2) = case unpackInt32X8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> case unpackInt32X8# m2 of
        (# x9, x10, x11, x12, x13, x14, x15, x16 #) -> (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16)

{-# INLINE unsafeInsertInt32X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X16 :: Int32X16 -> Int32 -> Int -> Int32X16
unsafeInsertInt32X16 (Int32X16 m1 m2) (I32# y) _i@(I# ip) | _i < 8 = Int32X16 (insertInt32X8# m1 y (ip -# 0#)) m2
                                                          | otherwise = Int32X16 m1 (insertInt32X8# m2 y (ip -# 8#))

{-# INLINE[1] mapInt32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt32X16 :: (Int32 -> Int32) -> Int32X16 -> Int32X16
mapInt32X16 f = mapInt32X16# (\ x -> case f (I32# x) of { I32# y -> y})

{-# RULES "mapVector abs" mapInt32X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt32X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt32X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt32X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt32X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt32X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt32X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt32X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt32X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt32X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt32X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt32X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt32X16# #-}
-- | Unboxed helper function.
mapInt32X16# :: (Int# -> Int#) -> Int32X16 -> Int32X16
mapInt32X16# f = \ v -> case unpackInt32X16 v of
    (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8, I32# x9, I32# x10, I32# x11, I32# x12, I32# x13, I32# x14, I32# x15, I32# x16) -> packInt32X16 (I32# (f x1), I32# (f x2), I32# (f x3), I32# (f x4), I32# (f x5), I32# (f x6), I32# (f x7), I32# (f x8), I32# (f x9), I32# (f x10), I32# (f x11), I32# (f x12), I32# (f x13), I32# (f x14), I32# (f x15), I32# (f x16))

{-# INLINE[1] zipInt32X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt32X16 :: (Int32 -> Int32 -> Int32) -> Int32X16 -> Int32X16 -> Int32X16
zipInt32X16 f = \ v1 v2 -> case unpackInt32X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt32X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt32X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt32X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt32X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt32X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt32X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt32X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt32X16 #-}
-- | Fold the elements of a vector to a single value
foldInt32X16 :: (Int32 -> Int32 -> Int32) -> Int32X16 -> Int32
foldInt32X16 f' = \ v -> case unpackInt32X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldInt32X16 (+) = sumVector #-}

{-# INLINE sumInt32X16 #-}
-- | Sum up the elements of a vector to a single value.
sumInt32X16 :: Int32X16 -> Int32
sumInt32X16 (Int32X16 x1 x2) = case unpackInt32X8# (plusInt32X8# x1 x2) of
    (# y1, y2, y3, y4, y5, y6, y7, y8 #) -> I32# y1 + I32# y2 + I32# y3 + I32# y4 + I32# y5 + I32# y6 + I32# y7 + I32# y8

{-# INLINE plusInt32X16 #-}
-- | Add two vectors element-wise.
plusInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
plusInt32X16 (Int32X16 m1_1 m2_1) (Int32X16 m1_2 m2_2) = Int32X16 (plusInt32X8# m1_1 m1_2) (plusInt32X8# m2_1 m2_2)

{-# INLINE minusInt32X16 #-}
-- | Subtract two vectors element-wise.
minusInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
minusInt32X16 (Int32X16 m1_1 m2_1) (Int32X16 m1_2 m2_2) = Int32X16 (minusInt32X8# m1_1 m1_2) (minusInt32X8# m2_1 m2_2)

{-# INLINE timesInt32X16 #-}
-- | Multiply two vectors element-wise.
timesInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
timesInt32X16 (Int32X16 m1_1 m2_1) (Int32X16 m1_2 m2_2) = Int32X16 (timesInt32X8# m1_1 m1_2) (timesInt32X8# m2_1 m2_2)

{-# INLINE quotInt32X16 #-}
-- | Rounds towards zero element-wise.
quotInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
quotInt32X16 (Int32X16 m1_1 m2_1) (Int32X16 m1_2 m2_2) = Int32X16 (quotInt32X8# m1_1 m1_2) (quotInt32X8# m2_1 m2_2)

{-# INLINE remInt32X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X16 :: Int32X16 -> Int32X16 -> Int32X16
remInt32X16 (Int32X16 m1_1 m2_1) (Int32X16 m1_2 m2_2) = Int32X16 (remInt32X8# m1_1 m1_2) (remInt32X8# m2_1 m2_2)

{-# INLINE negateInt32X16 #-}
-- | Negate element-wise.
negateInt32X16 :: Int32X16 -> Int32X16
negateInt32X16 (Int32X16 m1_1 m2_1) = Int32X16 (negateInt32X8# m1_1) (negateInt32X8# m2_1)

{-# INLINE indexInt32X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X16Array :: ByteArray -> Int -> Int32X16
indexInt32X16Array (ByteArray a) (I# i) = Int32X16 (indexInt32X8Array# a ((i *# 2#) +# 0#)) (indexInt32X8Array# a ((i *# 2#) +# 1#))

{-# INLINE readInt32X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X16
readInt32X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32X8Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt32X8Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Int32X16 m1 m2 #))

{-# INLINE writeInt32X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X16 -> m ()
writeInt32X16Array (MutableByteArray a) (I# i) (Int32X16 m1 m2) = primitive_ (writeInt32X8Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeInt32X8Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexInt32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X16OffAddr :: Addr -> Int -> Int32X16
indexInt32X16OffAddr (Addr a) (I# i) = Int32X16 (indexInt32X8OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexInt32X8OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#)

{-# INLINE readInt32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X16OffAddr :: PrimMonad m => Addr -> Int -> m Int32X16
readInt32X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt32X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s1 of
        (# s2, m2 #) -> (# s2, Int32X16 m1 m2 #))

{-# INLINE writeInt32X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X16OffAddr :: PrimMonad m => Addr -> Int -> Int32X16 -> m ()
writeInt32X16OffAddr (Addr a) (I# i) (Int32X16 m1 m2) = primitive_ (writeInt32X8OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeInt32X8OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m2)


