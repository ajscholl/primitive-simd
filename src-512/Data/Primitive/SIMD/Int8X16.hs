{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X16 (Int8X16) where

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

-- ** Int8X16
data Int8X16 = Int8X16 Int8X16# deriving Typeable

abs' :: Int8 -> Int8
abs' (I8# x) = I8# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I8# x) of
    I8# y -> y

signum' :: Int8 -> Int8
signum' (I8# x) = I8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I8# x) of
    I8# y -> y

instance Eq Int8X16 where
    a == b = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int8X16 where
    a `compare` b = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int8X16 where
    showsPrec _ a s = case unpackInt8X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int8X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int8X16 where
    (+) = plusInt8X16
    (-) = minusInt8X16
    (*) = timesInt8X16
    negate = negateInt8X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X16 where
    type Elem Int8X16 = Int8
    type ElemTuple Int8X16 = (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 1
    broadcastVector    = broadcastInt8X16
    generateVector     = generateInt8X16
    unsafeInsertVector = unsafeInsertInt8X16
    packVector         = packInt8X16
    unpackVector       = unpackInt8X16
    mapVector          = mapInt8X16
    zipVector          = zipInt8X16
    foldVector         = foldInt8X16

instance SIMDIntVector Int8X16 where
    quotVector = quotInt8X16
    remVector  = remInt8X16

instance Prim Int8X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X16 = V_Int8X16 (PV.Vector Int8X16)
newtype instance UV.MVector s Int8X16 = MV_Int8X16 (PMV.MVector s Int8X16)

instance Vector UV.Vector Int8X16 where
    basicUnsafeFreeze (MV_Int8X16 v) = V_Int8X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X16 v) = MV_Int8X16 <$> PV.unsafeThaw v
    basicLength (V_Int8X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X16 v) = V_Int8X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X16 m) (V_Int8X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X16 where
    basicLength (MV_Int8X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X16 v) = MV_Int8X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X16 v) (MV_Int8X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X16

{-# INLINE broadcastInt8X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X16 :: Int8 -> Int8X16
broadcastInt8X16 (I8# x) = Int8X16 (broadcastInt8X16# x)

{-# INLINE[1] generateInt8X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateInt8X16 :: (Int -> Int8) -> Int8X16
generateInt8X16 f = packInt8X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packInt8X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X16 :: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) -> Int8X16
packInt8X16 (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16) = Int8X16 (packInt8X16# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #))

{-# INLINE unpackInt8X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X16 :: Int8X16 -> (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
unpackInt8X16 (Int8X16 m1) = case unpackInt8X16# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #) -> (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16)

{-# INLINE unsafeInsertInt8X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X16 :: Int8X16 -> Int8 -> Int -> Int8X16
unsafeInsertInt8X16 (Int8X16 m1) (I8# y) _i@(I# ip) = Int8X16 (insertInt8X16# m1 y (ip -# 0#))

{-# INLINE[1] mapInt8X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X16 :: (Int8 -> Int8) -> Int8X16 -> Int8X16
mapInt8X16 f = mapInt8X16# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X16# #-}
-- | Unboxed helper function.
mapInt8X16# :: (Int# -> Int#) -> Int8X16 -> Int8X16
mapInt8X16# f = \ v -> case unpackInt8X16 v of
    (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16) -> packInt8X16 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4), I8# (f x5), I8# (f x6), I8# (f x7), I8# (f x8), I8# (f x9), I8# (f x10), I8# (f x11), I8# (f x12), I8# (f x13), I8# (f x14), I8# (f x15), I8# (f x16))

{-# INLINE[1] zipInt8X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X16 :: (Int8 -> Int8 -> Int8) -> Int8X16 -> Int8X16 -> Int8X16
zipInt8X16 f = \ v1 v2 -> case unpackInt8X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt8X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt8X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt8X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X16 #-}
-- | Fold the elements of a vector to a single value
foldInt8X16 :: (Int8 -> Int8 -> Int8) -> Int8X16 -> Int8
foldInt8X16 f' = \ v -> case unpackInt8X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusInt8X16 #-}
-- | Add two vectors element-wise.
plusInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
plusInt8X16 (Int8X16 m1_1) (Int8X16 m1_2) = Int8X16 (plusInt8X16# m1_1 m1_2)

{-# INLINE minusInt8X16 #-}
-- | Subtract two vectors element-wise.
minusInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
minusInt8X16 (Int8X16 m1_1) (Int8X16 m1_2) = Int8X16 (minusInt8X16# m1_1 m1_2)

{-# INLINE timesInt8X16 #-}
-- | Multiply two vectors element-wise.
timesInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
timesInt8X16 (Int8X16 m1_1) (Int8X16 m1_2) = Int8X16 (timesInt8X16# m1_1 m1_2)

{-# INLINE quotInt8X16 #-}
-- | Rounds towards zero element-wise.
quotInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
quotInt8X16 (Int8X16 m1_1) (Int8X16 m1_2) = Int8X16 (quotInt8X16# m1_1 m1_2)

{-# INLINE remInt8X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X16 :: Int8X16 -> Int8X16 -> Int8X16
remInt8X16 (Int8X16 m1_1) (Int8X16 m1_2) = Int8X16 (remInt8X16# m1_1 m1_2)

{-# INLINE negateInt8X16 #-}
-- | Negate element-wise.
negateInt8X16 :: Int8X16 -> Int8X16
negateInt8X16 (Int8X16 m1_1) = Int8X16 (negateInt8X16# m1_1)

{-# INLINE indexInt8X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X16Array :: ByteArray -> Int -> Int8X16
indexInt8X16Array (ByteArray a) (I# i) = Int8X16 (indexInt8X16Array# a i)

{-# INLINE readInt8X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X16
readInt8X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8X16Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X16 m1 #))

{-# INLINE writeInt8X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X16 -> m ()
writeInt8X16Array (MutableByteArray a) (I# i) (Int8X16 m1) = primitive_ (writeInt8X16Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X16OffAddr :: Addr -> Int -> Int8X16
indexInt8X16OffAddr (Addr a) (I# i) = Int8X16 (indexInt8X16OffAddr# (plusAddr# a (i *# 16#)) 0#)

{-# INLINE readInt8X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X16OffAddr :: PrimMonad m => Addr -> Int -> m Int8X16
readInt8X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X16 m1 #))

{-# INLINE writeInt8X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X16OffAddr :: PrimMonad m => Addr -> Int -> Int8X16 -> m ()
writeInt8X16OffAddr (Addr a) (I# i) (Int8X16 m1) = primitive_ (writeInt8X16OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1)


