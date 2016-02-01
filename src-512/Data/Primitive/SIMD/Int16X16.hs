{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X16 (Int16X16) where

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

-- ** Int16X16
data Int16X16 = Int16X16 Int16X16# deriving Typeable

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

instance Eq Int16X16 where
    a == b = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Int16X16 where
    a `compare` b = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Int16X16 where
    showsPrec _ a s = case unpackInt16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Int16X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Int16X16 where
    (+) = plusInt16X16
    (-) = minusInt16X16
    (*) = timesInt16X16
    negate = negateInt16X16
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X16 where
    type Elem Int16X16 = Int16
    type ElemTuple Int16X16 = (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 2
    broadcastVector    = broadcastInt16X16
    unsafeInsertVector = unsafeInsertInt16X16
    packVector         = packInt16X16
    unpackVector       = unpackInt16X16
    mapVector          = mapInt16X16
    zipVector          = zipInt16X16
    foldVector         = foldInt16X16

instance SIMDIntVector Int16X16 where
    quotVector = quotInt16X16
    remVector  = remInt16X16

instance Prim Int16X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X16 = V_Int16X16 (PV.Vector Int16X16)
newtype instance UV.MVector s Int16X16 = MV_Int16X16 (PMV.MVector s Int16X16)

instance Vector UV.Vector Int16X16 where
    basicUnsafeFreeze (MV_Int16X16 v) = V_Int16X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X16 v) = MV_Int16X16 <$> PV.unsafeThaw v
    basicLength (V_Int16X16 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X16 v) = V_Int16X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X16 m) (V_Int16X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X16 where
    basicLength (MV_Int16X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X16 v) = MV_Int16X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X16 v) (MV_Int16X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X16

{-# INLINE broadcastInt16X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X16 :: Int16 -> Int16X16
broadcastInt16X16 (I16# x) = Int16X16 (broadcastInt16X16# x)

{-# INLINE packInt16X16 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X16 :: (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16) -> Int16X16
packInt16X16 (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16) = Int16X16 (packInt16X16# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #))

{-# INLINE unpackInt16X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X16 :: Int16X16 -> (Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16)
unpackInt16X16 (Int16X16 m1) = case unpackInt16X16# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #) -> (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16)

{-# INLINE unsafeInsertInt16X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X16 :: Int16X16 -> Int16 -> Int -> Int16X16
unsafeInsertInt16X16 (Int16X16 m1) (I16# y) _i@(I# ip) = Int16X16 (insertInt16X16# m1 y (ip -# 0#))

{-# INLINE[1] mapInt16X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X16 :: (Int16 -> Int16) -> Int16X16 -> Int16X16
mapInt16X16 f = mapInt16X16# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X16 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X16 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X16 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X16# #-}
-- | Unboxed helper function.
mapInt16X16# :: (Int# -> Int#) -> Int16X16 -> Int16X16
mapInt16X16# f = \ v -> case unpackInt16X16 v of
    (I16# x1, I16# x2, I16# x3, I16# x4, I16# x5, I16# x6, I16# x7, I16# x8, I16# x9, I16# x10, I16# x11, I16# x12, I16# x13, I16# x14, I16# x15, I16# x16) -> packInt16X16 (I16# (f x1), I16# (f x2), I16# (f x3), I16# (f x4), I16# (f x5), I16# (f x6), I16# (f x7), I16# (f x8), I16# (f x9), I16# (f x10), I16# (f x11), I16# (f x12), I16# (f x13), I16# (f x14), I16# (f x15), I16# (f x16))

{-# INLINE[1] zipInt16X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X16 :: (Int16 -> Int16 -> Int16) -> Int16X16 -> Int16X16 -> Int16X16
zipInt16X16 f = \ v1 v2 -> case unpackInt16X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackInt16X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packInt16X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipInt16X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X16 #-}
-- | Fold the elements of a vector to a single value
foldInt16X16 :: (Int16 -> Int16 -> Int16) -> Int16X16 -> Int16
foldInt16X16 f' = \ v -> case unpackInt16X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# INLINE plusInt16X16 #-}
-- | Add two vectors element-wise.
plusInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
plusInt16X16 (Int16X16 m1_1) (Int16X16 m1_2) = Int16X16 (plusInt16X16# m1_1 m1_2)

{-# INLINE minusInt16X16 #-}
-- | Subtract two vectors element-wise.
minusInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
minusInt16X16 (Int16X16 m1_1) (Int16X16 m1_2) = Int16X16 (minusInt16X16# m1_1 m1_2)

{-# INLINE timesInt16X16 #-}
-- | Multiply two vectors element-wise.
timesInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
timesInt16X16 (Int16X16 m1_1) (Int16X16 m1_2) = Int16X16 (timesInt16X16# m1_1 m1_2)

{-# INLINE quotInt16X16 #-}
-- | Rounds towards zero element-wise.
quotInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
quotInt16X16 (Int16X16 m1_1) (Int16X16 m1_2) = Int16X16 (quotInt16X16# m1_1 m1_2)

{-# INLINE remInt16X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X16 :: Int16X16 -> Int16X16 -> Int16X16
remInt16X16 (Int16X16 m1_1) (Int16X16 m1_2) = Int16X16 (remInt16X16# m1_1 m1_2)

{-# INLINE negateInt16X16 #-}
-- | Negate element-wise.
negateInt16X16 :: Int16X16 -> Int16X16
negateInt16X16 (Int16X16 m1_1) = Int16X16 (negateInt16X16# m1_1)

{-# INLINE indexInt16X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X16Array :: ByteArray -> Int -> Int16X16
indexInt16X16Array (ByteArray a) (I# i) = Int16X16 (indexInt16X16Array# a i)

{-# INLINE readInt16X16Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X16
readInt16X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16X16Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int16X16 m1 #))

{-# INLINE writeInt16X16Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X16 -> m ()
writeInt16X16Array (MutableByteArray a) (I# i) (Int16X16 m1) = primitive_ (writeInt16X16Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X16OffAddr :: Addr -> Int -> Int16X16
indexInt16X16OffAddr (Addr a) (I# i) = Int16X16 (indexInt16X16OffAddr# (plusAddr# a (i *# 32#)) 0#)

{-# INLINE readInt16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X16OffAddr :: PrimMonad m => Addr -> Int -> m Int16X16
readInt16X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int16X16 m1 #))

{-# INLINE writeInt16X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X16OffAddr :: PrimMonad m => Addr -> Int -> Int16X16 -> m ()
writeInt16X16OffAddr (Addr a) (I# i) (Int16X16 m1) = primitive_ (writeInt16X16OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1)


