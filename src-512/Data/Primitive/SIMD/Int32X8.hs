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
data Int32X8 = Int32X8 Int32X8# deriving Typeable

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
broadcastInt32X8 (I32# x) = Int32X8 (broadcastInt32X8# x)

{-# INLINE packInt32X8 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X8 :: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) -> Int32X8
packInt32X8 (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8) = Int32X8 (packInt32X8# (# x1, x2, x3, x4, x5, x6, x7, x8 #))

{-# INLINE unpackInt32X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X8 :: Int32X8 -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
unpackInt32X8 (Int32X8 m1) = case unpackInt32X8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> (I32# x1, I32# x2, I32# x3, I32# x4, I32# x5, I32# x6, I32# x7, I32# x8)

{-# INLINE unsafeInsertInt32X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X8 :: Int32X8 -> Int32 -> Int -> Int32X8
unsafeInsertInt32X8 (Int32X8 m1) (I32# y) _i@(I# ip) = Int32X8 (insertInt32X8# m1 y (ip -# 0#))

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
plusInt32X8 (Int32X8 m1_1) (Int32X8 m1_2) = Int32X8 (plusInt32X8# m1_1 m1_2)

{-# INLINE minusInt32X8 #-}
-- | Subtract two vectors element-wise.
minusInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
minusInt32X8 (Int32X8 m1_1) (Int32X8 m1_2) = Int32X8 (minusInt32X8# m1_1 m1_2)

{-# INLINE timesInt32X8 #-}
-- | Multiply two vectors element-wise.
timesInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
timesInt32X8 (Int32X8 m1_1) (Int32X8 m1_2) = Int32X8 (timesInt32X8# m1_1 m1_2)

{-# INLINE quotInt32X8 #-}
-- | Rounds towards zero element-wise.
quotInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
quotInt32X8 (Int32X8 m1_1) (Int32X8 m1_2) = Int32X8 (quotInt32X8# m1_1 m1_2)

{-# INLINE remInt32X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X8 :: Int32X8 -> Int32X8 -> Int32X8
remInt32X8 (Int32X8 m1_1) (Int32X8 m1_2) = Int32X8 (remInt32X8# m1_1 m1_2)

{-# INLINE negateInt32X8 #-}
-- | Negate element-wise.
negateInt32X8 :: Int32X8 -> Int32X8
negateInt32X8 (Int32X8 m1_1) = Int32X8 (negateInt32X8# m1_1)

{-# INLINE indexInt32X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X8Array :: ByteArray -> Int -> Int32X8
indexInt32X8Array (ByteArray a) (I# i) = Int32X8 (indexInt32X8Array# a i)

{-# INLINE readInt32X8Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X8
readInt32X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32X8Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int32X8 m1 #))

{-# INLINE writeInt32X8Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X8 -> m ()
writeInt32X8Array (MutableByteArray a) (I# i) (Int32X8 m1) = primitive_ (writeInt32X8Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X8OffAddr :: Addr -> Int -> Int32X8
indexInt32X8OffAddr (Addr a) (I# i) = Int32X8 (indexInt32X8OffAddr# (plusAddr# a (i *# 32#)) 0#)

{-# INLINE readInt32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X8OffAddr :: PrimMonad m => Addr -> Int -> m Int32X8
readInt32X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int32X8 m1 #))

{-# INLINE writeInt32X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X8OffAddr :: PrimMonad m => Addr -> Int -> Int32X8 -> m ()
writeInt32X8OffAddr (Addr a) (I# i) (Int32X8 m1) = primitive_ (writeInt32X8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1)


