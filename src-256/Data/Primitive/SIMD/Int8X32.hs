{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X32 (Int8X32) where

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

-- ** Int8X32
data Int8X32 = Int8X32 Int8X32# deriving Typeable

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

instance Eq Int8X32 where
    a == b = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Int8X32 where
    a `compare` b = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Int8X32 where
    showsPrec _ a s = case unpackInt8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Int8X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Int8X32 where
    (+) = plusInt8X32
    (-) = minusInt8X32
    (*) = timesInt8X32
    negate = negateInt8X32
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X32 where
    type Elem Int8X32 = Int8
    type ElemTuple Int8X32 = (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 1
    broadcastVector    = broadcastInt8X32
    unsafeInsertVector = unsafeInsertInt8X32
    packVector         = packInt8X32
    unpackVector       = unpackInt8X32
    mapVector          = mapInt8X32
    zipVector          = zipInt8X32
    foldVector         = foldInt8X32

instance SIMDIntVector Int8X32 where
    quotVector = quotInt8X32
    remVector  = remInt8X32

instance Prim Int8X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X32 = V_Int8X32 (PV.Vector Int8X32)
newtype instance UV.MVector s Int8X32 = MV_Int8X32 (PMV.MVector s Int8X32)

instance Vector UV.Vector Int8X32 where
    basicUnsafeFreeze (MV_Int8X32 v) = V_Int8X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X32 v) = MV_Int8X32 <$> PV.unsafeThaw v
    basicLength (V_Int8X32 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X32 v) = V_Int8X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X32 m) (V_Int8X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X32 where
    basicLength (MV_Int8X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X32 v) = MV_Int8X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X32 v) (MV_Int8X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X32

{-# INLINE broadcastInt8X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X32 :: Int8 -> Int8X32
broadcastInt8X32 (I8# x) = Int8X32 (broadcastInt8X32# x)

{-# INLINE packInt8X32 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X32 :: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) -> Int8X32
packInt8X32 (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32) = Int8X32 (packInt8X32# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #))

{-# INLINE unpackInt8X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X32 :: Int8X32 -> (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
unpackInt8X32 (Int8X32 m1) = case unpackInt8X32# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #) -> (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32)

{-# INLINE unsafeInsertInt8X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X32 :: Int8X32 -> Int8 -> Int -> Int8X32
unsafeInsertInt8X32 (Int8X32 m1) (I8# y) _i@(I# ip) = Int8X32 (insertInt8X32# m1 y (ip -# 0#))

{-# INLINE[1] mapInt8X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X32 :: (Int8 -> Int8) -> Int8X32 -> Int8X32
mapInt8X32 f = mapInt8X32# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X32 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X32 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X32 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X32# #-}
-- | Unboxed helper function.
mapInt8X32# :: (Int# -> Int#) -> Int8X32 -> Int8X32
mapInt8X32# f = \ v -> case unpackInt8X32 v of
    (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8, I8# x9, I8# x10, I8# x11, I8# x12, I8# x13, I8# x14, I8# x15, I8# x16, I8# x17, I8# x18, I8# x19, I8# x20, I8# x21, I8# x22, I8# x23, I8# x24, I8# x25, I8# x26, I8# x27, I8# x28, I8# x29, I8# x30, I8# x31, I8# x32) -> packInt8X32 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4), I8# (f x5), I8# (f x6), I8# (f x7), I8# (f x8), I8# (f x9), I8# (f x10), I8# (f x11), I8# (f x12), I8# (f x13), I8# (f x14), I8# (f x15), I8# (f x16), I8# (f x17), I8# (f x18), I8# (f x19), I8# (f x20), I8# (f x21), I8# (f x22), I8# (f x23), I8# (f x24), I8# (f x25), I8# (f x26), I8# (f x27), I8# (f x28), I8# (f x29), I8# (f x30), I8# (f x31), I8# (f x32))

{-# INLINE[1] zipInt8X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X32 :: (Int8 -> Int8 -> Int8) -> Int8X32 -> Int8X32 -> Int8X32
zipInt8X32 f = \ v1 v2 -> case unpackInt8X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackInt8X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packInt8X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipInt8X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X32 #-}
-- | Fold the elements of a vector to a single value
foldInt8X32 :: (Int8 -> Int8 -> Int8) -> Int8X32 -> Int8
foldInt8X32 f' = \ v -> case unpackInt8X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# INLINE plusInt8X32 #-}
-- | Add two vectors element-wise.
plusInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
plusInt8X32 (Int8X32 m1_1) (Int8X32 m1_2) = Int8X32 (plusInt8X32# m1_1 m1_2)

{-# INLINE minusInt8X32 #-}
-- | Subtract two vectors element-wise.
minusInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
minusInt8X32 (Int8X32 m1_1) (Int8X32 m1_2) = Int8X32 (minusInt8X32# m1_1 m1_2)

{-# INLINE timesInt8X32 #-}
-- | Multiply two vectors element-wise.
timesInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
timesInt8X32 (Int8X32 m1_1) (Int8X32 m1_2) = Int8X32 (timesInt8X32# m1_1 m1_2)

{-# INLINE quotInt8X32 #-}
-- | Rounds towards zero element-wise.
quotInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
quotInt8X32 (Int8X32 m1_1) (Int8X32 m1_2) = Int8X32 (quotInt8X32# m1_1 m1_2)

{-# INLINE remInt8X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X32 :: Int8X32 -> Int8X32 -> Int8X32
remInt8X32 (Int8X32 m1_1) (Int8X32 m1_2) = Int8X32 (remInt8X32# m1_1 m1_2)

{-# INLINE negateInt8X32 #-}
-- | Negate element-wise.
negateInt8X32 :: Int8X32 -> Int8X32
negateInt8X32 (Int8X32 m1_1) = Int8X32 (negateInt8X32# m1_1)

{-# INLINE indexInt8X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X32Array :: ByteArray -> Int -> Int8X32
indexInt8X32Array (ByteArray a) (I# i) = Int8X32 (indexInt8X32Array# a i)

{-# INLINE readInt8X32Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X32
readInt8X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8X32Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X32 m1 #))

{-# INLINE writeInt8X32Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X32 -> m ()
writeInt8X32Array (MutableByteArray a) (I# i) (Int8X32 m1) = primitive_ (writeInt8X32Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X32OffAddr :: Addr -> Int -> Int8X32
indexInt8X32OffAddr (Addr a) (I# i) = Int8X32 (indexInt8X32OffAddr# (plusAddr# a (i *# 32#)) 0#)

{-# INLINE readInt8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X32OffAddr :: PrimMonad m => Addr -> Int -> m Int8X32
readInt8X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8X32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X32 m1 #))

{-# INLINE writeInt8X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X32OffAddr :: PrimMonad m => Addr -> Int -> Int8X32 -> m ()
writeInt8X32OffAddr (Addr a) (I# i) (Int8X32 m1) = primitive_ (writeInt8X32OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1)


