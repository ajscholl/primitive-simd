{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int32X4 (Int32X4) where

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

-- ** Int32X4
data Int32X4 = Int32X4 Int# Int# Int# Int# deriving Typeable

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

instance Eq Int32X4 where
    a == b = case unpackInt32X4 a of
        (x1, x2, x3, x4) -> case unpackInt32X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Int32X4 where
    a `compare` b = case unpackInt32X4 a of
        (x1, x2, x3, x4) -> case unpackInt32X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Int32X4 where
    showsPrec _ a s = case unpackInt32X4 a of
        (x1, x2, x3, x4) -> "Int32X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Int32X4 where
    (+) = plusInt32X4
    (-) = minusInt32X4
    (*) = timesInt32X4
    negate = negateInt32X4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int32X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int32X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int32X4 where
    type Elem Int32X4 = Int32
    type ElemTuple Int32X4 = (Int32, Int32, Int32, Int32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 4
    broadcastVector    = broadcastInt32X4
    unsafeInsertVector = unsafeInsertInt32X4
    packVector         = packInt32X4
    unpackVector       = unpackInt32X4
    mapVector          = mapInt32X4
    zipVector          = zipInt32X4
    foldVector         = foldInt32X4

instance SIMDIntVector Int32X4 where
    quotVector = quotInt32X4
    remVector  = remInt32X4

instance Prim Int32X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt32X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt32X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt32X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt32X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt32X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt32X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int32X4 = V_Int32X4 (PV.Vector Int32X4)
newtype instance UV.MVector s Int32X4 = MV_Int32X4 (PMV.MVector s Int32X4)

instance Vector UV.Vector Int32X4 where
    basicUnsafeFreeze (MV_Int32X4 v) = V_Int32X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int32X4 v) = MV_Int32X4 <$> PV.unsafeThaw v
    basicLength (V_Int32X4 v) = PV.length v
    basicUnsafeSlice start len (V_Int32X4 v) = V_Int32X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int32X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int32X4 m) (V_Int32X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int32X4 where
    basicLength (MV_Int32X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int32X4 v) = MV_Int32X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int32X4 v) (MV_Int32X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int32X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int32X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int32X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int32X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int32X4

{-# INLINE broadcastInt32X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt32X4 :: Int32 -> Int32X4
broadcastInt32X4 (I32# x) = case broadcastInt32# x of
    v -> Int32X4 v v v v

{-# INLINE packInt32X4 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X4 :: (Int32, Int32, Int32, Int32) -> Int32X4
packInt32X4 (I32# x1, I32# x2, I32# x3, I32# x4) = Int32X4 (packInt32# (# x1 #)) (packInt32# (# x2 #)) (packInt32# (# x3 #)) (packInt32# (# x4 #))

{-# INLINE unpackInt32X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X4 :: Int32X4 -> (Int32, Int32, Int32, Int32)
unpackInt32X4 (Int32X4 m1 m2 m3 m4) = case unpackInt32# m1 of
    (# x1 #) -> case unpackInt32# m2 of
        (# x2 #) -> case unpackInt32# m3 of
            (# x3 #) -> case unpackInt32# m4 of
                (# x4 #) -> (I32# x1, I32# x2, I32# x3, I32# x4)

{-# INLINE unsafeInsertInt32X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X4 :: Int32X4 -> Int32 -> Int -> Int32X4
unsafeInsertInt32X4 (Int32X4 m1 m2 m3 m4) (I32# y) _i@(I# ip) | _i < 1 = Int32X4 (insertInt32# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 2 = Int32X4 m1 (insertInt32# m2 y (ip -# 1#)) m3 m4
                                                              | _i < 3 = Int32X4 m1 m2 (insertInt32# m3 y (ip -# 2#)) m4
                                                              | otherwise = Int32X4 m1 m2 m3 (insertInt32# m4 y (ip -# 3#))

{-# INLINE[1] mapInt32X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt32X4 :: (Int32 -> Int32) -> Int32X4 -> Int32X4
mapInt32X4 f = mapInt32X4# (\ x -> case f (I32# x) of { I32# y -> y})

{-# RULES "mapVector abs" mapInt32X4 abs = abs #-}
{-# RULES "mapVector signum" mapInt32X4 signum = signum #-}
{-# RULES "mapVector negate" mapInt32X4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt32X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt32X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt32X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt32X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt32X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt32X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt32X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt32X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt32X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt32X4# #-}
-- | Unboxed helper function.
mapInt32X4# :: (Int# -> Int#) -> Int32X4 -> Int32X4
mapInt32X4# f = \ v -> case unpackInt32X4 v of
    (I32# x1, I32# x2, I32# x3, I32# x4) -> packInt32X4 (I32# (f x1), I32# (f x2), I32# (f x3), I32# (f x4))

{-# INLINE[1] zipInt32X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt32X4 :: (Int32 -> Int32 -> Int32) -> Int32X4 -> Int32X4 -> Int32X4
zipInt32X4 f = \ v1 v2 -> case unpackInt32X4 v1 of
    (x1, x2, x3, x4) -> case unpackInt32X4 v2 of
        (y1, y2, y3, y4) -> packInt32X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipInt32X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt32X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt32X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt32X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt32X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt32X4 #-}
-- | Fold the elements of a vector to a single value
foldInt32X4 :: (Int32 -> Int32 -> Int32) -> Int32X4 -> Int32
foldInt32X4 f' = \ v -> case unpackInt32X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusInt32X4 #-}
-- | Add two vectors element-wise.
plusInt32X4 :: Int32X4 -> Int32X4 -> Int32X4
plusInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) (Int32X4 m1_2 m2_2 m3_2 m4_2) = Int32X4 (plusInt32# m1_1 m1_2) (plusInt32# m2_1 m2_2) (plusInt32# m3_1 m3_2) (plusInt32# m4_1 m4_2)

{-# INLINE minusInt32X4 #-}
-- | Subtract two vectors element-wise.
minusInt32X4 :: Int32X4 -> Int32X4 -> Int32X4
minusInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) (Int32X4 m1_2 m2_2 m3_2 m4_2) = Int32X4 (minusInt32# m1_1 m1_2) (minusInt32# m2_1 m2_2) (minusInt32# m3_1 m3_2) (minusInt32# m4_1 m4_2)

{-# INLINE timesInt32X4 #-}
-- | Multiply two vectors element-wise.
timesInt32X4 :: Int32X4 -> Int32X4 -> Int32X4
timesInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) (Int32X4 m1_2 m2_2 m3_2 m4_2) = Int32X4 (timesInt32# m1_1 m1_2) (timesInt32# m2_1 m2_2) (timesInt32# m3_1 m3_2) (timesInt32# m4_1 m4_2)

{-# INLINE quotInt32X4 #-}
-- | Rounds towards zero element-wise.
quotInt32X4 :: Int32X4 -> Int32X4 -> Int32X4
quotInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) (Int32X4 m1_2 m2_2 m3_2 m4_2) = Int32X4 (quotInt32# m1_1 m1_2) (quotInt32# m2_1 m2_2) (quotInt32# m3_1 m3_2) (quotInt32# m4_1 m4_2)

{-# INLINE remInt32X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X4 :: Int32X4 -> Int32X4 -> Int32X4
remInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) (Int32X4 m1_2 m2_2 m3_2 m4_2) = Int32X4 (remInt32# m1_1 m1_2) (remInt32# m2_1 m2_2) (remInt32# m3_1 m3_2) (remInt32# m4_1 m4_2)

{-# INLINE negateInt32X4 #-}
-- | Negate element-wise.
negateInt32X4 :: Int32X4 -> Int32X4
negateInt32X4 (Int32X4 m1_1 m2_1 m3_1 m4_1) = Int32X4 (negateInt32# m1_1) (negateInt32# m2_1) (negateInt32# m3_1) (negateInt32# m4_1)

{-# INLINE indexInt32X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X4Array :: ByteArray -> Int -> Int32X4
indexInt32X4Array (ByteArray a) (I# i) = Int32X4 (indexInt32Array# a ((i *# 4#) +# 0#)) (indexInt32Array# a ((i *# 4#) +# 1#)) (indexInt32Array# a ((i *# 4#) +# 2#)) (indexInt32Array# a ((i *# 4#) +# 3#))

{-# INLINE readInt32X4Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X4
readInt32X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt32Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt32Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt32Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Int32X4 m1 m2 m3 m4 #))

{-# INLINE writeInt32X4Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X4 -> m ()
writeInt32X4Array (MutableByteArray a) (I# i) (Int32X4 m1 m2 m3 m4) = primitive_ (writeInt32Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeInt32Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeInt32Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeInt32Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexInt32X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X4OffAddr :: Addr -> Int -> Int32X4
indexInt32X4OffAddr (Addr a) (I# i) = Int32X4 (indexInt32OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexInt32OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#)

{-# INLINE readInt32X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X4OffAddr :: PrimMonad m => Addr -> Int -> m Int32X4
readInt32X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s3 of
                (# s4, m4 #) -> (# s4, Int32X4 m1 m2 m3 m4 #))

{-# INLINE writeInt32X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X4OffAddr :: PrimMonad m => Addr -> Int -> Int32X4 -> m ()
writeInt32X4OffAddr (Addr a) (I# i) (Int32X4 m1 m2 m3 m4) = primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m2) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m3) >> primitive_ (writeInt32OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m4)


