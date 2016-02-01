{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word16X16 (Word16X16) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

import GHC.Word
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

-- ** Word16X16
data Word16X16 = Word16X16 Word16X8# Word16X8# deriving Typeable

abs' :: Word16 -> Word16
abs' (W16# x) = W16# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W16# x) of
    W16# y -> y

signum' :: Word16 -> Word16
signum' (W16# x) = W16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W16# x) of
    W16# y -> y

instance Eq Word16X16 where
    a == b = case unpackWord16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Word16X16 where
    a `compare` b = case unpackWord16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord16X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Word16X16 where
    showsPrec _ a s = case unpackWord16X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Word16X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Word16X16 where
    (+) = plusWord16X16
    (-) = minusWord16X16
    (*) = timesWord16X16
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word16X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word16X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word16X16 where
    type Elem Word16X16 = Word16
    type ElemTuple Word16X16 = (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 2
    broadcastVector    = broadcastWord16X16
    unsafeInsertVector = unsafeInsertWord16X16
    packVector         = packWord16X16
    unpackVector       = unpackWord16X16
    mapVector          = mapWord16X16
    zipVector          = zipWord16X16
    foldVector         = foldWord16X16
    sumVector          = sumWord16X16

instance SIMDIntVector Word16X16 where
    quotVector = quotWord16X16
    remVector  = remWord16X16

instance Prim Word16X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord16X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord16X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord16X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord16X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord16X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord16X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word16X16 = V_Word16X16 (PV.Vector Word16X16)
newtype instance UV.MVector s Word16X16 = MV_Word16X16 (PMV.MVector s Word16X16)

instance Vector UV.Vector Word16X16 where
    basicUnsafeFreeze (MV_Word16X16 v) = V_Word16X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word16X16 v) = MV_Word16X16 <$> PV.unsafeThaw v
    basicLength (V_Word16X16 v) = PV.length v
    basicUnsafeSlice start len (V_Word16X16 v) = V_Word16X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word16X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word16X16 m) (V_Word16X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word16X16 where
    basicLength (MV_Word16X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word16X16 v) = MV_Word16X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word16X16 v) (MV_Word16X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word16X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word16X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word16X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word16X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word16X16

{-# INLINE broadcastWord16X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord16X16 :: Word16 -> Word16X16
broadcastWord16X16 (W16# x) = case broadcastWord16X8# x of
    v -> Word16X16 v v

{-# INLINE packWord16X16 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X16 :: (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16) -> Word16X16
packWord16X16 (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16) = Word16X16 (packWord16X8# (# x1, x2, x3, x4, x5, x6, x7, x8 #)) (packWord16X8# (# x9, x10, x11, x12, x13, x14, x15, x16 #))

{-# INLINE unpackWord16X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X16 :: Word16X16 -> (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
unpackWord16X16 (Word16X16 m1 m2) = case unpackWord16X8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> case unpackWord16X8# m2 of
        (# x9, x10, x11, x12, x13, x14, x15, x16 #) -> (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16)

{-# INLINE unsafeInsertWord16X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X16 :: Word16X16 -> Word16 -> Int -> Word16X16
unsafeInsertWord16X16 (Word16X16 m1 m2) (W16# y) _i@(I# ip) | _i < 8 = Word16X16 (insertWord16X8# m1 y (ip -# 0#)) m2
                                                            | otherwise = Word16X16 m1 (insertWord16X8# m2 y (ip -# 8#))

{-# INLINE[1] mapWord16X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord16X16 :: (Word16 -> Word16) -> Word16X16 -> Word16X16
mapWord16X16 f = mapWord16X16# (\ x -> case f (W16# x) of { W16# y -> y})

{-# RULES "mapVector abs" mapWord16X16 abs = abs #-}
{-# RULES "mapVector signum" mapWord16X16 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord16X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord16X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord16X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord16X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord16X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord16X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord16X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord16X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord16X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord16X16# #-}
-- | Unboxed helper function.
mapWord16X16# :: (Word# -> Word#) -> Word16X16 -> Word16X16
mapWord16X16# f = \ v -> case unpackWord16X16 v of
    (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8, W16# x9, W16# x10, W16# x11, W16# x12, W16# x13, W16# x14, W16# x15, W16# x16) -> packWord16X16 (W16# (f x1), W16# (f x2), W16# (f x3), W16# (f x4), W16# (f x5), W16# (f x6), W16# (f x7), W16# (f x8), W16# (f x9), W16# (f x10), W16# (f x11), W16# (f x12), W16# (f x13), W16# (f x14), W16# (f x15), W16# (f x16))

{-# INLINE[1] zipWord16X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord16X16 :: (Word16 -> Word16 -> Word16) -> Word16X16 -> Word16X16 -> Word16X16
zipWord16X16 f = \ v1 v2 -> case unpackWord16X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord16X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packWord16X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipWord16X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord16X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord16X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord16X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord16X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord16X16 #-}
-- | Fold the elements of a vector to a single value
foldWord16X16 :: (Word16 -> Word16 -> Word16) -> Word16X16 -> Word16
foldWord16X16 f' = \ v -> case unpackWord16X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldWord16X16 (+) = sumVector #-}

{-# INLINE sumWord16X16 #-}
-- | Sum up the elements of a vector to a single value.
sumWord16X16 :: Word16X16 -> Word16
sumWord16X16 (Word16X16 x1 x2) = case unpackWord16X8# (plusWord16X8# x1 x2) of
    (# y1, y2, y3, y4, y5, y6, y7, y8 #) -> W16# y1 + W16# y2 + W16# y3 + W16# y4 + W16# y5 + W16# y6 + W16# y7 + W16# y8

{-# INLINE plusWord16X16 #-}
-- | Add two vectors element-wise.
plusWord16X16 :: Word16X16 -> Word16X16 -> Word16X16
plusWord16X16 (Word16X16 m1_1 m2_1) (Word16X16 m1_2 m2_2) = Word16X16 (plusWord16X8# m1_1 m1_2) (plusWord16X8# m2_1 m2_2)

{-# INLINE minusWord16X16 #-}
-- | Subtract two vectors element-wise.
minusWord16X16 :: Word16X16 -> Word16X16 -> Word16X16
minusWord16X16 (Word16X16 m1_1 m2_1) (Word16X16 m1_2 m2_2) = Word16X16 (minusWord16X8# m1_1 m1_2) (minusWord16X8# m2_1 m2_2)

{-# INLINE timesWord16X16 #-}
-- | Multiply two vectors element-wise.
timesWord16X16 :: Word16X16 -> Word16X16 -> Word16X16
timesWord16X16 (Word16X16 m1_1 m2_1) (Word16X16 m1_2 m2_2) = Word16X16 (timesWord16X8# m1_1 m1_2) (timesWord16X8# m2_1 m2_2)

{-# INLINE quotWord16X16 #-}
-- | Rounds towards zero element-wise.
quotWord16X16 :: Word16X16 -> Word16X16 -> Word16X16
quotWord16X16 (Word16X16 m1_1 m2_1) (Word16X16 m1_2 m2_2) = Word16X16 (quotWord16X8# m1_1 m1_2) (quotWord16X8# m2_1 m2_2)

{-# INLINE remWord16X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X16 :: Word16X16 -> Word16X16 -> Word16X16
remWord16X16 (Word16X16 m1_1 m2_1) (Word16X16 m1_2 m2_2) = Word16X16 (remWord16X8# m1_1 m1_2) (remWord16X8# m2_1 m2_2)

{-# INLINE indexWord16X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X16Array :: ByteArray -> Int -> Word16X16
indexWord16X16Array (ByteArray a) (I# i) = Word16X16 (indexWord16X8Array# a ((i *# 2#) +# 0#)) (indexWord16X8Array# a ((i *# 2#) +# 1#))

{-# INLINE readWord16X16Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X16
readWord16X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16X8Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord16X8Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Word16X16 m1 m2 #))

{-# INLINE writeWord16X16Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X16 -> m ()
writeWord16X16Array (MutableByteArray a) (I# i) (Word16X16 m1 m2) = primitive_ (writeWord16X8Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeWord16X8Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexWord16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X16OffAddr :: Addr -> Int -> Word16X16
indexWord16X16OffAddr (Addr a) (I# i) = Word16X16 (indexWord16X8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexWord16X8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#)

{-# INLINE readWord16X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X16OffAddr :: PrimMonad m => Addr -> Int -> m Word16X16
readWord16X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord16X8OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s1 of
        (# s2, m2 #) -> (# s2, Word16X16 m1 m2 #))

{-# INLINE writeWord16X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X16OffAddr :: PrimMonad m => Addr -> Int -> Word16X16 -> m ()
writeWord16X16OffAddr (Addr a) (I# i) (Word16X16 m1 m2) = primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeWord16X8OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m2)


