{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X32 (Word8X32) where

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

-- ** Word8X32
data Word8X32 = Word8X32 Word8X16# Word8X16# deriving Typeable

abs' :: Word8 -> Word8
abs' (W8# x) = W8# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W8# x) of
    W8# y -> y

signum' :: Word8 -> Word8
signum' (W8# x) = W8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W8# x) of
    W8# y -> y

instance Eq Word8X32 where
    a == b = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32

instance Ord Word8X32 where
    a `compare` b = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32

instance Show Word8X32 where
    showsPrec _ a s = case unpackWord8X32 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> "Word8X32 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (")" ++ s))))))))))))))))))))))))))))))))

instance Num Word8X32 where
    (+) = plusWord8X32
    (-) = minusWord8X32
    (*) = timesWord8X32
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X32 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X32 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X32 where
    type Elem Word8X32 = Word8
    type ElemTuple Word8X32 = (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 32
    elementSize _      = 1
    broadcastVector    = broadcastWord8X32
    unsafeInsertVector = unsafeInsertWord8X32
    packVector         = packWord8X32
    unpackVector       = unpackWord8X32
    mapVector          = mapWord8X32
    zipVector          = zipWord8X32
    foldVector         = foldWord8X32
    sumVector          = sumWord8X32

instance SIMDIntVector Word8X32 where
    quotVector = quotWord8X32
    remVector  = remWord8X32

instance Prim Word8X32 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X32Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X32Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X32Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X32OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X32OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X32OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X32 = V_Word8X32 (PV.Vector Word8X32)
newtype instance UV.MVector s Word8X32 = MV_Word8X32 (PMV.MVector s Word8X32)

instance Vector UV.Vector Word8X32 where
    basicUnsafeFreeze (MV_Word8X32 v) = V_Word8X32 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X32 v) = MV_Word8X32 <$> PV.unsafeThaw v
    basicLength (V_Word8X32 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X32 v) = V_Word8X32(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X32 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X32 m) (V_Word8X32 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X32 where
    basicLength (MV_Word8X32 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X32 v) = MV_Word8X32(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X32 v) (MV_Word8X32 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X32 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X32 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X32 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X32 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X32

{-# INLINE broadcastWord8X32 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X32 :: Word8 -> Word8X32
broadcastWord8X32 (W8# x) = case broadcastWord8X16# x of
    v -> Word8X32 v v

{-# INLINE packWord8X32 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X32 :: (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8) -> Word8X32
packWord8X32 (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32) = Word8X32 (packWord8X16# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #)) (packWord8X16# (# x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #))

{-# INLINE unpackWord8X32 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X32 :: Word8X32 -> (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
unpackWord8X32 (Word8X32 m1 m2) = case unpackWord8X16# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #) -> case unpackWord8X16# m2 of
        (# x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #) -> (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32)

{-# INLINE unsafeInsertWord8X32 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X32 :: Word8X32 -> Word8 -> Int -> Word8X32
unsafeInsertWord8X32 (Word8X32 m1 m2) (W8# y) _i@(I# ip) | _i < 16 = Word8X32 (insertWord8X16# m1 y (ip -# 0#)) m2
                                                         | otherwise = Word8X32 m1 (insertWord8X16# m2 y (ip -# 16#))

{-# INLINE[1] mapWord8X32 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X32 :: (Word8 -> Word8) -> Word8X32 -> Word8X32
mapWord8X32 f = mapWord8X32# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X32 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X32 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X32 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X32 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X32 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X32 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X32 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X32 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X32 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X32 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X32 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X32# #-}
-- | Unboxed helper function.
mapWord8X32# :: (Word# -> Word#) -> Word8X32 -> Word8X32
mapWord8X32# f = \ v -> case unpackWord8X32 v of
    (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8, W8# x9, W8# x10, W8# x11, W8# x12, W8# x13, W8# x14, W8# x15, W8# x16, W8# x17, W8# x18, W8# x19, W8# x20, W8# x21, W8# x22, W8# x23, W8# x24, W8# x25, W8# x26, W8# x27, W8# x28, W8# x29, W8# x30, W8# x31, W8# x32) -> packWord8X32 (W8# (f x1), W8# (f x2), W8# (f x3), W8# (f x4), W8# (f x5), W8# (f x6), W8# (f x7), W8# (f x8), W8# (f x9), W8# (f x10), W8# (f x11), W8# (f x12), W8# (f x13), W8# (f x14), W8# (f x15), W8# (f x16), W8# (f x17), W8# (f x18), W8# (f x19), W8# (f x20), W8# (f x21), W8# (f x22), W8# (f x23), W8# (f x24), W8# (f x25), W8# (f x26), W8# (f x27), W8# (f x28), W8# (f x29), W8# (f x30), W8# (f x31), W8# (f x32))

{-# INLINE[1] zipWord8X32 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X32 :: (Word8 -> Word8 -> Word8) -> Word8X32 -> Word8X32 -> Word8X32
zipWord8X32 f = \ v1 v2 -> case unpackWord8X32 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> case unpackWord8X32 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32) -> packWord8X32 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16, f x17 y17, f x18 y18, f x19 y19, f x20 y20, f x21 y21, f x22 y22, f x23 y23, f x24 y24, f x25 y25, f x26 y26, f x27 y27, f x28 y28, f x29 y29, f x30 y30, f x31 y31, f x32 y32)

{-# RULES "zipVector +" forall a b . zipWord8X32 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X32 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X32 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X32 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X32 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X32 #-}
-- | Fold the elements of a vector to a single value
foldWord8X32 :: (Word8 -> Word8 -> Word8) -> Word8X32 -> Word8
foldWord8X32 f' = \ v -> case unpackWord8X32 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldWord8X32 (+) = sumVector #-}

{-# INLINE sumWord8X32 #-}
-- | Sum up the elements of a vector to a single value.
sumWord8X32 :: Word8X32 -> Word8
sumWord8X32 (Word8X32 x1 x2) = case unpackWord8X16# (plusWord8X16# x1 x2) of
    (# y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 #) -> W8# y1 + W8# y2 + W8# y3 + W8# y4 + W8# y5 + W8# y6 + W8# y7 + W8# y8 + W8# y9 + W8# y10 + W8# y11 + W8# y12 + W8# y13 + W8# y14 + W8# y15 + W8# y16

{-# INLINE plusWord8X32 #-}
-- | Add two vectors element-wise.
plusWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
plusWord8X32 (Word8X32 m1_1 m2_1) (Word8X32 m1_2 m2_2) = Word8X32 (plusWord8X16# m1_1 m1_2) (plusWord8X16# m2_1 m2_2)

{-# INLINE minusWord8X32 #-}
-- | Subtract two vectors element-wise.
minusWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
minusWord8X32 (Word8X32 m1_1 m2_1) (Word8X32 m1_2 m2_2) = Word8X32 (minusWord8X16# m1_1 m1_2) (minusWord8X16# m2_1 m2_2)

{-# INLINE timesWord8X32 #-}
-- | Multiply two vectors element-wise.
timesWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
timesWord8X32 (Word8X32 m1_1 m2_1) (Word8X32 m1_2 m2_2) = Word8X32 (timesWord8X16# m1_1 m1_2) (timesWord8X16# m2_1 m2_2)

{-# INLINE quotWord8X32 #-}
-- | Rounds towards zero element-wise.
quotWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
quotWord8X32 (Word8X32 m1_1 m2_1) (Word8X32 m1_2 m2_2) = Word8X32 (quotWord8X16# m1_1 m1_2) (quotWord8X16# m2_1 m2_2)

{-# INLINE remWord8X32 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X32 :: Word8X32 -> Word8X32 -> Word8X32
remWord8X32 (Word8X32 m1_1 m2_1) (Word8X32 m1_2 m2_2) = Word8X32 (remWord8X16# m1_1 m1_2) (remWord8X16# m2_1 m2_2)

{-# INLINE indexWord8X32Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X32Array :: ByteArray -> Int -> Word8X32
indexWord8X32Array (ByteArray a) (I# i) = Word8X32 (indexWord8X16Array# a ((i *# 2#) +# 0#)) (indexWord8X16Array# a ((i *# 2#) +# 1#))

{-# INLINE readWord8X32Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X32
readWord8X32Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8X16Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8X16Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Word8X32 m1 m2 #))

{-# INLINE writeWord8X32Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X32Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X32 -> m ()
writeWord8X32Array (MutableByteArray a) (I# i) (Word8X32 m1 m2) = primitive_ (writeWord8X16Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeWord8X16Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexWord8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X32OffAddr :: Addr -> Int -> Word8X32
indexWord8X32OffAddr (Addr a) (I# i) = Word8X32 (indexWord8X16OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexWord8X16OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#)

{-# INLINE readWord8X32OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X32OffAddr :: PrimMonad m => Addr -> Int -> m Word8X32
readWord8X32OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s1 of
        (# s2, m2 #) -> (# s2, Word8X32 m1 m2 #))

{-# INLINE writeWord8X32OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X32OffAddr :: PrimMonad m => Addr -> Int -> Word8X32 -> m ()
writeWord8X32OffAddr (Addr a) (I# i) (Word8X32 m1 m2) = primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m2)


