{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word32X16 (Word32X16) where

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

-- ** Word32X16
data Word32X16 = Word32X16 Word32X4# Word32X4# Word32X4# Word32X4# deriving Typeable

abs' :: Word32 -> Word32
abs' (W32# x) = W32# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W32# x) of
    W32# y -> y

signum' :: Word32 -> Word32
signum' (W32# x) = W32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W32# x) of
    W32# y -> y

instance Eq Word32X16 where
    a == b = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16

instance Ord Word32X16 where
    a `compare` b = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 b of
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16

instance Show Word32X16 where
    showsPrec _ a s = case unpackWord32X16 a of
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> "Word32X16 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (")" ++ s))))))))))))))))

instance Num Word32X16 where
    (+) = plusWord32X16
    (-) = minusWord32X16
    (*) = timesWord32X16
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word32X16 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word32X16 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word32X16 where
    type Elem Word32X16 = Word32
    type ElemTuple Word32X16 = (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 16
    elementSize _      = 4
    broadcastVector    = broadcastWord32X16
    generateVector     = generateWord32X16
    unsafeInsertVector = unsafeInsertWord32X16
    packVector         = packWord32X16
    unpackVector       = unpackWord32X16
    mapVector          = mapWord32X16
    zipVector          = zipWord32X16
    foldVector         = foldWord32X16
    sumVector          = sumWord32X16

instance SIMDIntVector Word32X16 where
    quotVector = quotWord32X16
    remVector  = remWord32X16

instance Prim Word32X16 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord32X16Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord32X16Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord32X16Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord32X16OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord32X16OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord32X16OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word32X16 = V_Word32X16 (PV.Vector Word32X16)
newtype instance UV.MVector s Word32X16 = MV_Word32X16 (PMV.MVector s Word32X16)

instance Vector UV.Vector Word32X16 where
    basicUnsafeFreeze (MV_Word32X16 v) = V_Word32X16 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word32X16 v) = MV_Word32X16 <$> PV.unsafeThaw v
    basicLength (V_Word32X16 v) = PV.length v
    basicUnsafeSlice start len (V_Word32X16 v) = V_Word32X16(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word32X16 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word32X16 m) (V_Word32X16 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word32X16 where
    basicLength (MV_Word32X16 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word32X16 v) = MV_Word32X16(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word32X16 v) (MV_Word32X16 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word32X16 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word32X16 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word32X16 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word32X16 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word32X16

{-# INLINE broadcastWord32X16 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord32X16 :: Word32 -> Word32X16
broadcastWord32X16 (W32# x) = case broadcastWord32X4# x of
    v -> Word32X16 v v v v

{-# INLINE[1] generateWord32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord32X16 :: (Int -> Word32) -> Word32X16
generateWord32X16 f = packWord32X16 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7, f 8, f 9, f 10, f 11, f 12, f 13, f 14, f 15)

{-# INLINE packWord32X16 #-}
-- | Pack the elements of a tuple into a vector.
packWord32X16 :: (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32) -> Word32X16
packWord32X16 (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16) = Word32X16 (packWord32X4# (# x1, x2, x3, x4 #)) (packWord32X4# (# x5, x6, x7, x8 #)) (packWord32X4# (# x9, x10, x11, x12 #)) (packWord32X4# (# x13, x14, x15, x16 #))

{-# INLINE unpackWord32X16 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord32X16 :: Word32X16 -> (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
unpackWord32X16 (Word32X16 m1 m2 m3 m4) = case unpackWord32X4# m1 of
    (# x1, x2, x3, x4 #) -> case unpackWord32X4# m2 of
        (# x5, x6, x7, x8 #) -> case unpackWord32X4# m3 of
            (# x9, x10, x11, x12 #) -> case unpackWord32X4# m4 of
                (# x13, x14, x15, x16 #) -> (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16)

{-# INLINE unsafeInsertWord32X16 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord32X16 :: Word32X16 -> Word32 -> Int -> Word32X16
unsafeInsertWord32X16 (Word32X16 m1 m2 m3 m4) (W32# y) _i@(I# ip) | _i < 4 = Word32X16 (insertWord32X4# m1 y (ip -# 0#)) m2 m3 m4
                                                                  | _i < 8 = Word32X16 m1 (insertWord32X4# m2 y (ip -# 4#)) m3 m4
                                                                  | _i < 12 = Word32X16 m1 m2 (insertWord32X4# m3 y (ip -# 8#)) m4
                                                                  | otherwise = Word32X16 m1 m2 m3 (insertWord32X4# m4 y (ip -# 12#))

{-# INLINE[1] mapWord32X16 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord32X16 :: (Word32 -> Word32) -> Word32X16 -> Word32X16
mapWord32X16 f = mapWord32X16# (\ x -> case f (W32# x) of { W32# y -> y})

{-# RULES "mapVector abs" mapWord32X16 abs = abs #-}
{-# RULES "mapVector signum" mapWord32X16 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord32X16 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord32X16 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord32X16 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord32X16 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord32X16 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord32X16 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord32X16 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord32X16 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord32X16 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord32X16# #-}
-- | Unboxed helper function.
mapWord32X16# :: (Word# -> Word#) -> Word32X16 -> Word32X16
mapWord32X16# f = \ v -> case unpackWord32X16 v of
    (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8, W32# x9, W32# x10, W32# x11, W32# x12, W32# x13, W32# x14, W32# x15, W32# x16) -> packWord32X16 (W32# (f x1), W32# (f x2), W32# (f x3), W32# (f x4), W32# (f x5), W32# (f x6), W32# (f x7), W32# (f x8), W32# (f x9), W32# (f x10), W32# (f x11), W32# (f x12), W32# (f x13), W32# (f x14), W32# (f x15), W32# (f x16))

{-# INLINE[1] zipWord32X16 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord32X16 :: (Word32 -> Word32 -> Word32) -> Word32X16 -> Word32X16 -> Word32X16
zipWord32X16 f = \ v1 v2 -> case unpackWord32X16 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> case unpackWord32X16 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16) -> packWord32X16 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8, f x9 y9, f x10 y10, f x11 y11, f x12 y12, f x13 y13, f x14 y14, f x15 y15, f x16 y16)

{-# RULES "zipVector +" forall a b . zipWord32X16 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord32X16 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord32X16 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord32X16 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord32X16 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord32X16 #-}
-- | Fold the elements of a vector to a single value
foldWord32X16 :: (Word32 -> Word32 -> Word32) -> Word32X16 -> Word32
foldWord32X16 f' = \ v -> case unpackWord32X16 v of
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldWord32X16 (+) = sumVector #-}

{-# INLINE sumWord32X16 #-}
-- | Sum up the elements of a vector to a single value.
sumWord32X16 :: Word32X16 -> Word32
sumWord32X16 (Word32X16 x1 x2 x3 x4) = case unpackWord32X4# (plusWord32X4# x1 (plusWord32X4# x2 (plusWord32X4# x3 x4))) of
    (# y1, y2, y3, y4 #) -> W32# y1 + W32# y2 + W32# y3 + W32# y4

{-# INLINE plusWord32X16 #-}
-- | Add two vectors element-wise.
plusWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
plusWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1) (Word32X16 m1_2 m2_2 m3_2 m4_2) = Word32X16 (plusWord32X4# m1_1 m1_2) (plusWord32X4# m2_1 m2_2) (plusWord32X4# m3_1 m3_2) (plusWord32X4# m4_1 m4_2)

{-# INLINE minusWord32X16 #-}
-- | Subtract two vectors element-wise.
minusWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
minusWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1) (Word32X16 m1_2 m2_2 m3_2 m4_2) = Word32X16 (minusWord32X4# m1_1 m1_2) (minusWord32X4# m2_1 m2_2) (minusWord32X4# m3_1 m3_2) (minusWord32X4# m4_1 m4_2)

{-# INLINE timesWord32X16 #-}
-- | Multiply two vectors element-wise.
timesWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
timesWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1) (Word32X16 m1_2 m2_2 m3_2 m4_2) = Word32X16 (timesWord32X4# m1_1 m1_2) (timesWord32X4# m2_1 m2_2) (timesWord32X4# m3_1 m3_2) (timesWord32X4# m4_1 m4_2)

{-# INLINE quotWord32X16 #-}
-- | Rounds towards zero element-wise.
quotWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
quotWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1) (Word32X16 m1_2 m2_2 m3_2 m4_2) = Word32X16 (quotWord32X4# m1_1 m1_2) (quotWord32X4# m2_1 m2_2) (quotWord32X4# m3_1 m3_2) (quotWord32X4# m4_1 m4_2)

{-# INLINE remWord32X16 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord32X16 :: Word32X16 -> Word32X16 -> Word32X16
remWord32X16 (Word32X16 m1_1 m2_1 m3_1 m4_1) (Word32X16 m1_2 m2_2 m3_2 m4_2) = Word32X16 (remWord32X4# m1_1 m1_2) (remWord32X4# m2_1 m2_2) (remWord32X4# m3_1 m3_2) (remWord32X4# m4_1 m4_2)

{-# INLINE indexWord32X16Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord32X16Array :: ByteArray -> Int -> Word32X16
indexWord32X16Array (ByteArray a) (I# i) = Word32X16 (indexWord32X4Array# a ((i *# 4#) +# 0#)) (indexWord32X4Array# a ((i *# 4#) +# 1#)) (indexWord32X4Array# a ((i *# 4#) +# 2#)) (indexWord32X4Array# a ((i *# 4#) +# 3#))

{-# INLINE readWord32X16Array #-}
-- | Read a vector from specified index of the mutable array.
readWord32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word32X16
readWord32X16Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord32X4Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord32X4Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord32X4Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord32X4Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Word32X16 m1 m2 m3 m4 #))

{-# INLINE writeWord32X16Array #-}
-- | Write a vector to specified index of mutable array.
writeWord32X16Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word32X16 -> m ()
writeWord32X16Array (MutableByteArray a) (I# i) (Word32X16 m1 m2 m3 m4) = primitive_ (writeWord32X4Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeWord32X4Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeWord32X4Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeWord32X4Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexWord32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord32X16OffAddr :: Addr -> Int -> Word32X16
indexWord32X16OffAddr (Addr a) (I# i) = Word32X16 (indexWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#)

{-# INLINE readWord32X16OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord32X16OffAddr :: PrimMonad m => Addr -> Int -> m Word32X16
readWord32X16OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord32X4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord32X4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord32X4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord32X4OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s3 of
                (# s4, m4 #) -> (# s4, Word32X16 m1 m2 m3 m4 #))

{-# INLINE writeWord32X16OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord32X16OffAddr :: PrimMonad m => Addr -> Int -> Word32X16 -> m ()
writeWord32X16OffAddr (Addr a) (I# i) (Word32X16 m1 m2 m3 m4) = primitive_ (writeWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m2) >> primitive_ (writeWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m3) >> primitive_ (writeWord32X4OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m4)


