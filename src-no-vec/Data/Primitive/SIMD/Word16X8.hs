{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word16X8 (Word16X8) where

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

-- ** Word16X8
data Word16X8 = Word16X8 Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord16# :: Word# -> Word#
broadcastWord16# v = v

packWord16# :: (# Word# #) -> Word#
packWord16# (# v #) = v

unpackWord16# :: Word# -> (# Word# #)
unpackWord16# v = (# v #)

insertWord16# :: Word# -> Word# -> Int# -> Word#
insertWord16# _ v _ = v

plusWord16# :: Word# -> Word# -> Word#
plusWord16# a b = case W16# a + W16# b of W16# c -> c

minusWord16# :: Word# -> Word# -> Word#
minusWord16# a b = case W16# a - W16# b of W16# c -> c

timesWord16# :: Word# -> Word# -> Word#
timesWord16# a b = case W16# a * W16# b of W16# c -> c

quotWord16# :: Word# -> Word# -> Word#
quotWord16# a b = case W16# a `quot` W16# b of W16# c -> c

remWord16# :: Word# -> Word# -> Word#
remWord16# a b = case W16# a `rem` W16# b of W16# c -> c

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

instance Eq Word16X8 where
    a == b = case unpackWord16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord16X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Word16X8 where
    a `compare` b = case unpackWord16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord16X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Word16X8 where
    showsPrec _ a s = case unpackWord16X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Word16X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Word16X8 where
    (+) = plusWord16X8
    (-) = minusWord16X8
    (*) = timesWord16X8
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word16X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word16X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word16X8 where
    type Elem Word16X8 = Word16
    type ElemTuple Word16X8 = (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 2
    broadcastVector    = broadcastWord16X8
    generateVector     = generateWord16X8
    unsafeInsertVector = unsafeInsertWord16X8
    packVector         = packWord16X8
    unpackVector       = unpackWord16X8
    mapVector          = mapWord16X8
    zipVector          = zipWord16X8
    foldVector         = foldWord16X8

instance SIMDIntVector Word16X8 where
    quotVector = quotWord16X8
    remVector  = remWord16X8

instance Prim Word16X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord16X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord16X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord16X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord16X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord16X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord16X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word16X8 = V_Word16X8 (PV.Vector Word16X8)
newtype instance UV.MVector s Word16X8 = MV_Word16X8 (PMV.MVector s Word16X8)

instance Vector UV.Vector Word16X8 where
    basicUnsafeFreeze (MV_Word16X8 v) = V_Word16X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word16X8 v) = MV_Word16X8 <$> PV.unsafeThaw v
    basicLength (V_Word16X8 v) = PV.length v
    basicUnsafeSlice start len (V_Word16X8 v) = V_Word16X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word16X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word16X8 m) (V_Word16X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word16X8 where
    basicLength (MV_Word16X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word16X8 v) = MV_Word16X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word16X8 v) (MV_Word16X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word16X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word16X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word16X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word16X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word16X8

{-# INLINE broadcastWord16X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord16X8 :: Word16 -> Word16X8
broadcastWord16X8 (W16# x) = case broadcastWord16# x of
    v -> Word16X8 v v v v v v v v

{-# INLINE[1] generateWord16X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord16X8 :: (Int -> Word16) -> Word16X8
generateWord16X8 f = packWord16X8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packWord16X8 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X8 :: (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16) -> Word16X8
packWord16X8 (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8) = Word16X8 (packWord16# (# x1 #)) (packWord16# (# x2 #)) (packWord16# (# x3 #)) (packWord16# (# x4 #)) (packWord16# (# x5 #)) (packWord16# (# x6 #)) (packWord16# (# x7 #)) (packWord16# (# x8 #))

{-# INLINE unpackWord16X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X8 :: Word16X8 -> (Word16, Word16, Word16, Word16, Word16, Word16, Word16, Word16)
unpackWord16X8 (Word16X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackWord16# m1 of
    (# x1 #) -> case unpackWord16# m2 of
        (# x2 #) -> case unpackWord16# m3 of
            (# x3 #) -> case unpackWord16# m4 of
                (# x4 #) -> case unpackWord16# m5 of
                    (# x5 #) -> case unpackWord16# m6 of
                        (# x6 #) -> case unpackWord16# m7 of
                            (# x7 #) -> case unpackWord16# m8 of
                                (# x8 #) -> (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8)

{-# INLINE unsafeInsertWord16X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X8 :: Word16X8 -> Word16 -> Int -> Word16X8
unsafeInsertWord16X8 (Word16X8 m1 m2 m3 m4 m5 m6 m7 m8) (W16# y) _i@(I# ip) | _i < 1 = Word16X8 (insertWord16# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                            | _i < 2 = Word16X8 m1 (insertWord16# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                            | _i < 3 = Word16X8 m1 m2 (insertWord16# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                            | _i < 4 = Word16X8 m1 m2 m3 (insertWord16# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                            | _i < 5 = Word16X8 m1 m2 m3 m4 (insertWord16# m5 y (ip -# 4#)) m6 m7 m8
                                                                            | _i < 6 = Word16X8 m1 m2 m3 m4 m5 (insertWord16# m6 y (ip -# 5#)) m7 m8
                                                                            | _i < 7 = Word16X8 m1 m2 m3 m4 m5 m6 (insertWord16# m7 y (ip -# 6#)) m8
                                                                            | otherwise = Word16X8 m1 m2 m3 m4 m5 m6 m7 (insertWord16# m8 y (ip -# 7#))

{-# INLINE[1] mapWord16X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord16X8 :: (Word16 -> Word16) -> Word16X8 -> Word16X8
mapWord16X8 f = mapWord16X8# (\ x -> case f (W16# x) of { W16# y -> y})

{-# RULES "mapVector abs" mapWord16X8 abs = abs #-}
{-# RULES "mapVector signum" mapWord16X8 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord16X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord16X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord16X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord16X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord16X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord16X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord16X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord16X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord16X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord16X8# #-}
-- | Unboxed helper function.
mapWord16X8# :: (Word# -> Word#) -> Word16X8 -> Word16X8
mapWord16X8# f = \ v -> case unpackWord16X8 v of
    (W16# x1, W16# x2, W16# x3, W16# x4, W16# x5, W16# x6, W16# x7, W16# x8) -> packWord16X8 (W16# (f x1), W16# (f x2), W16# (f x3), W16# (f x4), W16# (f x5), W16# (f x6), W16# (f x7), W16# (f x8))

{-# INLINE[1] zipWord16X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord16X8 :: (Word16 -> Word16 -> Word16) -> Word16X8 -> Word16X8 -> Word16X8
zipWord16X8 f = \ v1 v2 -> case unpackWord16X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord16X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packWord16X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipWord16X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord16X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord16X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord16X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord16X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord16X8 #-}
-- | Fold the elements of a vector to a single value
foldWord16X8 :: (Word16 -> Word16 -> Word16) -> Word16X8 -> Word16
foldWord16X8 f' = \ v -> case unpackWord16X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusWord16X8 #-}
-- | Add two vectors element-wise.
plusWord16X8 :: Word16X8 -> Word16X8 -> Word16X8
plusWord16X8 (Word16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word16X8 (plusWord16# m1_1 m1_2) (plusWord16# m2_1 m2_2) (plusWord16# m3_1 m3_2) (plusWord16# m4_1 m4_2) (plusWord16# m5_1 m5_2) (plusWord16# m6_1 m6_2) (plusWord16# m7_1 m7_2) (plusWord16# m8_1 m8_2)

{-# INLINE minusWord16X8 #-}
-- | Subtract two vectors element-wise.
minusWord16X8 :: Word16X8 -> Word16X8 -> Word16X8
minusWord16X8 (Word16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word16X8 (minusWord16# m1_1 m1_2) (minusWord16# m2_1 m2_2) (minusWord16# m3_1 m3_2) (minusWord16# m4_1 m4_2) (minusWord16# m5_1 m5_2) (minusWord16# m6_1 m6_2) (minusWord16# m7_1 m7_2) (minusWord16# m8_1 m8_2)

{-# INLINE timesWord16X8 #-}
-- | Multiply two vectors element-wise.
timesWord16X8 :: Word16X8 -> Word16X8 -> Word16X8
timesWord16X8 (Word16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word16X8 (timesWord16# m1_1 m1_2) (timesWord16# m2_1 m2_2) (timesWord16# m3_1 m3_2) (timesWord16# m4_1 m4_2) (timesWord16# m5_1 m5_2) (timesWord16# m6_1 m6_2) (timesWord16# m7_1 m7_2) (timesWord16# m8_1 m8_2)

{-# INLINE quotWord16X8 #-}
-- | Rounds towards zero element-wise.
quotWord16X8 :: Word16X8 -> Word16X8 -> Word16X8
quotWord16X8 (Word16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word16X8 (quotWord16# m1_1 m1_2) (quotWord16# m2_1 m2_2) (quotWord16# m3_1 m3_2) (quotWord16# m4_1 m4_2) (quotWord16# m5_1 m5_2) (quotWord16# m6_1 m6_2) (quotWord16# m7_1 m7_2) (quotWord16# m8_1 m8_2)

{-# INLINE remWord16X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X8 :: Word16X8 -> Word16X8 -> Word16X8
remWord16X8 (Word16X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word16X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word16X8 (remWord16# m1_1 m1_2) (remWord16# m2_1 m2_2) (remWord16# m3_1 m3_2) (remWord16# m4_1 m4_2) (remWord16# m5_1 m5_2) (remWord16# m6_1 m6_2) (remWord16# m7_1 m7_2) (remWord16# m8_1 m8_2)

{-# INLINE indexWord16X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X8Array :: ByteArray -> Int -> Word16X8
indexWord16X8Array (ByteArray a) (I# i) = Word16X8 (indexWord16Array# a ((i *# 8#) +# 0#)) (indexWord16Array# a ((i *# 8#) +# 1#)) (indexWord16Array# a ((i *# 8#) +# 2#)) (indexWord16Array# a ((i *# 8#) +# 3#)) (indexWord16Array# a ((i *# 8#) +# 4#)) (indexWord16Array# a ((i *# 8#) +# 5#)) (indexWord16Array# a ((i *# 8#) +# 6#)) (indexWord16Array# a ((i *# 8#) +# 7#))

{-# INLINE readWord16X8Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X8
readWord16X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord16Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord16Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord16Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord16Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord16Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord16Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord16Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Word16X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord16X8Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X8 -> m ()
writeWord16X8Array (MutableByteArray a) (I# i) (Word16X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord16Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeWord16Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexWord16X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X8OffAddr :: Addr -> Int -> Word16X8
indexWord16X8OffAddr (Addr a) (I# i) = Word16X8 (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0#)

{-# INLINE readWord16X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X8OffAddr :: PrimMonad m => Addr -> Int -> m Word16X8
readWord16X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 6#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 10#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 14#) s7 of
                                (# s8, m8 #) -> (# s8, Word16X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord16X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X8OffAddr :: PrimMonad m => Addr -> Int -> Word16X8 -> m ()
writeWord16X8OffAddr (Addr a) (I# i) (Word16X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 2#)) 0# m2) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m3) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 6#)) 0# m4) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m5) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 10#)) 0# m6) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m7) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 16#) +# 14#)) 0# m8)


