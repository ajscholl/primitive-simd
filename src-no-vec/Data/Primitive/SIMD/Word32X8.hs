{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word32X8 (Word32X8) where

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

-- ** Word32X8
data Word32X8 = Word32X8 Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord32# :: Word# -> Word#
broadcastWord32# v = v

packWord32# :: (# Word# #) -> Word#
packWord32# (# v #) = v

unpackWord32# :: Word# -> (# Word# #)
unpackWord32# v = (# v #)

insertWord32# :: Word# -> Word# -> Int# -> Word#
insertWord32# _ v _ = v

plusWord32# :: Word# -> Word# -> Word#
plusWord32# a b = case W32# a + W32# b of W32# c -> c

minusWord32# :: Word# -> Word# -> Word#
minusWord32# a b = case W32# a - W32# b of W32# c -> c

timesWord32# :: Word# -> Word# -> Word#
timesWord32# a b = case W32# a * W32# b of W32# c -> c

quotWord32# :: Word# -> Word# -> Word#
quotWord32# a b = case W32# a `quot` W32# b of W32# c -> c

remWord32# :: Word# -> Word# -> Word#
remWord32# a b = case W32# a `rem` W32# b of W32# c -> c

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

instance Eq Word32X8 where
    a == b = case unpackWord32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord32X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Word32X8 where
    a `compare` b = case unpackWord32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord32X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Word32X8 where
    showsPrec _ a s = case unpackWord32X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Word32X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Word32X8 where
    (+) = plusWord32X8
    (-) = minusWord32X8
    (*) = timesWord32X8
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word32X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word32X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word32X8 where
    type Elem Word32X8 = Word32
    type ElemTuple Word32X8 = (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 4
    broadcastVector    = broadcastWord32X8
    generateVector     = generateWord32X8
    unsafeInsertVector = unsafeInsertWord32X8
    packVector         = packWord32X8
    unpackVector       = unpackWord32X8
    mapVector          = mapWord32X8
    zipVector          = zipWord32X8
    foldVector         = foldWord32X8

instance SIMDIntVector Word32X8 where
    quotVector = quotWord32X8
    remVector  = remWord32X8

instance Prim Word32X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord32X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord32X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord32X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord32X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord32X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord32X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word32X8 = V_Word32X8 (PV.Vector Word32X8)
newtype instance UV.MVector s Word32X8 = MV_Word32X8 (PMV.MVector s Word32X8)

instance Vector UV.Vector Word32X8 where
    basicUnsafeFreeze (MV_Word32X8 v) = V_Word32X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word32X8 v) = MV_Word32X8 <$> PV.unsafeThaw v
    basicLength (V_Word32X8 v) = PV.length v
    basicUnsafeSlice start len (V_Word32X8 v) = V_Word32X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word32X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word32X8 m) (V_Word32X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word32X8 where
    basicLength (MV_Word32X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word32X8 v) = MV_Word32X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word32X8 v) (MV_Word32X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word32X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word32X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word32X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word32X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word32X8

{-# INLINE broadcastWord32X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord32X8 :: Word32 -> Word32X8
broadcastWord32X8 (W32# x) = case broadcastWord32# x of
    v -> Word32X8 v v v v v v v v

{-# INLINE[1] generateWord32X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord32X8 :: (Int -> Word32) -> Word32X8
generateWord32X8 f = packWord32X8 (f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7)

{-# INLINE packWord32X8 #-}
-- | Pack the elements of a tuple into a vector.
packWord32X8 :: (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32) -> Word32X8
packWord32X8 (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8) = Word32X8 (packWord32# (# x1 #)) (packWord32# (# x2 #)) (packWord32# (# x3 #)) (packWord32# (# x4 #)) (packWord32# (# x5 #)) (packWord32# (# x6 #)) (packWord32# (# x7 #)) (packWord32# (# x8 #))

{-# INLINE unpackWord32X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord32X8 :: Word32X8 -> (Word32, Word32, Word32, Word32, Word32, Word32, Word32, Word32)
unpackWord32X8 (Word32X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackWord32# m1 of
    (# x1 #) -> case unpackWord32# m2 of
        (# x2 #) -> case unpackWord32# m3 of
            (# x3 #) -> case unpackWord32# m4 of
                (# x4 #) -> case unpackWord32# m5 of
                    (# x5 #) -> case unpackWord32# m6 of
                        (# x6 #) -> case unpackWord32# m7 of
                            (# x7 #) -> case unpackWord32# m8 of
                                (# x8 #) -> (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8)

{-# INLINE unsafeInsertWord32X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord32X8 :: Word32X8 -> Word32 -> Int -> Word32X8
unsafeInsertWord32X8 (Word32X8 m1 m2 m3 m4 m5 m6 m7 m8) (W32# y) _i@(I# ip) | _i < 1 = Word32X8 (insertWord32# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                            | _i < 2 = Word32X8 m1 (insertWord32# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                            | _i < 3 = Word32X8 m1 m2 (insertWord32# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                            | _i < 4 = Word32X8 m1 m2 m3 (insertWord32# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                            | _i < 5 = Word32X8 m1 m2 m3 m4 (insertWord32# m5 y (ip -# 4#)) m6 m7 m8
                                                                            | _i < 6 = Word32X8 m1 m2 m3 m4 m5 (insertWord32# m6 y (ip -# 5#)) m7 m8
                                                                            | _i < 7 = Word32X8 m1 m2 m3 m4 m5 m6 (insertWord32# m7 y (ip -# 6#)) m8
                                                                            | otherwise = Word32X8 m1 m2 m3 m4 m5 m6 m7 (insertWord32# m8 y (ip -# 7#))

{-# INLINE[1] mapWord32X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord32X8 :: (Word32 -> Word32) -> Word32X8 -> Word32X8
mapWord32X8 f = mapWord32X8# (\ x -> case f (W32# x) of { W32# y -> y})

{-# RULES "mapVector abs" mapWord32X8 abs = abs #-}
{-# RULES "mapVector signum" mapWord32X8 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord32X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord32X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord32X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord32X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord32X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord32X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord32X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord32X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord32X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord32X8# #-}
-- | Unboxed helper function.
mapWord32X8# :: (Word# -> Word#) -> Word32X8 -> Word32X8
mapWord32X8# f = \ v -> case unpackWord32X8 v of
    (W32# x1, W32# x2, W32# x3, W32# x4, W32# x5, W32# x6, W32# x7, W32# x8) -> packWord32X8 (W32# (f x1), W32# (f x2), W32# (f x3), W32# (f x4), W32# (f x5), W32# (f x6), W32# (f x7), W32# (f x8))

{-# INLINE[1] zipWord32X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord32X8 :: (Word32 -> Word32 -> Word32) -> Word32X8 -> Word32X8 -> Word32X8
zipWord32X8 f = \ v1 v2 -> case unpackWord32X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord32X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packWord32X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipWord32X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord32X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord32X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord32X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord32X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord32X8 #-}
-- | Fold the elements of a vector to a single value
foldWord32X8 :: (Word32 -> Word32 -> Word32) -> Word32X8 -> Word32
foldWord32X8 f' = \ v -> case unpackWord32X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusWord32X8 #-}
-- | Add two vectors element-wise.
plusWord32X8 :: Word32X8 -> Word32X8 -> Word32X8
plusWord32X8 (Word32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word32X8 (plusWord32# m1_1 m1_2) (plusWord32# m2_1 m2_2) (plusWord32# m3_1 m3_2) (plusWord32# m4_1 m4_2) (plusWord32# m5_1 m5_2) (plusWord32# m6_1 m6_2) (plusWord32# m7_1 m7_2) (plusWord32# m8_1 m8_2)

{-# INLINE minusWord32X8 #-}
-- | Subtract two vectors element-wise.
minusWord32X8 :: Word32X8 -> Word32X8 -> Word32X8
minusWord32X8 (Word32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word32X8 (minusWord32# m1_1 m1_2) (minusWord32# m2_1 m2_2) (minusWord32# m3_1 m3_2) (minusWord32# m4_1 m4_2) (minusWord32# m5_1 m5_2) (minusWord32# m6_1 m6_2) (minusWord32# m7_1 m7_2) (minusWord32# m8_1 m8_2)

{-# INLINE timesWord32X8 #-}
-- | Multiply two vectors element-wise.
timesWord32X8 :: Word32X8 -> Word32X8 -> Word32X8
timesWord32X8 (Word32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word32X8 (timesWord32# m1_1 m1_2) (timesWord32# m2_1 m2_2) (timesWord32# m3_1 m3_2) (timesWord32# m4_1 m4_2) (timesWord32# m5_1 m5_2) (timesWord32# m6_1 m6_2) (timesWord32# m7_1 m7_2) (timesWord32# m8_1 m8_2)

{-# INLINE quotWord32X8 #-}
-- | Rounds towards zero element-wise.
quotWord32X8 :: Word32X8 -> Word32X8 -> Word32X8
quotWord32X8 (Word32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word32X8 (quotWord32# m1_1 m1_2) (quotWord32# m2_1 m2_2) (quotWord32# m3_1 m3_2) (quotWord32# m4_1 m4_2) (quotWord32# m5_1 m5_2) (quotWord32# m6_1 m6_2) (quotWord32# m7_1 m7_2) (quotWord32# m8_1 m8_2)

{-# INLINE remWord32X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord32X8 :: Word32X8 -> Word32X8 -> Word32X8
remWord32X8 (Word32X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word32X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word32X8 (remWord32# m1_1 m1_2) (remWord32# m2_1 m2_2) (remWord32# m3_1 m3_2) (remWord32# m4_1 m4_2) (remWord32# m5_1 m5_2) (remWord32# m6_1 m6_2) (remWord32# m7_1 m7_2) (remWord32# m8_1 m8_2)

{-# INLINE indexWord32X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord32X8Array :: ByteArray -> Int -> Word32X8
indexWord32X8Array (ByteArray a) (I# i) = Word32X8 (indexWord32Array# a ((i *# 8#) +# 0#)) (indexWord32Array# a ((i *# 8#) +# 1#)) (indexWord32Array# a ((i *# 8#) +# 2#)) (indexWord32Array# a ((i *# 8#) +# 3#)) (indexWord32Array# a ((i *# 8#) +# 4#)) (indexWord32Array# a ((i *# 8#) +# 5#)) (indexWord32Array# a ((i *# 8#) +# 6#)) (indexWord32Array# a ((i *# 8#) +# 7#))

{-# INLINE readWord32X8Array #-}
-- | Read a vector from specified index of the mutable array.
readWord32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word32X8
readWord32X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord32Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord32Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord32Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord32Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord32Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord32Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord32Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord32Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Word32X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord32X8Array #-}
-- | Write a vector to specified index of mutable array.
writeWord32X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word32X8 -> m ()
writeWord32X8Array (MutableByteArray a) (I# i) (Word32X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord32Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeWord32Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexWord32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord32X8OffAddr :: Addr -> Int -> Word32X8
indexWord32X8OffAddr (Addr a) (I# i) = Word32X8 (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0#)

{-# INLINE readWord32X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord32X8OffAddr :: PrimMonad m => Addr -> Int -> m Word32X8
readWord32X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 12#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 16#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 20#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 24#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 32#) +# 28#) s7 of
                                (# s8, m8 #) -> (# s8, Word32X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord32X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord32X8OffAddr :: PrimMonad m => Addr -> Int -> Word32X8 -> m ()
writeWord32X8OffAddr (Addr a) (I# i) (Word32X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 0#)) 0# m1) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 4#)) 0# m2) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 8#)) 0# m3) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 12#)) 0# m4) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 16#)) 0# m5) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 20#)) 0# m6) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 24#)) 0# m7) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 32#) +# 28#)) 0# m8)


