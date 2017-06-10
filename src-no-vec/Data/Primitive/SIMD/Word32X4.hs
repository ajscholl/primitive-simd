{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word32X4 (Word32X4) where

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

-- ** Word32X4
data Word32X4 = Word32X4 Word# Word# Word# Word# deriving Typeable

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

instance Eq Word32X4 where
    a == b = case unpackWord32X4 a of
        (x1, x2, x3, x4) -> case unpackWord32X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Word32X4 where
    a `compare` b = case unpackWord32X4 a of
        (x1, x2, x3, x4) -> case unpackWord32X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Word32X4 where
    showsPrec _ a s = case unpackWord32X4 a of
        (x1, x2, x3, x4) -> "Word32X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Word32X4 where
    (+) = plusWord32X4
    (-) = minusWord32X4
    (*) = timesWord32X4
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word32X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word32X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word32X4 where
    type Elem Word32X4 = Word32
    type ElemTuple Word32X4 = (Word32, Word32, Word32, Word32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 4
    broadcastVector    = broadcastWord32X4
    generateVector     = generateWord32X4
    unsafeInsertVector = unsafeInsertWord32X4
    packVector         = packWord32X4
    unpackVector       = unpackWord32X4
    mapVector          = mapWord32X4
    zipVector          = zipWord32X4
    foldVector         = foldWord32X4

instance SIMDIntVector Word32X4 where
    quotVector = quotWord32X4
    remVector  = remWord32X4

instance Prim Word32X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord32X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord32X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord32X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord32X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord32X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord32X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word32X4 = V_Word32X4 (PV.Vector Word32X4)
newtype instance UV.MVector s Word32X4 = MV_Word32X4 (PMV.MVector s Word32X4)

instance Vector UV.Vector Word32X4 where
    basicUnsafeFreeze (MV_Word32X4 v) = V_Word32X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word32X4 v) = MV_Word32X4 <$> PV.unsafeThaw v
    basicLength (V_Word32X4 v) = PV.length v
    basicUnsafeSlice start len (V_Word32X4 v) = V_Word32X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word32X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word32X4 m) (V_Word32X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word32X4 where
    basicLength (MV_Word32X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word32X4 v) = MV_Word32X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word32X4 v) (MV_Word32X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word32X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word32X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word32X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word32X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word32X4

{-# INLINE broadcastWord32X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord32X4 :: Word32 -> Word32X4
broadcastWord32X4 (W32# x) = case broadcastWord32# x of
    v -> Word32X4 v v v v

{-# INLINE[1] generateWord32X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord32X4 :: (Int -> Word32) -> Word32X4
generateWord32X4 f = packWord32X4 (f 0, f 1, f 2, f 3)

{-# INLINE packWord32X4 #-}
-- | Pack the elements of a tuple into a vector.
packWord32X4 :: (Word32, Word32, Word32, Word32) -> Word32X4
packWord32X4 (W32# x1, W32# x2, W32# x3, W32# x4) = Word32X4 (packWord32# (# x1 #)) (packWord32# (# x2 #)) (packWord32# (# x3 #)) (packWord32# (# x4 #))

{-# INLINE unpackWord32X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord32X4 :: Word32X4 -> (Word32, Word32, Word32, Word32)
unpackWord32X4 (Word32X4 m1 m2 m3 m4) = case unpackWord32# m1 of
    (# x1 #) -> case unpackWord32# m2 of
        (# x2 #) -> case unpackWord32# m3 of
            (# x3 #) -> case unpackWord32# m4 of
                (# x4 #) -> (W32# x1, W32# x2, W32# x3, W32# x4)

{-# INLINE unsafeInsertWord32X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord32X4 :: Word32X4 -> Word32 -> Int -> Word32X4
unsafeInsertWord32X4 (Word32X4 m1 m2 m3 m4) (W32# y) _i@(I# ip) | _i < 1 = Word32X4 (insertWord32# m1 y (ip -# 0#)) m2 m3 m4
                                                                | _i < 2 = Word32X4 m1 (insertWord32# m2 y (ip -# 1#)) m3 m4
                                                                | _i < 3 = Word32X4 m1 m2 (insertWord32# m3 y (ip -# 2#)) m4
                                                                | otherwise = Word32X4 m1 m2 m3 (insertWord32# m4 y (ip -# 3#))

{-# INLINE[1] mapWord32X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord32X4 :: (Word32 -> Word32) -> Word32X4 -> Word32X4
mapWord32X4 f = mapWord32X4# (\ x -> case f (W32# x) of { W32# y -> y})

{-# RULES "mapVector abs" mapWord32X4 abs = abs #-}
{-# RULES "mapVector signum" mapWord32X4 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord32X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord32X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord32X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord32X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord32X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord32X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord32X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord32X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord32X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord32X4# #-}
-- | Unboxed helper function.
mapWord32X4# :: (Word# -> Word#) -> Word32X4 -> Word32X4
mapWord32X4# f = \ v -> case unpackWord32X4 v of
    (W32# x1, W32# x2, W32# x3, W32# x4) -> packWord32X4 (W32# (f x1), W32# (f x2), W32# (f x3), W32# (f x4))

{-# INLINE[1] zipWord32X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord32X4 :: (Word32 -> Word32 -> Word32) -> Word32X4 -> Word32X4 -> Word32X4
zipWord32X4 f = \ v1 v2 -> case unpackWord32X4 v1 of
    (x1, x2, x3, x4) -> case unpackWord32X4 v2 of
        (y1, y2, y3, y4) -> packWord32X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipWord32X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord32X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord32X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord32X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord32X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord32X4 #-}
-- | Fold the elements of a vector to a single value
foldWord32X4 :: (Word32 -> Word32 -> Word32) -> Word32X4 -> Word32
foldWord32X4 f' = \ v -> case unpackWord32X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusWord32X4 #-}
-- | Add two vectors element-wise.
plusWord32X4 :: Word32X4 -> Word32X4 -> Word32X4
plusWord32X4 (Word32X4 m1_1 m2_1 m3_1 m4_1) (Word32X4 m1_2 m2_2 m3_2 m4_2) = Word32X4 (plusWord32# m1_1 m1_2) (plusWord32# m2_1 m2_2) (plusWord32# m3_1 m3_2) (plusWord32# m4_1 m4_2)

{-# INLINE minusWord32X4 #-}
-- | Subtract two vectors element-wise.
minusWord32X4 :: Word32X4 -> Word32X4 -> Word32X4
minusWord32X4 (Word32X4 m1_1 m2_1 m3_1 m4_1) (Word32X4 m1_2 m2_2 m3_2 m4_2) = Word32X4 (minusWord32# m1_1 m1_2) (minusWord32# m2_1 m2_2) (minusWord32# m3_1 m3_2) (minusWord32# m4_1 m4_2)

{-# INLINE timesWord32X4 #-}
-- | Multiply two vectors element-wise.
timesWord32X4 :: Word32X4 -> Word32X4 -> Word32X4
timesWord32X4 (Word32X4 m1_1 m2_1 m3_1 m4_1) (Word32X4 m1_2 m2_2 m3_2 m4_2) = Word32X4 (timesWord32# m1_1 m1_2) (timesWord32# m2_1 m2_2) (timesWord32# m3_1 m3_2) (timesWord32# m4_1 m4_2)

{-# INLINE quotWord32X4 #-}
-- | Rounds towards zero element-wise.
quotWord32X4 :: Word32X4 -> Word32X4 -> Word32X4
quotWord32X4 (Word32X4 m1_1 m2_1 m3_1 m4_1) (Word32X4 m1_2 m2_2 m3_2 m4_2) = Word32X4 (quotWord32# m1_1 m1_2) (quotWord32# m2_1 m2_2) (quotWord32# m3_1 m3_2) (quotWord32# m4_1 m4_2)

{-# INLINE remWord32X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord32X4 :: Word32X4 -> Word32X4 -> Word32X4
remWord32X4 (Word32X4 m1_1 m2_1 m3_1 m4_1) (Word32X4 m1_2 m2_2 m3_2 m4_2) = Word32X4 (remWord32# m1_1 m1_2) (remWord32# m2_1 m2_2) (remWord32# m3_1 m3_2) (remWord32# m4_1 m4_2)

{-# INLINE indexWord32X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord32X4Array :: ByteArray -> Int -> Word32X4
indexWord32X4Array (ByteArray a) (I# i) = Word32X4 (indexWord32Array# a ((i *# 4#) +# 0#)) (indexWord32Array# a ((i *# 4#) +# 1#)) (indexWord32Array# a ((i *# 4#) +# 2#)) (indexWord32Array# a ((i *# 4#) +# 3#))

{-# INLINE readWord32X4Array #-}
-- | Read a vector from specified index of the mutable array.
readWord32X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word32X4
readWord32X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord32Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord32Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord32Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord32Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Word32X4 m1 m2 m3 m4 #))

{-# INLINE writeWord32X4Array #-}
-- | Write a vector to specified index of mutable array.
writeWord32X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word32X4 -> m ()
writeWord32X4Array (MutableByteArray a) (I# i) (Word32X4 m1 m2 m3 m4) = primitive_ (writeWord32Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeWord32Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeWord32Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeWord32Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexWord32X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord32X4OffAddr :: Addr -> Int -> Word32X4
indexWord32X4OffAddr (Addr a) (I# i) = Word32X4 (indexWord32OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0#) (indexWord32OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0#)

{-# INLINE readWord32X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord32X4OffAddr :: PrimMonad m => Addr -> Int -> m Word32X4
readWord32X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 4#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 8#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord32OffAddr# (plusAddr# addr i') 0#) a ((i *# 16#) +# 12#) s3 of
                (# s4, m4 #) -> (# s4, Word32X4 m1 m2 m3 m4 #))

{-# INLINE writeWord32X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord32X4OffAddr :: PrimMonad m => Addr -> Int -> Word32X4 -> m ()
writeWord32X4OffAddr (Addr a) (I# i) (Word32X4 m1 m2 m3 m4) = primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 16#) +# 0#)) 0# m1) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 16#) +# 4#)) 0# m2) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 16#) +# 8#)) 0# m3) >> primitive_ (writeWord32OffAddr# (plusAddr# a ((i *# 16#) +# 12#)) 0# m4)


