{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X64 (Word8X64) where

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

-- ** Word8X64
data Word8X64 = Word8X64 Word8X16# Word8X16# Word8X16# Word8X16# deriving Typeable

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

instance Eq Word8X64 where
    a == b = case unpackWord8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackWord8X64 b of
            Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32 && x33 == y33 && x34 == y34 && x35 == y35 && x36 == y36 && x37 == y37 && x38 == y38 && x39 == y39 && x40 == y40 && x41 == y41 && x42 == y42 && x43 == y43 && x44 == y44 && x45 == y45 && x46 == y46 && x47 == y47 && x48 == y48 && x49 == y49 && x50 == y50 && x51 == y51 && x52 == y52 && x53 == y53 && x54 == y54 && x55 == y55 && x56 == y56 && x57 == y57 && x58 == y58 && x59 == y59 && x60 == y60 && x61 == y61 && x62 == y62 && x63 == y63 && x64 == y64

instance Ord Word8X64 where
    a `compare` b = case unpackWord8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackWord8X64 b of
            Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32 <> x33 `compare` y33 <> x34 `compare` y34 <> x35 `compare` y35 <> x36 `compare` y36 <> x37 `compare` y37 <> x38 `compare` y38 <> x39 `compare` y39 <> x40 `compare` y40 <> x41 `compare` y41 <> x42 `compare` y42 <> x43 `compare` y43 <> x44 `compare` y44 <> x45 `compare` y45 <> x46 `compare` y46 <> x47 `compare` y47 <> x48 `compare` y48 <> x49 `compare` y49 <> x50 `compare` y50 <> x51 `compare` y51 <> x52 `compare` y52 <> x53 `compare` y53 <> x54 `compare` y54 <> x55 `compare` y55 <> x56 `compare` y56 <> x57 `compare` y57 <> x58 `compare` y58 <> x59 `compare` y59 <> x60 `compare` y60 <> x61 `compare` y61 <> x62 `compare` y62 <> x63 `compare` y63 <> x64 `compare` y64

instance Show Word8X64 where
    showsPrec _ a s = case unpackWord8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> "Word8X64 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (", " ++ shows x33 (", " ++ shows x34 (", " ++ shows x35 (", " ++ shows x36 (", " ++ shows x37 (", " ++ shows x38 (", " ++ shows x39 (", " ++ shows x40 (", " ++ shows x41 (", " ++ shows x42 (", " ++ shows x43 (", " ++ shows x44 (", " ++ shows x45 (", " ++ shows x46 (", " ++ shows x47 (", " ++ shows x48 (", " ++ shows x49 (", " ++ shows x50 (", " ++ shows x51 (", " ++ shows x52 (", " ++ shows x53 (", " ++ shows x54 (", " ++ shows x55 (", " ++ shows x56 (", " ++ shows x57 (", " ++ shows x58 (", " ++ shows x59 (", " ++ shows x60 (", " ++ shows x61 (", " ++ shows x62 (", " ++ shows x63 (", " ++ shows x64 (")" ++ s))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

instance Num Word8X64 where
    (+) = plusWord8X64
    (-) = minusWord8X64
    (*) = timesWord8X64
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X64 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X64 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X64 where
    type Elem Word8X64 = Word8
    type ElemTuple Word8X64 = Tuple64 Word8
    nullVector         = broadcastVector 0
    vectorSize  _      = 64
    elementSize _      = 1
    broadcastVector    = broadcastWord8X64
    generateVector     = generateWord8X64
    unsafeInsertVector = unsafeInsertWord8X64
    packVector         = packWord8X64
    unpackVector       = unpackWord8X64
    mapVector          = mapWord8X64
    zipVector          = zipWord8X64
    foldVector         = foldWord8X64
    sumVector          = sumWord8X64

instance SIMDIntVector Word8X64 where
    quotVector = quotWord8X64
    remVector  = remWord8X64

instance Prim Word8X64 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X64Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X64Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X64Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X64OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X64OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X64OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X64 = V_Word8X64 (PV.Vector Word8X64)
newtype instance UV.MVector s Word8X64 = MV_Word8X64 (PMV.MVector s Word8X64)

instance Vector UV.Vector Word8X64 where
    basicUnsafeFreeze (MV_Word8X64 v) = V_Word8X64 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X64 v) = MV_Word8X64 <$> PV.unsafeThaw v
    basicLength (V_Word8X64 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X64 v) = V_Word8X64(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X64 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X64 m) (V_Word8X64 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X64 where
    basicLength (MV_Word8X64 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X64 v) = MV_Word8X64(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X64 v) (MV_Word8X64 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X64 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X64 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X64 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X64 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X64

{-# INLINE broadcastWord8X64 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X64 :: Word8 -> Word8X64
broadcastWord8X64 (W8# x) = case broadcastWord8X16# x of
    v -> Word8X64 v v v v

{-# INLINE[1] generateWord8X64 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
generateWord8X64 :: (Int -> Word8) -> Word8X64
generateWord8X64 f = packWord8X64 (Tuple64 (f 0) (f 1) (f 2) (f 3) (f 4) (f 5) (f 6) (f 7) (f 8) (f 9) (f 10) (f 11) (f 12) (f 13) (f 14) (f 15) (f 16) (f 17) (f 18) (f 19) (f 20) (f 21) (f 22) (f 23) (f 24) (f 25) (f 26) (f 27) (f 28) (f 29) (f 30) (f 31) (f 32) (f 33) (f 34) (f 35) (f 36) (f 37) (f 38) (f 39) (f 40) (f 41) (f 42) (f 43) (f 44) (f 45) (f 46) (f 47) (f 48) (f 49) (f 50) (f 51) (f 52) (f 53) (f 54) (f 55) (f 56) (f 57) (f 58) (f 59) (f 60) (f 61) (f 62) (f 63))

{-# INLINE packWord8X64 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X64 :: Tuple64 Word8 -> Word8X64
packWord8X64 (Tuple64 (W8# x1) (W8# x2) (W8# x3) (W8# x4) (W8# x5) (W8# x6) (W8# x7) (W8# x8) (W8# x9) (W8# x10) (W8# x11) (W8# x12) (W8# x13) (W8# x14) (W8# x15) (W8# x16) (W8# x17) (W8# x18) (W8# x19) (W8# x20) (W8# x21) (W8# x22) (W8# x23) (W8# x24) (W8# x25) (W8# x26) (W8# x27) (W8# x28) (W8# x29) (W8# x30) (W8# x31) (W8# x32) (W8# x33) (W8# x34) (W8# x35) (W8# x36) (W8# x37) (W8# x38) (W8# x39) (W8# x40) (W8# x41) (W8# x42) (W8# x43) (W8# x44) (W8# x45) (W8# x46) (W8# x47) (W8# x48) (W8# x49) (W8# x50) (W8# x51) (W8# x52) (W8# x53) (W8# x54) (W8# x55) (W8# x56) (W8# x57) (W8# x58) (W8# x59) (W8# x60) (W8# x61) (W8# x62) (W8# x63) (W8# x64)) = Word8X64 (packWord8X16# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #)) (packWord8X16# (# x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #)) (packWord8X16# (# x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48 #)) (packWord8X16# (# x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64 #))

{-# INLINE unpackWord8X64 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X64 :: Word8X64 -> Tuple64 Word8
unpackWord8X64 (Word8X64 m1 m2 m3 m4) = case unpackWord8X16# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 #) -> case unpackWord8X16# m2 of
        (# x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #) -> case unpackWord8X16# m3 of
            (# x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48 #) -> case unpackWord8X16# m4 of
                (# x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64 #) -> Tuple64 (W8# x1) (W8# x2) (W8# x3) (W8# x4) (W8# x5) (W8# x6) (W8# x7) (W8# x8) (W8# x9) (W8# x10) (W8# x11) (W8# x12) (W8# x13) (W8# x14) (W8# x15) (W8# x16) (W8# x17) (W8# x18) (W8# x19) (W8# x20) (W8# x21) (W8# x22) (W8# x23) (W8# x24) (W8# x25) (W8# x26) (W8# x27) (W8# x28) (W8# x29) (W8# x30) (W8# x31) (W8# x32) (W8# x33) (W8# x34) (W8# x35) (W8# x36) (W8# x37) (W8# x38) (W8# x39) (W8# x40) (W8# x41) (W8# x42) (W8# x43) (W8# x44) (W8# x45) (W8# x46) (W8# x47) (W8# x48) (W8# x49) (W8# x50) (W8# x51) (W8# x52) (W8# x53) (W8# x54) (W8# x55) (W8# x56) (W8# x57) (W8# x58) (W8# x59) (W8# x60) (W8# x61) (W8# x62) (W8# x63) (W8# x64)

{-# INLINE unsafeInsertWord8X64 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X64 :: Word8X64 -> Word8 -> Int -> Word8X64
unsafeInsertWord8X64 (Word8X64 m1 m2 m3 m4) (W8# y) _i@(I# ip) | _i < 16 = Word8X64 (insertWord8X16# m1 y (ip -# 0#)) m2 m3 m4
                                                               | _i < 32 = Word8X64 m1 (insertWord8X16# m2 y (ip -# 16#)) m3 m4
                                                               | _i < 48 = Word8X64 m1 m2 (insertWord8X16# m3 y (ip -# 32#)) m4
                                                               | otherwise = Word8X64 m1 m2 m3 (insertWord8X16# m4 y (ip -# 48#))

{-# INLINE[1] mapWord8X64 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X64 :: (Word8 -> Word8) -> Word8X64 -> Word8X64
mapWord8X64 f = mapWord8X64# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X64 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X64 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X64 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X64 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X64 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X64 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X64 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X64 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X64 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X64 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X64 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X64# #-}
-- | Unboxed helper function.
mapWord8X64# :: (Word# -> Word#) -> Word8X64 -> Word8X64
mapWord8X64# f = \ v -> case unpackWord8X64 v of
    Tuple64 (W8# x1) (W8# x2) (W8# x3) (W8# x4) (W8# x5) (W8# x6) (W8# x7) (W8# x8) (W8# x9) (W8# x10) (W8# x11) (W8# x12) (W8# x13) (W8# x14) (W8# x15) (W8# x16) (W8# x17) (W8# x18) (W8# x19) (W8# x20) (W8# x21) (W8# x22) (W8# x23) (W8# x24) (W8# x25) (W8# x26) (W8# x27) (W8# x28) (W8# x29) (W8# x30) (W8# x31) (W8# x32) (W8# x33) (W8# x34) (W8# x35) (W8# x36) (W8# x37) (W8# x38) (W8# x39) (W8# x40) (W8# x41) (W8# x42) (W8# x43) (W8# x44) (W8# x45) (W8# x46) (W8# x47) (W8# x48) (W8# x49) (W8# x50) (W8# x51) (W8# x52) (W8# x53) (W8# x54) (W8# x55) (W8# x56) (W8# x57) (W8# x58) (W8# x59) (W8# x60) (W8# x61) (W8# x62) (W8# x63) (W8# x64) -> packWord8X64 (Tuple64 (W8# (f x1)) (W8# (f x2)) (W8# (f x3)) (W8# (f x4)) (W8# (f x5)) (W8# (f x6)) (W8# (f x7)) (W8# (f x8)) (W8# (f x9)) (W8# (f x10)) (W8# (f x11)) (W8# (f x12)) (W8# (f x13)) (W8# (f x14)) (W8# (f x15)) (W8# (f x16)) (W8# (f x17)) (W8# (f x18)) (W8# (f x19)) (W8# (f x20)) (W8# (f x21)) (W8# (f x22)) (W8# (f x23)) (W8# (f x24)) (W8# (f x25)) (W8# (f x26)) (W8# (f x27)) (W8# (f x28)) (W8# (f x29)) (W8# (f x30)) (W8# (f x31)) (W8# (f x32)) (W8# (f x33)) (W8# (f x34)) (W8# (f x35)) (W8# (f x36)) (W8# (f x37)) (W8# (f x38)) (W8# (f x39)) (W8# (f x40)) (W8# (f x41)) (W8# (f x42)) (W8# (f x43)) (W8# (f x44)) (W8# (f x45)) (W8# (f x46)) (W8# (f x47)) (W8# (f x48)) (W8# (f x49)) (W8# (f x50)) (W8# (f x51)) (W8# (f x52)) (W8# (f x53)) (W8# (f x54)) (W8# (f x55)) (W8# (f x56)) (W8# (f x57)) (W8# (f x58)) (W8# (f x59)) (W8# (f x60)) (W8# (f x61)) (W8# (f x62)) (W8# (f x63)) (W8# (f x64)))

{-# INLINE[1] zipWord8X64 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X64 :: (Word8 -> Word8 -> Word8) -> Word8X64 -> Word8X64 -> Word8X64
zipWord8X64 f = \ v1 v2 -> case unpackWord8X64 v1 of
    Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackWord8X64 v2 of
        Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> packWord8X64 (Tuple64 (f x1 y1) (f x2 y2) (f x3 y3) (f x4 y4) (f x5 y5) (f x6 y6) (f x7 y7) (f x8 y8) (f x9 y9) (f x10 y10) (f x11 y11) (f x12 y12) (f x13 y13) (f x14 y14) (f x15 y15) (f x16 y16) (f x17 y17) (f x18 y18) (f x19 y19) (f x20 y20) (f x21 y21) (f x22 y22) (f x23 y23) (f x24 y24) (f x25 y25) (f x26 y26) (f x27 y27) (f x28 y28) (f x29 y29) (f x30 y30) (f x31 y31) (f x32 y32) (f x33 y33) (f x34 y34) (f x35 y35) (f x36 y36) (f x37 y37) (f x38 y38) (f x39 y39) (f x40 y40) (f x41 y41) (f x42 y42) (f x43 y43) (f x44 y44) (f x45 y45) (f x46 y46) (f x47 y47) (f x48 y48) (f x49 y49) (f x50 y50) (f x51 y51) (f x52 y52) (f x53 y53) (f x54 y54) (f x55 y55) (f x56 y56) (f x57 y57) (f x58 y58) (f x59 y59) (f x60 y60) (f x61 y61) (f x62 y62) (f x63 y63) (f x64 y64))

{-# RULES "zipVector +" forall a b . zipWord8X64 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X64 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X64 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X64 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X64 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X64 #-}
-- | Fold the elements of a vector to a single value
foldWord8X64 :: (Word8 -> Word8 -> Word8) -> Word8X64 -> Word8
foldWord8X64 f' = \ v -> case unpackWord8X64 v of
    Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32 `f` x33 `f` x34 `f` x35 `f` x36 `f` x37 `f` x38 `f` x39 `f` x40 `f` x41 `f` x42 `f` x43 `f` x44 `f` x45 `f` x46 `f` x47 `f` x48 `f` x49 `f` x50 `f` x51 `f` x52 `f` x53 `f` x54 `f` x55 `f` x56 `f` x57 `f` x58 `f` x59 `f` x60 `f` x61 `f` x62 `f` x63 `f` x64
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldWord8X64 (+) = sumVector #-}

{-# INLINE sumWord8X64 #-}
-- | Sum up the elements of a vector to a single value.
sumWord8X64 :: Word8X64 -> Word8
sumWord8X64 (Word8X64 x1 x2 x3 x4) = case unpackWord8X16# (plusWord8X16# x1 (plusWord8X16# x2 (plusWord8X16# x3 x4))) of
    (# y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 #) -> W8# y1 + W8# y2 + W8# y3 + W8# y4 + W8# y5 + W8# y6 + W8# y7 + W8# y8 + W8# y9 + W8# y10 + W8# y11 + W8# y12 + W8# y13 + W8# y14 + W8# y15 + W8# y16

{-# INLINE plusWord8X64 #-}
-- | Add two vectors element-wise.
plusWord8X64 :: Word8X64 -> Word8X64 -> Word8X64
plusWord8X64 (Word8X64 m1_1 m2_1 m3_1 m4_1) (Word8X64 m1_2 m2_2 m3_2 m4_2) = Word8X64 (plusWord8X16# m1_1 m1_2) (plusWord8X16# m2_1 m2_2) (plusWord8X16# m3_1 m3_2) (plusWord8X16# m4_1 m4_2)

{-# INLINE minusWord8X64 #-}
-- | Subtract two vectors element-wise.
minusWord8X64 :: Word8X64 -> Word8X64 -> Word8X64
minusWord8X64 (Word8X64 m1_1 m2_1 m3_1 m4_1) (Word8X64 m1_2 m2_2 m3_2 m4_2) = Word8X64 (minusWord8X16# m1_1 m1_2) (minusWord8X16# m2_1 m2_2) (minusWord8X16# m3_1 m3_2) (minusWord8X16# m4_1 m4_2)

{-# INLINE timesWord8X64 #-}
-- | Multiply two vectors element-wise.
timesWord8X64 :: Word8X64 -> Word8X64 -> Word8X64
timesWord8X64 (Word8X64 m1_1 m2_1 m3_1 m4_1) (Word8X64 m1_2 m2_2 m3_2 m4_2) = Word8X64 (timesWord8X16# m1_1 m1_2) (timesWord8X16# m2_1 m2_2) (timesWord8X16# m3_1 m3_2) (timesWord8X16# m4_1 m4_2)

{-# INLINE quotWord8X64 #-}
-- | Rounds towards zero element-wise.
quotWord8X64 :: Word8X64 -> Word8X64 -> Word8X64
quotWord8X64 (Word8X64 m1_1 m2_1 m3_1 m4_1) (Word8X64 m1_2 m2_2 m3_2 m4_2) = Word8X64 (quotWord8X16# m1_1 m1_2) (quotWord8X16# m2_1 m2_2) (quotWord8X16# m3_1 m3_2) (quotWord8X16# m4_1 m4_2)

{-# INLINE remWord8X64 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X64 :: Word8X64 -> Word8X64 -> Word8X64
remWord8X64 (Word8X64 m1_1 m2_1 m3_1 m4_1) (Word8X64 m1_2 m2_2 m3_2 m4_2) = Word8X64 (remWord8X16# m1_1 m1_2) (remWord8X16# m2_1 m2_2) (remWord8X16# m3_1 m3_2) (remWord8X16# m4_1 m4_2)

{-# INLINE indexWord8X64Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X64Array :: ByteArray -> Int -> Word8X64
indexWord8X64Array (ByteArray a) (I# i) = Word8X64 (indexWord8X16Array# a ((i *# 4#) +# 0#)) (indexWord8X16Array# a ((i *# 4#) +# 1#)) (indexWord8X16Array# a ((i *# 4#) +# 2#)) (indexWord8X16Array# a ((i *# 4#) +# 3#))

{-# INLINE readWord8X64Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X64Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X64
readWord8X64Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8X16Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8X16Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord8X16Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord8X16Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Word8X64 m1 m2 m3 m4 #))

{-# INLINE writeWord8X64Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X64Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X64 -> m ()
writeWord8X64Array (MutableByteArray a) (I# i) (Word8X64 m1 m2 m3 m4) = primitive_ (writeWord8X16Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeWord8X16Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeWord8X16Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeWord8X16Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexWord8X64OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X64OffAddr :: Addr -> Int -> Word8X64
indexWord8X64OffAddr (Addr a) (I# i) = Word8X64 (indexWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0#) (indexWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#) (indexWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0#)

{-# INLINE readWord8X64OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X64OffAddr :: PrimMonad m => Addr -> Int -> m Word8X64
readWord8X64OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 16#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord8X16OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 48#) s3 of
                (# s4, m4 #) -> (# s4, Word8X64 m1 m2 m3 m4 #))

{-# INLINE writeWord8X64OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X64OffAddr :: PrimMonad m => Addr -> Int -> Word8X64 -> m ()
writeWord8X64OffAddr (Addr a) (I# i) (Word8X64 m1 m2 m3 m4) = primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 16#)) 0# m2) >> primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m3) >> primitive_ (writeWord8X16OffAddr# (plusAddr# a ((i *# 64#) +# 48#)) 0# m4)


