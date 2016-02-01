{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X64 (Int8X64) where

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

-- ** Int8X64
data Int8X64 = Int8X64 Int8X32# Int8X32# deriving Typeable

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

instance Eq Int8X64 where
    a == b = case unpackInt8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackInt8X64 b of
            Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8 && x9 == y9 && x10 == y10 && x11 == y11 && x12 == y12 && x13 == y13 && x14 == y14 && x15 == y15 && x16 == y16 && x17 == y17 && x18 == y18 && x19 == y19 && x20 == y20 && x21 == y21 && x22 == y22 && x23 == y23 && x24 == y24 && x25 == y25 && x26 == y26 && x27 == y27 && x28 == y28 && x29 == y29 && x30 == y30 && x31 == y31 && x32 == y32 && x33 == y33 && x34 == y34 && x35 == y35 && x36 == y36 && x37 == y37 && x38 == y38 && x39 == y39 && x40 == y40 && x41 == y41 && x42 == y42 && x43 == y43 && x44 == y44 && x45 == y45 && x46 == y46 && x47 == y47 && x48 == y48 && x49 == y49 && x50 == y50 && x51 == y51 && x52 == y52 && x53 == y53 && x54 == y54 && x55 == y55 && x56 == y56 && x57 == y57 && x58 == y58 && x59 == y59 && x60 == y60 && x61 == y61 && x62 == y62 && x63 == y63 && x64 == y64

instance Ord Int8X64 where
    a `compare` b = case unpackInt8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackInt8X64 b of
            Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8 <> x9 `compare` y9 <> x10 `compare` y10 <> x11 `compare` y11 <> x12 `compare` y12 <> x13 `compare` y13 <> x14 `compare` y14 <> x15 `compare` y15 <> x16 `compare` y16 <> x17 `compare` y17 <> x18 `compare` y18 <> x19 `compare` y19 <> x20 `compare` y20 <> x21 `compare` y21 <> x22 `compare` y22 <> x23 `compare` y23 <> x24 `compare` y24 <> x25 `compare` y25 <> x26 `compare` y26 <> x27 `compare` y27 <> x28 `compare` y28 <> x29 `compare` y29 <> x30 `compare` y30 <> x31 `compare` y31 <> x32 `compare` y32 <> x33 `compare` y33 <> x34 `compare` y34 <> x35 `compare` y35 <> x36 `compare` y36 <> x37 `compare` y37 <> x38 `compare` y38 <> x39 `compare` y39 <> x40 `compare` y40 <> x41 `compare` y41 <> x42 `compare` y42 <> x43 `compare` y43 <> x44 `compare` y44 <> x45 `compare` y45 <> x46 `compare` y46 <> x47 `compare` y47 <> x48 `compare` y48 <> x49 `compare` y49 <> x50 `compare` y50 <> x51 `compare` y51 <> x52 `compare` y52 <> x53 `compare` y53 <> x54 `compare` y54 <> x55 `compare` y55 <> x56 `compare` y56 <> x57 `compare` y57 <> x58 `compare` y58 <> x59 `compare` y59 <> x60 `compare` y60 <> x61 `compare` y61 <> x62 `compare` y62 <> x63 `compare` y63 <> x64 `compare` y64

instance Show Int8X64 where
    showsPrec _ a s = case unpackInt8X64 a of
        Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> "Int8X64 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (", " ++ shows x9 (", " ++ shows x10 (", " ++ shows x11 (", " ++ shows x12 (", " ++ shows x13 (", " ++ shows x14 (", " ++ shows x15 (", " ++ shows x16 (", " ++ shows x17 (", " ++ shows x18 (", " ++ shows x19 (", " ++ shows x20 (", " ++ shows x21 (", " ++ shows x22 (", " ++ shows x23 (", " ++ shows x24 (", " ++ shows x25 (", " ++ shows x26 (", " ++ shows x27 (", " ++ shows x28 (", " ++ shows x29 (", " ++ shows x30 (", " ++ shows x31 (", " ++ shows x32 (", " ++ shows x33 (", " ++ shows x34 (", " ++ shows x35 (", " ++ shows x36 (", " ++ shows x37 (", " ++ shows x38 (", " ++ shows x39 (", " ++ shows x40 (", " ++ shows x41 (", " ++ shows x42 (", " ++ shows x43 (", " ++ shows x44 (", " ++ shows x45 (", " ++ shows x46 (", " ++ shows x47 (", " ++ shows x48 (", " ++ shows x49 (", " ++ shows x50 (", " ++ shows x51 (", " ++ shows x52 (", " ++ shows x53 (", " ++ shows x54 (", " ++ shows x55 (", " ++ shows x56 (", " ++ shows x57 (", " ++ shows x58 (", " ++ shows x59 (", " ++ shows x60 (", " ++ shows x61 (", " ++ shows x62 (", " ++ shows x63 (", " ++ shows x64 (")" ++ s))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

instance Num Int8X64 where
    (+) = plusInt8X64
    (-) = minusInt8X64
    (*) = timesInt8X64
    negate = negateInt8X64
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X64 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X64 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X64 where
    type Elem Int8X64 = Int8
    type ElemTuple Int8X64 = Tuple64 Int8
    nullVector         = broadcastVector 0
    vectorSize  _      = 64
    elementSize _      = 1
    broadcastVector    = broadcastInt8X64
    unsafeInsertVector = unsafeInsertInt8X64
    packVector         = packInt8X64
    unpackVector       = unpackInt8X64
    mapVector          = mapInt8X64
    zipVector          = zipInt8X64
    foldVector         = foldInt8X64
    sumVector          = sumInt8X64

instance SIMDIntVector Int8X64 where
    quotVector = quotInt8X64
    remVector  = remInt8X64

instance Prim Int8X64 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X64Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X64Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X64Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X64OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X64OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X64OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X64 = V_Int8X64 (PV.Vector Int8X64)
newtype instance UV.MVector s Int8X64 = MV_Int8X64 (PMV.MVector s Int8X64)

instance Vector UV.Vector Int8X64 where
    basicUnsafeFreeze (MV_Int8X64 v) = V_Int8X64 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X64 v) = MV_Int8X64 <$> PV.unsafeThaw v
    basicLength (V_Int8X64 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X64 v) = V_Int8X64(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X64 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X64 m) (V_Int8X64 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X64 where
    basicLength (MV_Int8X64 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X64 v) = MV_Int8X64(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X64 v) (MV_Int8X64 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X64 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X64 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X64 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X64 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X64

{-# INLINE broadcastInt8X64 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X64 :: Int8 -> Int8X64
broadcastInt8X64 (I8# x) = case broadcastInt8X32# x of
    v -> Int8X64 v v

{-# INLINE packInt8X64 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X64 :: Tuple64 Int8 -> Int8X64
packInt8X64 (Tuple64 (I8# x1) (I8# x2) (I8# x3) (I8# x4) (I8# x5) (I8# x6) (I8# x7) (I8# x8) (I8# x9) (I8# x10) (I8# x11) (I8# x12) (I8# x13) (I8# x14) (I8# x15) (I8# x16) (I8# x17) (I8# x18) (I8# x19) (I8# x20) (I8# x21) (I8# x22) (I8# x23) (I8# x24) (I8# x25) (I8# x26) (I8# x27) (I8# x28) (I8# x29) (I8# x30) (I8# x31) (I8# x32) (I8# x33) (I8# x34) (I8# x35) (I8# x36) (I8# x37) (I8# x38) (I8# x39) (I8# x40) (I8# x41) (I8# x42) (I8# x43) (I8# x44) (I8# x45) (I8# x46) (I8# x47) (I8# x48) (I8# x49) (I8# x50) (I8# x51) (I8# x52) (I8# x53) (I8# x54) (I8# x55) (I8# x56) (I8# x57) (I8# x58) (I8# x59) (I8# x60) (I8# x61) (I8# x62) (I8# x63) (I8# x64)) = Int8X64 (packInt8X32# (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #)) (packInt8X32# (# x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64 #))

{-# INLINE unpackInt8X64 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X64 :: Int8X64 -> Tuple64 Int8
unpackInt8X64 (Int8X64 m1 m2) = case unpackInt8X32# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32 #) -> case unpackInt8X32# m2 of
        (# x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64 #) -> Tuple64 (I8# x1) (I8# x2) (I8# x3) (I8# x4) (I8# x5) (I8# x6) (I8# x7) (I8# x8) (I8# x9) (I8# x10) (I8# x11) (I8# x12) (I8# x13) (I8# x14) (I8# x15) (I8# x16) (I8# x17) (I8# x18) (I8# x19) (I8# x20) (I8# x21) (I8# x22) (I8# x23) (I8# x24) (I8# x25) (I8# x26) (I8# x27) (I8# x28) (I8# x29) (I8# x30) (I8# x31) (I8# x32) (I8# x33) (I8# x34) (I8# x35) (I8# x36) (I8# x37) (I8# x38) (I8# x39) (I8# x40) (I8# x41) (I8# x42) (I8# x43) (I8# x44) (I8# x45) (I8# x46) (I8# x47) (I8# x48) (I8# x49) (I8# x50) (I8# x51) (I8# x52) (I8# x53) (I8# x54) (I8# x55) (I8# x56) (I8# x57) (I8# x58) (I8# x59) (I8# x60) (I8# x61) (I8# x62) (I8# x63) (I8# x64)

{-# INLINE unsafeInsertInt8X64 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X64 :: Int8X64 -> Int8 -> Int -> Int8X64
unsafeInsertInt8X64 (Int8X64 m1 m2) (I8# y) _i@(I# ip) | _i < 32 = Int8X64 (insertInt8X32# m1 y (ip -# 0#)) m2
                                                       | otherwise = Int8X64 m1 (insertInt8X32# m2 y (ip -# 32#))

{-# INLINE[1] mapInt8X64 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X64 :: (Int8 -> Int8) -> Int8X64 -> Int8X64
mapInt8X64 f = mapInt8X64# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X64 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X64 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X64 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X64 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X64 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X64 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X64 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X64 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X64 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X64 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X64 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X64 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X64# #-}
-- | Unboxed helper function.
mapInt8X64# :: (Int# -> Int#) -> Int8X64 -> Int8X64
mapInt8X64# f = \ v -> case unpackInt8X64 v of
    Tuple64 (I8# x1) (I8# x2) (I8# x3) (I8# x4) (I8# x5) (I8# x6) (I8# x7) (I8# x8) (I8# x9) (I8# x10) (I8# x11) (I8# x12) (I8# x13) (I8# x14) (I8# x15) (I8# x16) (I8# x17) (I8# x18) (I8# x19) (I8# x20) (I8# x21) (I8# x22) (I8# x23) (I8# x24) (I8# x25) (I8# x26) (I8# x27) (I8# x28) (I8# x29) (I8# x30) (I8# x31) (I8# x32) (I8# x33) (I8# x34) (I8# x35) (I8# x36) (I8# x37) (I8# x38) (I8# x39) (I8# x40) (I8# x41) (I8# x42) (I8# x43) (I8# x44) (I8# x45) (I8# x46) (I8# x47) (I8# x48) (I8# x49) (I8# x50) (I8# x51) (I8# x52) (I8# x53) (I8# x54) (I8# x55) (I8# x56) (I8# x57) (I8# x58) (I8# x59) (I8# x60) (I8# x61) (I8# x62) (I8# x63) (I8# x64) -> packInt8X64 (Tuple64 (I8# (f x1)) (I8# (f x2)) (I8# (f x3)) (I8# (f x4)) (I8# (f x5)) (I8# (f x6)) (I8# (f x7)) (I8# (f x8)) (I8# (f x9)) (I8# (f x10)) (I8# (f x11)) (I8# (f x12)) (I8# (f x13)) (I8# (f x14)) (I8# (f x15)) (I8# (f x16)) (I8# (f x17)) (I8# (f x18)) (I8# (f x19)) (I8# (f x20)) (I8# (f x21)) (I8# (f x22)) (I8# (f x23)) (I8# (f x24)) (I8# (f x25)) (I8# (f x26)) (I8# (f x27)) (I8# (f x28)) (I8# (f x29)) (I8# (f x30)) (I8# (f x31)) (I8# (f x32)) (I8# (f x33)) (I8# (f x34)) (I8# (f x35)) (I8# (f x36)) (I8# (f x37)) (I8# (f x38)) (I8# (f x39)) (I8# (f x40)) (I8# (f x41)) (I8# (f x42)) (I8# (f x43)) (I8# (f x44)) (I8# (f x45)) (I8# (f x46)) (I8# (f x47)) (I8# (f x48)) (I8# (f x49)) (I8# (f x50)) (I8# (f x51)) (I8# (f x52)) (I8# (f x53)) (I8# (f x54)) (I8# (f x55)) (I8# (f x56)) (I8# (f x57)) (I8# (f x58)) (I8# (f x59)) (I8# (f x60)) (I8# (f x61)) (I8# (f x62)) (I8# (f x63)) (I8# (f x64)))

{-# INLINE[1] zipInt8X64 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X64 :: (Int8 -> Int8 -> Int8) -> Int8X64 -> Int8X64 -> Int8X64
zipInt8X64 f = \ v1 v2 -> case unpackInt8X64 v1 of
    Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> case unpackInt8X64 v2 of
        Tuple64 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 -> packInt8X64 (Tuple64 (f x1 y1) (f x2 y2) (f x3 y3) (f x4 y4) (f x5 y5) (f x6 y6) (f x7 y7) (f x8 y8) (f x9 y9) (f x10 y10) (f x11 y11) (f x12 y12) (f x13 y13) (f x14 y14) (f x15 y15) (f x16 y16) (f x17 y17) (f x18 y18) (f x19 y19) (f x20 y20) (f x21 y21) (f x22 y22) (f x23 y23) (f x24 y24) (f x25 y25) (f x26 y26) (f x27 y27) (f x28 y28) (f x29 y29) (f x30 y30) (f x31 y31) (f x32 y32) (f x33 y33) (f x34 y34) (f x35 y35) (f x36 y36) (f x37 y37) (f x38 y38) (f x39 y39) (f x40 y40) (f x41 y41) (f x42 y42) (f x43 y43) (f x44 y44) (f x45 y45) (f x46 y46) (f x47 y47) (f x48 y48) (f x49 y49) (f x50 y50) (f x51 y51) (f x52 y52) (f x53 y53) (f x54 y54) (f x55 y55) (f x56 y56) (f x57 y57) (f x58 y58) (f x59 y59) (f x60 y60) (f x61 y61) (f x62 y62) (f x63 y63) (f x64 y64))

{-# RULES "zipVector +" forall a b . zipInt8X64 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X64 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X64 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X64 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X64 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X64 #-}
-- | Fold the elements of a vector to a single value
foldInt8X64 :: (Int8 -> Int8 -> Int8) -> Int8X64 -> Int8
foldInt8X64 f' = \ v -> case unpackInt8X64 v of
    Tuple64 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8 `f` x9 `f` x10 `f` x11 `f` x12 `f` x13 `f` x14 `f` x15 `f` x16 `f` x17 `f` x18 `f` x19 `f` x20 `f` x21 `f` x22 `f` x23 `f` x24 `f` x25 `f` x26 `f` x27 `f` x28 `f` x29 `f` x30 `f` x31 `f` x32 `f` x33 `f` x34 `f` x35 `f` x36 `f` x37 `f` x38 `f` x39 `f` x40 `f` x41 `f` x42 `f` x43 `f` x44 `f` x45 `f` x46 `f` x47 `f` x48 `f` x49 `f` x50 `f` x51 `f` x52 `f` x53 `f` x54 `f` x55 `f` x56 `f` x57 `f` x58 `f` x59 `f` x60 `f` x61 `f` x62 `f` x63 `f` x64
    where f !x !y = f' x y

{-# RULES "foldVector (+)" foldInt8X64 (+) = sumVector #-}

{-# INLINE sumInt8X64 #-}
-- | Sum up the elements of a vector to a single value.
sumInt8X64 :: Int8X64 -> Int8
sumInt8X64 (Int8X64 x1 x2) = case unpackInt8X32# (plusInt8X32# x1 x2) of
    (# y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32 #) -> I8# y1 + I8# y2 + I8# y3 + I8# y4 + I8# y5 + I8# y6 + I8# y7 + I8# y8 + I8# y9 + I8# y10 + I8# y11 + I8# y12 + I8# y13 + I8# y14 + I8# y15 + I8# y16 + I8# y17 + I8# y18 + I8# y19 + I8# y20 + I8# y21 + I8# y22 + I8# y23 + I8# y24 + I8# y25 + I8# y26 + I8# y27 + I8# y28 + I8# y29 + I8# y30 + I8# y31 + I8# y32

{-# INLINE plusInt8X64 #-}
-- | Add two vectors element-wise.
plusInt8X64 :: Int8X64 -> Int8X64 -> Int8X64
plusInt8X64 (Int8X64 m1_1 m2_1) (Int8X64 m1_2 m2_2) = Int8X64 (plusInt8X32# m1_1 m1_2) (plusInt8X32# m2_1 m2_2)

{-# INLINE minusInt8X64 #-}
-- | Subtract two vectors element-wise.
minusInt8X64 :: Int8X64 -> Int8X64 -> Int8X64
minusInt8X64 (Int8X64 m1_1 m2_1) (Int8X64 m1_2 m2_2) = Int8X64 (minusInt8X32# m1_1 m1_2) (minusInt8X32# m2_1 m2_2)

{-# INLINE timesInt8X64 #-}
-- | Multiply two vectors element-wise.
timesInt8X64 :: Int8X64 -> Int8X64 -> Int8X64
timesInt8X64 (Int8X64 m1_1 m2_1) (Int8X64 m1_2 m2_2) = Int8X64 (timesInt8X32# m1_1 m1_2) (timesInt8X32# m2_1 m2_2)

{-# INLINE quotInt8X64 #-}
-- | Rounds towards zero element-wise.
quotInt8X64 :: Int8X64 -> Int8X64 -> Int8X64
quotInt8X64 (Int8X64 m1_1 m2_1) (Int8X64 m1_2 m2_2) = Int8X64 (quotInt8X32# m1_1 m1_2) (quotInt8X32# m2_1 m2_2)

{-# INLINE remInt8X64 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X64 :: Int8X64 -> Int8X64 -> Int8X64
remInt8X64 (Int8X64 m1_1 m2_1) (Int8X64 m1_2 m2_2) = Int8X64 (remInt8X32# m1_1 m1_2) (remInt8X32# m2_1 m2_2)

{-# INLINE negateInt8X64 #-}
-- | Negate element-wise.
negateInt8X64 :: Int8X64 -> Int8X64
negateInt8X64 (Int8X64 m1_1 m2_1) = Int8X64 (negateInt8X32# m1_1) (negateInt8X32# m2_1)

{-# INLINE indexInt8X64Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X64Array :: ByteArray -> Int -> Int8X64
indexInt8X64Array (ByteArray a) (I# i) = Int8X64 (indexInt8X32Array# a ((i *# 2#) +# 0#)) (indexInt8X32Array# a ((i *# 2#) +# 1#))

{-# INLINE readInt8X64Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X64Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X64
readInt8X64Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8X32Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt8X32Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Int8X64 m1 m2 #))

{-# INLINE writeInt8X64Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X64Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X64 -> m ()
writeInt8X64Array (MutableByteArray a) (I# i) (Int8X64 m1 m2) = primitive_ (writeInt8X32Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeInt8X32Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexInt8X64OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X64OffAddr :: Addr -> Int -> Int8X64
indexInt8X64OffAddr (Addr a) (I# i) = Int8X64 (indexInt8X32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0#) (indexInt8X32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0#)

{-# INLINE readInt8X64OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X64OffAddr :: PrimMonad m => Addr -> Int -> m Int8X64
readInt8X64OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8X32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt8X32OffAddr# (plusAddr# addr i') 0#) a ((i *# 64#) +# 32#) s1 of
        (# s2, m2 #) -> (# s2, Int8X64 m1 m2 #))

{-# INLINE writeInt8X64OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X64OffAddr :: PrimMonad m => Addr -> Int -> Int8X64 -> m ()
writeInt8X64OffAddr (Addr a) (I# i) (Int8X64 m1 m2) = primitive_ (writeInt8X32OffAddr# (plusAddr# a ((i *# 64#) +# 0#)) 0# m1) >> primitive_ (writeInt8X32OffAddr# (plusAddr# a ((i *# 64#) +# 32#)) 0# m2)


