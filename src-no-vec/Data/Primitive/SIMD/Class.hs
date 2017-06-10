{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}


module Data.Primitive.SIMD.Class where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Control.Monad.Primitive
import Data.Primitive



-- | The compiler only supports tuples up to 62 elements, so we have to use our own data type.
data Tuple64 a = Tuple64 a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a

-- * SIMD type classes

-- | This class provides basic operations to create and consume SIMD types.
--   Numeric operations on members of this class should compile to single
--   SIMD instructions although not all operations are (yet) supported by
--   GHC (e.g. 'sqrt', it is currently implemented as @mapVector sqrt@ which
--   has to unpack the vector, compute the results and pack them again).
class (Num v, Real (Elem v)) => SIMDVector v where
    -- | Type of the elements in the vector
    type Elem v
    -- | Type used to pack or unpack the vector
    type ElemTuple v
    -- | Vector with all elements initialized to zero.
    nullVector       :: v
    -- | Number of components (scalar elements) in the vector. The argument is not evaluated.
    vectorSize       :: v -> Int
    -- | Size of each (scalar) element in the vector in bytes. The argument is not evaluated.
    elementSize      :: v -> Int
    -- | Broadcast a scalar to all elements of a vector.
    broadcastVector  :: Elem v -> v
    -- | The vector that results from applying the given function to all indices in
    --   the range @0 .. 'vectorSize' - 1@.
    generateVector   :: (Int -> Elem v) -> v
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range an exception is thrown.
    insertVector     :: v -> Elem v -> Int -> v
    insertVector v e i | i < 0            = error $ "insertVector: negative argument: " ++ show i
                       | i < vectorSize v = unsafeInsertVector v e i
                       | otherwise        = error $ "insertVector: argument too large: " ++ show i
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range the behavior is undefined.
    unsafeInsertVector     :: v -> Elem v -> Int -> v
    -- | Apply a function to each element of a vector. Be very careful not to map
    --   branching functions over a vector as they could lead to quite a bit of
    --   code bloat (or make sure they are tagged with NOINLINE).
    mapVector        :: (Elem v -> Elem v) -> v -> v
    -- | Zip two vectors together using a combining function.
    zipVector        :: (Elem v -> Elem v -> Elem v) -> v -> v -> v
    -- | Fold the elements of a vector to a single value. The order in which
    --   the elements are combined is not specified.
    foldVector       :: (Elem v -> Elem v -> Elem v) -> v -> Elem v
    -- | Sum up the components of the vector. Equivalent to @foldVector (+)@.
    sumVector        :: v -> Elem v
    sumVector        = foldVector (+)
    -- | Pack some elements to a vector.
    packVector       :: ElemTuple v -> v
    -- | Unpack a vector.
    unpackVector     :: v -> ElemTuple v

-- | Provides vectorized versions of 'quot' and 'rem'. Implementing their 
--   type class is not possible for SIMD types as it would require
--   implementing 'toInteger'.
class SIMDVector v => SIMDIntVector v where
    -- | Rounds towards zero element-wise.
    quotVector :: v -> v -> v
    -- | Satisfies @(quotVector x y) * y + (remVector x y) == x@.
    remVector  :: v -> v -> v

{-# INLINE setByteArrayGeneric #-}
setByteArrayGeneric :: (Prim a, PrimMonad m) => MutableByteArray (PrimState m) -> Int -> Int -> a -> m ()
setByteArrayGeneric mba off n v | n <= 0 = return ()
                                | otherwise = do
    writeByteArray mba off v
    setByteArrayGeneric mba (off + 1) (n - 1) v

{-# INLINE setOffAddrGeneric #-}
setOffAddrGeneric :: (Prim a, PrimMonad m) => Addr -> Int -> Int -> a -> m ()
setOffAddrGeneric addr off n v | n <= 0 = return ()
                               | otherwise = do
    writeOffAddr addr off v
    setOffAddrGeneric addr (off + 1) (n - 1) v

