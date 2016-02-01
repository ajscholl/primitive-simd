-----------------------------------------------------------------------------
-- |
-- Module      :  Data.Primitive.SIMD
-- Copyright   :  (c) 2015 Anselm Jonas Scholl
-- License     :  BSD3
-- 
-- Maintainer  :  anselm.scholl@tu-harburg.de
-- Stability   :  experimental
-- Portability :  non-portable (uses GHC.Prim)
--
-- SIMD data types and functions.
--
-----------------------------------------------------------------------------
module Data.Primitive.SIMD (
     -- * SIMD type classes
     SIMDVector(..)
    ,SIMDIntVector(..)
     -- * SIMD data types
    ,Tuple64(..)
    ,Int8X16
    ,Int8X32
    ,Int8X64
    ,Int16X8
    ,Int16X16
    ,Int16X32
    ,Int32X4
    ,Int32X8
    ,Int32X16
    ,Int64X2
    ,Int64X4
    ,Int64X8
    ,Word8X16
    ,Word8X32
    ,Word8X64
    ,Word16X8
    ,Word16X16
    ,Word16X32
    ,Word32X4
    ,Word32X8
    ,Word32X16
    ,Word64X2
    ,Word64X4
    ,Word64X8
    ,FloatX4
    ,FloatX8
    ,FloatX16
    ,DoubleX2
    ,DoubleX4
    ,DoubleX8
    ,DoubleX16
    ) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class
import Data.Primitive.SIMD.Int8X16
import Data.Primitive.SIMD.Int8X32
import Data.Primitive.SIMD.Int8X64
import Data.Primitive.SIMD.Int16X8
import Data.Primitive.SIMD.Int16X16
import Data.Primitive.SIMD.Int16X32
import Data.Primitive.SIMD.Int32X4
import Data.Primitive.SIMD.Int32X8
import Data.Primitive.SIMD.Int32X16
import Data.Primitive.SIMD.Int64X2
import Data.Primitive.SIMD.Int64X4
import Data.Primitive.SIMD.Int64X8
import Data.Primitive.SIMD.Word8X16
import Data.Primitive.SIMD.Word8X32
import Data.Primitive.SIMD.Word8X64
import Data.Primitive.SIMD.Word16X8
import Data.Primitive.SIMD.Word16X16
import Data.Primitive.SIMD.Word16X32
import Data.Primitive.SIMD.Word32X4
import Data.Primitive.SIMD.Word32X8
import Data.Primitive.SIMD.Word32X16
import Data.Primitive.SIMD.Word64X2
import Data.Primitive.SIMD.Word64X4
import Data.Primitive.SIMD.Word64X8
import Data.Primitive.SIMD.FloatX4
import Data.Primitive.SIMD.FloatX8
import Data.Primitive.SIMD.FloatX16
import Data.Primitive.SIMD.DoubleX2
import Data.Primitive.SIMD.DoubleX4
import Data.Primitive.SIMD.DoubleX8
import Data.Primitive.SIMD.DoubleX16
