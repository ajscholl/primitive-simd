{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import Control.DeepSeq

import Criterion.Types
import Criterion.Main

import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Storable as VS

import Data.Primitive.SIMD

import System.Random (mkStdGen, randomRs)

-------------------------------------------------------------------------------
-- config

veclen :: Int
veclen = 16384

critConfig :: Config
critConfig = defaultConfig {
     csvFile    = Just $ "summary" ++ shows veclen ".csv"
    ,reportFile = Just $ "report" ++ shows veclen ".html"
}

-------------------------------------------------------------------------------
-- main

main :: IO ()
main = do

    -----------------------------------
    -- initialize single vectors

    putStrLn "constructing vectors"

    let lf1 = take veclen $ randomRs (-10000, 10000) $ mkStdGen 1 :: [Float]
        lf2 = take veclen $ randomRs (-10000, 10000) $ mkStdGen 2 :: [Float]

        ld1 = take veclen $ randomRs (-10000, 10000) $ mkStdGen 1 :: [Double]
        ld2 = take veclen $ randomRs (-10000, 10000) $ mkStdGen 2 :: [Double]

    let vuf1 = VU.fromList lf1 :: VU.Vector Float
        vuf2 = VU.fromList lf2 :: VU.Vector Float
    deepseq vuf1 $ deepseq vuf2 $ return ()

    let vud1 = VU.fromList ld1 :: VU.Vector Double
        vud2 = VU.fromList ld2 :: VU.Vector Double
    deepseq vud1 $ deepseq vud2 $ return ()

    let vsf1 = VS.fromList lf1 :: VS.Vector Float
        vsf2 = VS.fromList lf2 :: VS.Vector Float
    deepseq vsf1 $ deepseq vsf2 $ return ()

    let vsd1 = VS.fromList ld1 :: VS.Vector Double
        vsd2 = VS.fromList ld2 :: VS.Vector Double
    deepseq vsd1 $ deepseq vsd2 $ return ()



    let vsf1_4 = VS.unsafeCast vsf1 :: VS.Vector FloatX4
        vsf2_4 = VS.unsafeCast vsf2 :: VS.Vector FloatX4
    deepseq vsf1_4 $ deepseq vsf2_4 $ return ()

    let vsf1_8 = VS.unsafeCast vsf1 :: VS.Vector FloatX8
        vsf2_8 = VS.unsafeCast vsf2 :: VS.Vector FloatX8
    deepseq vsf1_8 $ deepseq vsf2_8 $ return ()

    let vsf1_16 = VS.unsafeCast vsf1 :: VS.Vector FloatX16
        vsf2_16 = VS.unsafeCast vsf2 :: VS.Vector FloatX16
    deepseq vsf1_16 $ deepseq vsf2_16 $ return ()

    let vsd1_2 = VS.unsafeCast vsd1 :: VS.Vector DoubleX2
        vsd2_2 = VS.unsafeCast vsd2 :: VS.Vector DoubleX2
    deepseq vsd1_2 $ deepseq vsd2_2 $ return ()

    let vsd1_4 = VS.unsafeCast vsd1 :: VS.Vector DoubleX4
        vsd2_4 = VS.unsafeCast vsd2 :: VS.Vector DoubleX4
    deepseq vsd1_4 $ deepseq vsd2_4 $ return ()

    let vsd1_8 = VS.unsafeCast vsd1 :: VS.Vector DoubleX8
        vsd2_8 = VS.unsafeCast vsd2 :: VS.Vector DoubleX8
    deepseq vsd1_8 $ deepseq vsd2_8 $ return ()



    let vuf1_4 = VU.convert vsf1_4 :: VU.Vector FloatX4
        vuf2_4 = VU.convert vsf2_4 :: VU.Vector FloatX4
    deepseq vuf1_4 $ deepseq vuf2_4 $ return ()

    let vuf1_8 = VU.convert vsf1_8 :: VU.Vector FloatX8
        vuf2_8 = VU.convert vsf2_8 :: VU.Vector FloatX8
    deepseq vuf1_8 $ deepseq vuf2_8 $ return ()

    let vuf1_16 = VU.convert vsf1_16 :: VU.Vector FloatX16
        vuf2_16 = VU.convert vsf2_16 :: VU.Vector FloatX16
    deepseq vuf1_16 $ deepseq vuf2_16 $ return ()

    let vud1_2 = VU.convert vsd1_2 :: VU.Vector DoubleX2
        vud2_2 = VU.convert vsd2_2 :: VU.Vector DoubleX2
    deepseq vud1_2 $ deepseq vud2_2 $ return ()

    let vud1_4 = VU.convert vsd1_4 :: VU.Vector DoubleX4
        vud2_4 = VU.convert vsd2_4 :: VU.Vector DoubleX4
    deepseq vud1_4 $ deepseq vud2_4 $ return ()

    let vud1_8 = VU.convert vsd1_8 :: VU.Vector DoubleX8
        vud2_8 = VU.convert vsd2_8 :: VU.Vector DoubleX8
    deepseq vud1_8 $ deepseq vud2_8 $ return ()

    -----------------------------------
    -- tests

    putStrLn "starting criterion"

    defaultMainWith critConfig
        [ bgroup "VU"
            [ bgroup "Float"
                [ bench "diff1"                 $ nf (distanceDiff1                vuf1) vuf2
                , bench "diff2"                 $ nf (distanceDiff2                vuf1) vuf2
                , bench "diff4"                 $ nf (distanceDiff4                vuf1) vuf2
                , bench "diff8"                 $ nf (distanceDiff8                vuf1) vuf2
                , bench "simd4-1"               $ nf (distanceSimd1                vuf1_4) vuf2_4
                , bench "simd8-1"               $ nf (distanceSimd1                vuf1_8) vuf2_8
                , bench "simd16-1"              $ nf (distanceSimd1                vuf1_16) vuf2_16
                , bench "simd4-2"               $ nf (distanceSimd2                vuf1_4) vuf2_4
                , bench "simd8-2"               $ nf (distanceSimd2                vuf1_8) vuf2_8
                , bench "simd16-2"              $ nf (distanceSimd2                vuf1_16) vuf2_16
                , bench "simd4-4"               $ nf (distanceSimd4                vuf1_4) vuf2_4
                , bench "simd8-4"               $ nf (distanceSimd4                vuf1_8) vuf2_8
                , bench "simd16-4"              $ nf (distanceSimd4                vuf1_16) vuf2_16
                , bench "simd4-8"               $ nf (distanceSimd8                vuf1_4) vuf2_4
                , bench "simd8-8"               $ nf (distanceSimd8                vuf1_8) vuf2_8
                , bench "simd16-8"              $ nf (distanceSimd8                vuf1_16) vuf2_16
                ]
            , bgroup "Double"
                [ bench "diff1"                 $ nf (distanceDiff1                vud1) vud2
                , bench "diff2"                 $ nf (distanceDiff2                vud1) vud2
                , bench "diff4"                 $ nf (distanceDiff4                vud1) vud2
                , bench "diff8"                 $ nf (distanceDiff8                vud1) vud2
                , bench "simd2-1"               $ nf (distanceSimd1                vud1_2) vud2_2
                , bench "simd4-1"               $ nf (distanceSimd1                vud1_4) vud2_4
                , bench "simd8-1"               $ nf (distanceSimd1                vud1_8) vud2_8
                , bench "simd2-2"               $ nf (distanceSimd2                vud1_2) vud2_2
                , bench "simd4-2"               $ nf (distanceSimd2                vud1_4) vud2_4
                , bench "simd8-2"               $ nf (distanceSimd2                vud1_8) vud2_8
                , bench "simd2-4"               $ nf (distanceSimd4                vud1_2) vud2_2
                , bench "simd4-4"               $ nf (distanceSimd4                vud1_4) vud2_4
                , bench "simd8-4"               $ nf (distanceSimd4                vud1_8) vud2_8
                , bench "simd2-8"                 $ nf (distanceSimd8                vud1_2) vud2_2
                , bench "simd4-8"                 $ nf (distanceSimd8                vud1_4) vud2_4
                , bench "simd8-8"                 $ nf (distanceSimd8                vud1_8) vud2_8
                ]
            ]
        , bgroup "VS"
            [ bgroup "Float"
                [ bench "diff1"                 $ nf (distanceDiff1                vsf1) vsf2
                , bench "diff2"                 $ nf (distanceDiff2                vsf1) vsf2
                , bench "diff4"                 $ nf (distanceDiff4                vsf1) vsf2
                , bench "diff8"                 $ nf (distanceDiff8                vsf1) vsf2
                , bench "simd4-1"               $ nf (distanceSimd1                vsf1_4) vsf2_4
                , bench "simd8-1"               $ nf (distanceSimd1                vsf1_8) vsf2_8
                , bench "simd16-1"              $ nf (distanceSimd1                vsf1_16) vsf2_16
                , bench "simd4-2"               $ nf (distanceSimd2                vsf1_4) vsf2_4
                , bench "simd8-2"               $ nf (distanceSimd2                vsf1_8) vsf2_8
                , bench "simd16-2"              $ nf (distanceSimd2                vsf1_16) vsf2_16
                , bench "simd4-4"               $ nf (distanceSimd4                vsf1_4) vsf2_4
                , bench "simd8-4"               $ nf (distanceSimd4                vsf1_8) vsf2_8
                , bench "simd16-4"              $ nf (distanceSimd4                vsf1_16) vsf2_16
                , bench "simd4-8"               $ nf (distanceSimd8                vsf1_4) vsf2_4
                , bench "simd8-8"               $ nf (distanceSimd8                vsf1_8) vsf2_8
                , bench "simd16-8"              $ nf (distanceSimd8                vsf1_16) vsf2_16
                ]
            , bgroup "Double"
                [ bench "diff1"                 $ nf (distanceDiff1                vsd1) vsd2
                , bench "diff2"                 $ nf (distanceDiff2                vsd1) vsd2
                , bench "diff4"                 $ nf (distanceDiff4                vsd1) vsd2
                , bench "diff8"                 $ nf (distanceDiff8                vsd1) vsd2
                , bench "simd2-1"               $ nf (distanceSimd1                vsd1_2) vsd2_2
                , bench "simd4-1"               $ nf (distanceSimd1                vsd1_4) vsd2_4
                , bench "simd8-1"               $ nf (distanceSimd1                vsd1_8) vsd2_8
                , bench "simd2-2"               $ nf (distanceSimd2                vsd1_2) vsd2_2
                , bench "simd4-2"               $ nf (distanceSimd2                vsd1_4) vsd2_4
                , bench "simd8-2"               $ nf (distanceSimd2                vsd1_8) vsd2_8
                , bench "simd2-4"               $ nf (distanceSimd4                vsd1_2) vsd2_2
                , bench "simd4-4"               $ nf (distanceSimd4                vsd1_4) vsd2_4
                , bench "simd8-4"               $ nf (distanceSimd4                vsd1_8) vsd2_8
                , bench "simd2-8"               $ nf (distanceSimd8                vsd1_2) vsd2_2
                , bench "simd4-8"               $ nf (distanceSimd8                vsd1_4) vsd2_4
                , bench "simd8-8"               $ nf (distanceSimd8                vsd1_8) vsd2_8
                ]
            ]
        ]

    putStrLn "reference values - float"

    mapM_ print [distanceDiff1 vuf1    vuf2
                ,distanceDiff2 vuf1    vuf2
                ,distanceDiff4 vuf1    vuf2
                ,distanceDiff8 vuf1    vuf2
                ,distanceSimd1 vuf1_4  vuf2_4
                ,distanceSimd1 vuf1_8  vuf2_8
                ,distanceSimd1 vuf1_16 vuf2_16
                ,distanceSimd2 vuf1_4  vuf2_4
                ,distanceSimd2 vuf1_8  vuf2_8
                ,distanceSimd2 vuf1_16 vuf2_16
                ,distanceSimd4 vuf1_4  vuf2_4
                ,distanceSimd4 vuf1_8  vuf2_8
                ,distanceSimd4 vuf1_16 vuf2_16
                ,distanceSimd8 vuf1_4  vuf2_4
                ,distanceSimd8 vuf1_8  vuf2_8
                ,distanceSimd8 vuf1_16 vuf2_16
                ,distanceDiff1 vsf1    vsf2
                ,distanceDiff2 vsf1    vsf2
                ,distanceDiff4 vsf1    vsf2
                ,distanceDiff8 vsf1    vsf2
                ,distanceSimd1 vsf1_4  vsf2_4
                ,distanceSimd1 vsf1_8  vsf2_8
                ,distanceSimd1 vsf1_16 vsf2_16
                ,distanceSimd2 vsf1_4  vsf2_4
                ,distanceSimd2 vsf1_8  vsf2_8
                ,distanceSimd2 vsf1_16 vsf2_16
                ,distanceSimd4 vsf1_4  vsf2_4
                ,distanceSimd4 vsf1_8  vsf2_8
                ,distanceSimd4 vsf1_16 vsf2_16
                ,distanceSimd8 vsf1_4  vsf2_4
                ,distanceSimd8 vsf1_8  vsf2_8
                ,distanceSimd8 vsf1_16 vsf2_16
                ]

    putStrLn "reference values - double"

    mapM_ print [distanceDiff1 vud1    vud2
                ,distanceDiff2 vud1    vud2
                ,distanceDiff4 vud1    vud2
                ,distanceDiff8 vud1    vud2
                ,distanceSimd1 vud1_2  vud2_2
                ,distanceSimd1 vud1_4  vud2_4
                ,distanceSimd1 vud1_8  vud2_8
                ,distanceSimd2 vud1_2  vud2_2
                ,distanceSimd2 vud1_4  vud2_4
                ,distanceSimd2 vud1_8  vud2_8
                ,distanceSimd4 vud1_2  vud2_2
                ,distanceSimd4 vud1_4  vud2_4
                ,distanceSimd4 vud1_8  vud2_8
                ,distanceSimd8 vud1_2  vud2_2
                ,distanceSimd8 vud1_4  vud2_4
                ,distanceSimd8 vud1_8  vud2_8
                ,distanceDiff1 vsd1    vsd2
                ,distanceDiff2 vsd1    vsd2
                ,distanceDiff4 vsd1    vsd2
                ,distanceDiff8 vsd1    vsd2
                ,distanceSimd1 vsd1_2  vsd2_2
                ,distanceSimd1 vsd1_4  vsd2_4
                ,distanceSimd1 vsd1_8  vsd2_8
                ,distanceSimd2 vsd1_2  vsd2_2
                ,distanceSimd2 vsd1_4  vsd2_4
                ,distanceSimd2 vsd1_8  vsd2_8
                ,distanceSimd4 vsd1_2  vsd2_2
                ,distanceSimd4 vsd1_4  vsd2_4
                ,distanceSimd4 vsd1_8  vsd2_8
                ,distanceSimd8 vsd1_2  vsd2_2
                ,distanceSimd8 vsd1_4  vsd2_4
                ,distanceSimd8 vsd1_8  vsd2_8
                ]

-------------------------------------------------------------------------------
-- distance functions

distanceDiff1 :: (VG.Vector v f, Floating f) => v f -> v f -> f
distanceDiff1 = distanceDiff1Helper sqrt

distanceDiff2 :: (VG.Vector v f, Floating f) => v f -> v f -> f
distanceDiff2 = distanceDiff2Helper sqrt

distanceDiff4 :: (VG.Vector v f, Floating f) => v f -> v f -> f
distanceDiff4 = distanceDiff4Helper sqrt

distanceDiff8 :: (VG.Vector v f, Floating f) => v f -> v f -> f
distanceDiff8 = distanceDiff8Helper sqrt

distanceSimd1 :: (VG.Vector v f, SIMDVector f, Num f, Floating (Elem f)) => v f -> v f -> Elem f
distanceSimd1 = distanceDiff1Helper (sqrt . sumVector)

distanceSimd2 :: (VG.Vector v f, SIMDVector f, Num f, Floating (Elem f)) => v f -> v f -> Elem f
distanceSimd2 = distanceDiff2Helper (sqrt . sumVector)

distanceSimd4 :: (VG.Vector v f, SIMDVector f, Num f, Floating (Elem f)) => v f -> v f -> Elem f
distanceSimd4 = distanceDiff4Helper (sqrt . sumVector)

distanceSimd8 :: (VG.Vector v f, SIMDVector f, Num f, Floating (Elem f)) => v f -> v f -> Elem f
distanceSimd8 = distanceDiff8Helper (sqrt . sumVector)

-- helpers

sqr :: Num a => a -> a
sqr a = a * a

{-# INLINE zipVecsWith #-}
zipVecsWith :: VG.Vector v a => (a -> a -> b -> b) -> b -> v a -> v a -> b
zipVecsWith f z v1 v2 = VG.ifoldl' (\ acc i a -> f a (VG.unsafeIndex v2 i) acc) z v1

distanceDiff1Helper :: (VG.Vector v a, Num a) => (a -> r) -> v a -> v a -> r
distanceDiff1Helper f v1 v2 = f $ zipVecsWith (\ a b acc -> acc + sqr (a - b)) 0 v1 v2

distanceDiff2Helper :: (VG.Vector v a, Num a) => (a -> r) -> v a -> v a -> r
distanceDiff2Helper f v1 v2 = f $ go 0 (VG.length v1-1)
    where
        go acc (-1) = acc
        go acc i = go acc' (i-2)
            where
                acc' = acc+diff1*diff1
                          +diff2*diff2
                diff1 = v1 `VG.unsafeIndex` i - v2 `VG.unsafeIndex` i
                diff2 = v1 `VG.unsafeIndex` (i-1) - v2 `VG.unsafeIndex` (i-1)

distanceDiff4Helper :: (VG.Vector v a, Num a) => (a -> r) -> v a -> v a -> r
distanceDiff4Helper f v1 v2 = f $ go 0 (VG.length v1-1)
    where
        go acc (-1) = acc
        go acc i = go acc' (i-4)
            where
                acc' = acc+diff1*diff1
                          +diff2*diff2
                          +diff3*diff3
                          +diff4*diff4
                diff1 = diff 0
                diff2 = diff 1
                diff j = v1 `VG.unsafeIndex` (i-j) - v2 `VG.unsafeIndex` (i-j)
                diff3 = diff 2
                diff4 = diff 3

distanceDiff8Helper :: (VG.Vector v a, Num a) => (a -> r) -> v a -> v a -> r
distanceDiff8Helper f v1 v2 = f $ go 0 (VG.length v1-1)
    where
        go acc (-1) = acc
        go acc i = go acc' (i-8)
            where
                acc' = acc+diff1*diff1
                          +diff2*diff2
                          +diff3*diff3
                          +diff4*diff4
                          +diff5*diff5
                          +diff6*diff6
                          +diff7*diff7
                          +diff8*diff8
                diff1 = diff 0
                diff2 = diff 1
                diff3 = diff 2
                diff4 = diff 3
                diff5 = diff 4
                diff6 = diff 5
                diff7 = diff 6
                diff8 = diff 7
                diff j = v1 `VG.unsafeIndex` (i-j) - v2 `VG.unsafeIndex` (i-j)
