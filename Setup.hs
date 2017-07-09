module Main where

import Control.Monad

import Data.Bits
import Data.Bifunctor
import Data.Char
import Data.List
import Data.Maybe

import Distribution.Simple
import Distribution.System
import Distribution.Simple.Program.Builtin (ghcProgram)
import Distribution.Simple.Program.Types (programFindLocation, ProgramSearchPathEntry(ProgramSearchPathDefault))
import Distribution.Verbosity (silent)

import System.Environment
import System.Exit

import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import System.IO.Temp (withSystemTempDirectory)
import System.Process (readProcessWithExitCode)

import System.Cpuid.Basic (supportsSSE2, supportsAVX2, supportsAVX512f)

import Generator (genCode)

data SSESupport = SSESupport {
     supportSSE2    :: Bool -- ^ Support for 128-bit vectors exists
    ,supportAVX2    :: Bool -- ^ Support for 256-bit vectors exists
    ,supportAVX512f :: Bool -- ^ Support for 512-bit vectors exists
} deriving Show

-- | Determine the processor and OS support for SSE2, AVX2 and AVX512f.
checkSSESupport :: IO SSESupport
checkSSESupport = case buildArch of
    I386 -> SSESupport <$>
        supportsSSE2 <*>
        supportsAVX2 <*>
        supportsAVX512f
    X86_64 -> SSESupport <$>
        supportsSSE2 <*>
        supportsAVX2 <*>
        supportsAVX512f
    _      -> do
        hPutStrLn stderr "WARNING: Unsupported architecture, defaulting to pure Haskell implementation"
        pure $ SSESupport False False False

-- | Check if LLVM works by compiling a simple hello-world program.
checkLLVMSupport :: IO Bool
checkLLVMSupport = case buildCompilerFlavor of
    GHC -> do
        mLoc <- fmap fst <$> programFindLocation ghcProgram silent [ProgramSearchPathDefault]
        case mLoc of
            Nothing -> pure False
            Just loc -> withSystemTempDirectory "llvm-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "LLVMTest.hs"
                    exeFile = tmpDir </> "LLVMTest.exe"
                writeFile hsFile "main = putStrLn \"Hello, World\""
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-fllvm", "-o", exeFile] ""
                case exitCode of
                    ExitSuccess -> do
                        exitVals <- readProcessWithExitCode exeFile [] ""
                        pure $ exitVals == (ExitSuccess, "Hello, World", "")
                    _ -> do
                        hPutStrLn stderr $ "WARNING: Failed to compile code with LLVM, the result was " ++ show exitCode
                        hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS
                        hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS
                        hPutStrLn stderr $ "=============================\nDisabled LLVM code generation"
                        pure False
    _ -> pure False

-- | Example code for our use of pattern synonyms. We use it to make sure we can
--   use them (we can't on GHC 8.0.1).
patSynTestCode :: String
patSynTestCode = unlines
    ["{-# LANGUAGE PatternSynonyms #-}"
    ,"{-# LANGUAGE ViewPatterns    #-}"
    ,"module PatSynTest where"
    ,""
    ,"data X a = X a a a a"
    ,""
    ,"unpackX :: X a -> (a, a, a, a)"
    ,"unpackX (X a b c d) = (a, b, c, d)"
    ,""
    ,"pattern VecX :: a -> a -> a -> a -> X a"
    ,"pattern VecX x1 x2 x3 x4 <- (unpackX -> (x1, x2, x3, x4)) where"
    ,"    VecX x1 x2 x3 x4 = X x1 x2 x3 x4"
    ,""
    ,"pattern VecXF :: Float -> Float -> Float -> Float -> X Float"
    ,"pattern VecXF x1 x2 x3 x4 = VecX x1 x2 x3 x4"
    ,""
    ,"pattern Y4F :: Float -> Float -> Float -> Float -> X Float"
    ,"pattern Y4F x1 x2 x3 x4 = Y x1 x2 x3 x4"
    ,""
    ,"pattern Y :: a -> a -> a -> a -> X a"
    ,"pattern Y a b c d = X a b c d"
    ]

-- | Check if we can compile pattern synonyms. Our detection scheme is not really
--   advanced,
getPatSynSupport :: IO Bool
getPatSynSupport = case buildCompilerFlavor of
    GHC -> do
        mLoc <- fmap fst <$> programFindLocation ghcProgram silent [ProgramSearchPathDefault]
        case mLoc of
            Nothing -> pure False
            Just loc -> withSystemTempDirectory "patsyn-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "PatSyns.hs"
                    exeFile = tmpDir </> "PatSyns.o"
                writeFile hsFile patSynTestCode
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-o", exeFile, "-c"] ""
                case exitCode of
                    ExitSuccess -> pure True
                    _ -> do
                        hPutStrLn stderr $ "WARNING: Failed to compile code with Pattern Synonyms, the result was " ++ show exitCode
                        hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS
                        hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS
                        hPutStrLn stderr $ "=============================\nDisabled pattern synonym code generation"
                        pure False
    _ -> do
        hPutStrLn stderr "WARNING: Unsupported compiler, compilation may fail..."
        pure False

genSrc :: Int -> IO ()
genSrc n = do
    usePatSyns <- getPatSynSupport
    unless usePatSyns $
        hPutStrLn stderr $ "WARNING: The compiler does not seem to support pattern synonyms "
            ++ "(GHC 8.0.1 does not correctly and crashes!), the synonyms Vec<2,4,8,16,32,64> will "
            ++ "be missing. If you encounter undefined references of that name, you need to use a "
            ++ "compiler supporting pattern synonyms."
    genCode "dist/build/autogen/Data/Primitive/SIMD" usePatSyns n

genSrcNoVec :: IO ()
genSrcNoVec = genSrc 0

genSrc128 :: IO ()
genSrc128 = genSrc (128 `quot` 8)

genSrc256 :: IO ()
genSrc256 = genSrc (256 `quot` 8)

genSrc512 :: IO ()
genSrc512 = genSrc (512 `quot` 8)

data Flag = NoVec | Vec128 | Vec256 | Vec512
    deriving (Enum, Bounded, Eq)

showFlag :: Flag -> String
showFlag NoVec = "no-vec"
showFlag Vec128 = "vec128"
showFlag Vec256 = "vec256"
showFlag Vec512 = "vec512"

parseFlags :: [String] -> [(Flag, Bool)]
parseFlags = mapMaybe parseFlag

parseFlag :: String -> Maybe (Flag, Bool)
parseFlag s = do
    s' <- stripPrefix "--flags=" s
    case s' of
        '-' : flagS -> do
            flag <- translateFlag flagS
            pure (flag, False)
        _ -> do
            flag <- translateFlag s'
            pure (flag, True)

translateFlag :: String -> Maybe Flag
translateFlag s = case [f | f <- [minBound .. maxBound], showFlag f == s] of
    [x] -> Just x
    _   -> Nothing

setFlag :: Flag -> [String] -> [String]
setFlag flag cArgs = filter (isNothing . parseFlag) cArgs
    ++ ["--flag=" ++ ['-' | f /= flag] ++ showFlag f | f <- [minBound .. maxBound]]

generateSource :: [String] -> IO [String]
generateSource cArgs = do
    -- to configure this package, you can also set an environment variable
    -- so you do not need to alter the command line options
    -- this way you can easily force 128-bit SIMD usage even if your computer
    -- supports 256 or 512-bit SIMD instructions, e.g. when building a package
    -- or executable for distribution.
    envOverride <- lookup "PRIMITIVE_SIMD_FLAG" <$> getEnvironment
    case envOverride of
        Just "no-vec" -> do
            genSrcNoVec
            pure $ setFlag NoVec cArgs
        Just "vec128" -> do
            genSrc128
            pure $ setFlag Vec128 cArgs
        Just "vec256" -> do
            genSrc256
            pure $ setFlag Vec256 cArgs
        Just "vec512" -> do
            genSrc512
            pure $ setFlag Vec512 cArgs
        Just vecFlag  -> fail $ "Invalid vector flag: " ++ show vecFlag
        Nothing -> do
            sse  <- checkSSESupport
            llvm <- checkLLVMSupport
            let flags = parseFlags cArgs
            if any snd flags
            then do
                let require flag b = unless b $
                        putStrLn $ "Configured with setting "
                            ++ showFlag flag
                            ++ ", but could not determine LLVM/processor support. "
                            ++ "We will try building, but this may fail."
                case filter snd flags of
                    [] -> fail "impossible..."
                    [(flag, _)] -> case flag of
                        NoVec -> genSrcNoVec
                        Vec128 -> do
                            genSrc128
                            require flag $ llvm && supportSSE2 sse
                        Vec256 -> do
                            genSrc256
                            require flag $ llvm && supportSSE2 sse && supportAVX2 sse
                        Vec512 -> do
                            genSrc512
                            require flag $ llvm && supportSSE2 sse && supportAVX2 sse && supportAVX512f sse
                    xs -> fail $ "More than one flag set! " ++ show (map (first showFlag) xs)
                pure cArgs
            else do
                flag <- case (llvm, supportAVX512f sse, supportAVX2 sse, supportSSE2 sse) of
                    (True, True, True, True) -> do
                        genSrc512
                        pure Vec512
                    (True, False, True, True) -> do
                        genSrc256
                        pure Vec256
                    (True, False, False, True) -> do
                        genSrc128
                        pure Vec128
                    _ -> do
                        genSrcNoVec
                        pure NoVec
                pure $ setFlag flag cArgs

main :: IO ()
main = do
    args <- getArgs
    case args of
        ("configure":cArgs) -> do
            newArgs <- generateSource cArgs
            defaultMainArgs $ "configure" : newArgs
        _                   -> defaultMainArgs args
