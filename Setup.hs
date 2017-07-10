{-# OPTIONS_GHC -Wall #-}
module Main where

import Control.Monad (when, unless)

import Data.Bifunctor (first)
import Data.List (stripPrefix)
import Data.Maybe (mapMaybe, isNothing)

import Distribution.Simple (UserHooks(..), simpleUserHooks, defaultMainWithHooksArgs, CompilerFlavor(..), buildCompilerFlavor)
import Distribution.Simple.BuildPaths (autogenModulesDir, exeExtension, objExtension)
import Distribution.System (Arch(..), buildArch)
import Distribution.Simple.Program.Builtin (ghcProgram)
import Distribution.Simple.Program.Types (programFindLocation, ProgramSearchPathEntry(ProgramSearchPathDefault))
import Distribution.Verbosity (silent)

import System.Environment (getArgs, getEnvironment)
import System.Exit (ExitCode(..))

import System.FilePath ((</>), replaceExtension)
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
checkSSESupport :: Bool -> IO SSESupport
checkSSESupport chatty = case buildArch of
    I386 -> SSESupport <$>
        supportsSSE2 <*>
        supportsAVX2 <*>
        supportsAVX512f
    X86_64 -> SSESupport <$>
        supportsSSE2 <*>
        supportsAVX2 <*>
        supportsAVX512f
    _      -> do
        when chatty $ hPutStrLn stderr "WARNING: Unsupported architecture, defaulting to pure Haskell implementation"
        pure $ SSESupport False False False

-- | Check if LLVM works by compiling a simple hello-world program.
checkLLVMSupport :: Bool -> IO Bool
checkLLVMSupport chatty = case buildCompilerFlavor of
    GHC -> do
        mLoc <- fmap fst <$> programFindLocation ghcProgram silent [ProgramSearchPathDefault]
        case mLoc of
            Nothing -> pure False
            Just loc -> withSystemTempDirectory "llvm-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "LLVMTest.hs"
                    exeFile = tmpDir </> replaceExtension "LLVMTest" exeExtension
                writeFile hsFile "main = putStrLn \"Hello, World\""
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-fllvm", "-o", exeFile] ""
                case exitCode of
                    ExitSuccess -> do
                        exitVals <- readProcessWithExitCode exeFile [] ""
                        pure $ exitVals == (ExitSuccess, "Hello, World", "")
                    _ -> do
                        when chatty $ do
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
getPatSynSupport :: Bool -> IO Bool
getPatSynSupport chatty = case buildCompilerFlavor of
    GHC -> do
        mLoc <- fmap fst <$> programFindLocation ghcProgram silent [ProgramSearchPathDefault]
        case mLoc of
            Nothing -> pure False
            Just loc -> withSystemTempDirectory "patsyn-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "PatSyns.hs"
                    exeFile = tmpDir </> replaceExtension "PatSyns" objExtension
                writeFile hsFile patSynTestCode
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-o", exeFile, "-c"] ""
                case exitCode of
                    ExitSuccess -> pure True
                    _ -> do
                        when chatty $ do
                            hPutStrLn stderr $ "WARNING: Failed to compile code with Pattern Synonyms, the result was " ++ show exitCode
                            hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS
                            hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS
                            hPutStrLn stderr $ "=============================\nDisabled pattern synonym code generation"
                        pure False
    _ -> do
        when chatty $ hPutStrLn stderr "WARNING: Unsupported compiler, compilation may fail..."
        pure False

-- | Generate sources in for the given vector width in the given directory.
--   Also takes care of figuring out the pattern synonym support.
genSrc :: Int -> FilePath -> IO ()
genSrc n autogenDir = do
    usePatSyns <- getPatSynSupport True
    unless usePatSyns $
        hPutStrLn stderr $ "WARNING: The compiler does not seem to support pattern synonyms "
            ++ "(GHC 8.0.1 does not correctly and crashes!), the synonyms Vec<2,4,8,16,32,64> will "
            ++ "be missing. If you encounter undefined references of that name, you need to use a "
            ++ "compiler supporting pattern synonyms."
    genCode (autogenDir </> "Data/Primitive/SIMD") usePatSyns n

-- | As 'genSrc', but takes a flag instead of the vector size in bytes.
genSrcForFlag :: Flag -> FilePath -> IO ()
genSrcForFlag NoVec = genSrc 0
genSrcForFlag Vec128 = genSrc (128 `quot` 8)
genSrcForFlag Vec256 = genSrc (256 `quot` 8)
genSrcForFlag Vec512 = genSrc (512 `quot` 8)

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

resolveFlags :: Bool -> [String] -> IO ([String], Flag)
resolveFlags chatty cArgs = do
    -- to configure this package, you can also set an environment variable
    -- so you do not need to alter the command line options
    -- this way you can easily force 128-bit SIMD usage even if your computer
    -- supports 256 or 512-bit SIMD instructions, e.g. when building a package
    -- or executable for distribution.
    envOverride <- lookup "PRIMITIVE_SIMD_FLAG" <$> getEnvironment
    case envOverride of
        Just vecFlag
            | Just flag <- translateFlag vecFlag
            -> pure (setFlag flag cArgs, flag)
            | otherwise
            -> fail $ "Invalid vector flag: " ++ show vecFlag
        Nothing -> do
            sse  <- checkSSESupport chatty
            llvm <- checkLLVMSupport chatty
            let flags = parseFlags cArgs
            if any snd flags
            then do
                let require flag b = do
                        unless (b || not chatty) $
                            hPutStrLn stderr $ "Configured with setting "
                                ++ showFlag flag
                                ++ ", but could not determine LLVM/processor support. "
                                ++ "We will try building, but this may fail."
                        pure flag
                flag <- case filter snd flags of
                    [] -> fail "impossible..."
                    [(flag, _)] -> case flag of
                        NoVec -> pure flag
                        Vec128 -> require flag $ llvm && supportSSE2 sse
                        Vec256 -> require flag $ llvm && supportSSE2 sse && supportAVX2 sse
                        Vec512 -> require flag $ llvm && supportSSE2 sse && supportAVX2 sse && supportAVX512f sse
                    xs -> fail $ "More than one flag set! " ++ show (map (first showFlag) xs)
                pure (cArgs, flag)
            else do
                let flag = case (llvm, supportAVX512f sse, supportAVX2 sse, supportSSE2 sse) of
                        (True, True, True, True)   -> Vec512
                        (True, False, True, True)  -> Vec256
                        (True, False, False, True) -> Vec128
                        _                          -> NoVec
                pure (setFlag flag cArgs, flag)

hooks :: UserHooks
hooks = simpleUserHooks {
     confHook = \ (pkgDesc, hookBuildInfo) confFlags -> do
        -- first get the local build information, so we can figure out the directory
        -- we have to place our sources in
        localBuildInfo <- confHook simpleUserHooks (pkgDesc, hookBuildInfo) confFlags
        -- then run 'resolveFlags' a second time, but this time without printing anything
        -- one could split it into two parts, but then we would have to duplicate a lot of logic
        strArgs <- getArgs
        (_, flag) <- resolveFlags False strArgs
        -- generate sources
        genSrcForFlag flag $ autogenModulesDir localBuildInfo
        pure localBuildInfo
}

main :: IO ()
main = do
    args <- getArgs
    args' <- case args of
        ("configure":cArgs) -> do
            (newArgs, _) <- resolveFlags True cArgs
            pure $ "configure" : newArgs
        _                   -> pure args
    defaultMainWithHooksArgs hooks args'
