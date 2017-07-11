{-# OPTIONS_GHC -Wall #-}
module Main where

import Control.Arrow (first)
import Control.Applicative (pure, (<$>), (<*>))
import Control.Monad (when, unless)

import Prelude

import Data.Char (isSpace)
import Data.List (stripPrefix)
import Data.Maybe (mapMaybe, isNothing)

import Distribution.ModuleName (ModuleName, fromString)
import Distribution.PackageDescription (BuildInfo(..), Library(..), PackageDescription(..))
import Distribution.Simple (UserHooks(..), simpleUserHooks, defaultMainWithHooksArgs, CompilerFlavor(..), buildCompilerFlavor)
import Distribution.Simple.BuildPaths (autogenModulesDir, exeExtension, objExtension)
import Distribution.Simple.Program.Builtin (ghcProgram)
import Distribution.Simple.Program.Types (programFindLocation, ProgramSearchPathEntry(ProgramSearchPathDefault))
import Distribution.System (Arch(..), buildArch)
import Distribution.Verbosity (silent)

import System.Environment (getArgs, getEnvironment)
import System.Exit (ExitCode(..))

import System.FilePath ((</>), replaceExtension)
import System.IO (hPutStrLn, stderr)
import System.IO.Temp (withSystemTempDirectory)
import System.Process (readProcessWithExitCode)

import System.Cpuid.Basic (supportsSSE2, supportsAVX2, supportsAVX512f)

import Generator (genCode, PatsMode(..))

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
            Nothing  -> do
                when chatty $ hPutStrLn stderr "Could not determine GHC location, disabled usage of LLVM"
                pure False
            Just loc -> withSystemTempDirectory "llvm-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "LLVMTest.hs"
                    exeFile = tmpDir </> replaceExtension "LLVMTest" exeExtension
                writeFile hsFile "main = putStrLn \"Hello, World\""
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-fllvm", "-o", exeFile] ""
                case exitCode of
                    ExitSuccess -> do
                        exitVals@(exitCode', stdoutS', stderrS') <- readProcessWithExitCode exeFile [] ""
                        if exitVals == (ExitSuccess, "Hello, World\n", "")
                        then pure True
                        else do
                            when chatty $ do
                                hPutStrLn stderr $ "WARNING: Code compiled with LLVM did not return expected output, the result was " ++ show exitCode'
                                hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS'
                                hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS'
                                hPutStrLn stderr $ "=============================\nDisabled LLVM code generation"
                            pure False
                    _ -> do
                        when chatty $ do
                            hPutStrLn stderr $ "WARNING: Failed to compile code with LLVM, the result was " ++ show exitCode
                            hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS
                            hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS
                            hPutStrLn stderr $ "=============================\nDisabled LLVM code generation"
                        pure False
    _ -> do
        when chatty $ hPutStrLn stderr "Usage of LLVM is currently only supported for GHC"
        pure False

-- | Example code for our use of pattern synonyms. We use it to make sure we can
--   use them (we can't on GHC 8.0.1).
patSynTestCode :: Bool -> String
patSynTestCode patSigs = unlines
    ["{-# LANGUAGE PatternSynonyms #-}"
    ,"{-# LANGUAGE ViewPatterns    #-}"
    ,"{-# LANGUAGE TypeFamilies    #-}"
    ,"module PatSynTest where"
    ,""
    ,"data X a = X a a"
    ,""
    ,"class Vector v where"
    ,"    type ElemType v"
    ,"    type ElemTuple v"
    ,"    packVector :: ElemTuple v -> v"
    ,"    unpackVector :: v -> ElemTuple v"
    ,""
    ,"instance Vector (X a) where"
    ,"    type ElemType (X a) = a"
    ,"    type ElemTuple (X a) = (a, a)"
    ,"    packVector (a, b) = X a b"
    ,"    unpackVector (X a b) = (a, b)"
    ,""
    ,if patSigs then "pattern Vec2 :: (Vector v, ElemTuple v ~ (a, b)) => a -> b -> v" else ""
    ,"pattern Vec2 x1 x2 <- (unpackVector -> (x1, x2)) where"
    ,"    Vec2 x1 x2 = packVector (x1, x2)"
    ]

-- | Check if we can compile pattern synonyms. Our detection scheme is not really
--   advanced,
getPatSynSupport :: Bool -> IO PatsMode
getPatSynSupport chatty = case buildCompilerFlavor of
    GHC -> do
        mLoc <- fmap fst <$> programFindLocation ghcProgram silent [ProgramSearchPathDefault]
        case mLoc of
            Nothing -> pure NoPats
            Just loc -> withSystemTempDirectory "patsyn-test" $ \ tmpDir -> do
                let hsFile = tmpDir </> "PatSyns.hs"
                    exeFile = tmpDir </> replaceExtension "PatSyns" objExtension
                writeFile hsFile (patSynTestCode True)
                (exitCode, stdoutS, stderrS) <- readProcessWithExitCode loc ["-O", hsFile, "-o", exeFile, "-c"] ""
                case exitCode of
                    ExitSuccess -> pure Pats
                    _           -> do
                        -- maybe we can get by without pattern signatures...
                        writeFile hsFile (patSynTestCode False)
                        (exitCode', _, _) <- readProcessWithExitCode loc ["-O", hsFile, "-o", exeFile, "-c"] ""
                        case exitCode' of
                            ExitSuccess -> pure NoPatSigs
                            _           -> do
                                when chatty $ do
                                    hPutStrLn stderr $ "WARNING: Failed to compile code with Pattern Synonyms, the result was " ++ show exitCode
                                    hPutStrLn stderr $ "=============================\nSTDOUT:\n" ++ stdoutS
                                    hPutStrLn stderr $ "=============================\nSTDERR:\n" ++ stderrS
                                    hPutStrLn stderr $ "=============================\nDisabled pattern synonym code generation"
                                pure NoPats
    _ -> do
        when chatty $ hPutStrLn stderr "WARNING: Unsupported compiler, compilation may fail..."
        pure NoPats

-- | Generate sources in for the given vector width in the given directory.
--   Also takes care of figuring out the pattern synonym support.
genSrc :: Int -> FilePath -> IO ()
genSrc n autogenDir = do
    usePatSyns <- getPatSynSupport True
    when (usePatSyns == NoPats) $
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
    ,sDistHook = \ pkgDesc mLocBuildInfo uHooks sDistFlags -> do
        -- we have to filter our the auto generated modules to avoid cabal complaining
        -- about not finding them
        let parseXAutogenModules :: String -> [ModuleName]
            parseXAutogenModules = map (fromString . filter (not. isSpace)) . lines
            filterModules :: BuildInfo -> [ModuleName] -> [ModuleName]
            filterModules bi = case maybe [] parseXAutogenModules $ lookup "x-autogen-modules" $ customFieldsBI bi of
                autogens -> filter (`notElem` autogens)
            fixBuildInfoAutogens :: BuildInfo -> BuildInfo
            fixBuildInfoAutogens bi = bi { otherModules = filterModules bi (otherModules bi) }
            fixLibraryAutogens :: Library -> Library
            fixLibraryAutogens lib  = lib {
                 exposedModules     = filterModules (libBuildInfo lib) (exposedModules lib)
                ,requiredSignatures = filterModules (libBuildInfo lib) (requiredSignatures lib)
                ,exposedSignatures  = filterModules (libBuildInfo lib) (exposedSignatures lib)
                ,libBuildInfo       = fixBuildInfoAutogens (libBuildInfo lib)
            }
            pkgDesc' :: PackageDescription
            pkgDesc' = pkgDesc { library = fixLibraryAutogens <$> library pkgDesc }
        sDistHook simpleUserHooks pkgDesc' mLocBuildInfo uHooks sDistFlags
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
