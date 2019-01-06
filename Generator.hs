{-# OPTIONS_GHC -Wall #-}
module Generator (genCode, PatsMode(..)) where

import Control.Monad

import Data.Char
import Data.List
import Data.Maybe
import Data.Version (Version(..))

import GHC.Exts (maxTupleSize)

import System.Directory (createDirectoryIfMissing)
import System.Info (compilerVersion)

{-

Note [Pipelining]

We could try to exploit pipelining on modern architectures by rewriting
a + b + c + d to (a + b) + (c + d), allowing (a + b) be computed independently
from (c + d). Currently we do not implement this as it can increase the register
pressure and we have no benchmark supporting increased throughput yet.

-}

-- | Type description
type TypeDesc = (String, Maybe Int, Int)

type TypeRange = (String, Maybe Int, Int, Int)

-- | Description of all primitive types, compressed form
primitiveTypes :: [TypeRange]
primitiveTypes = [("Int",    Just  8, 16, 64)
                 ,("Int",    Just 16,  8, 32)
                 ,("Int",    Just 32,  4, 16)
                 ,("Int",    Just 64,  2,  8)
                 ,("Word",   Just  8, 16, 64)
                 ,("Word",   Just 16,  8, 32)
                 ,("Word",   Just 32,  4, 16)
                 ,("Word",   Just 64,  2,  8)
                 ,("Float",  Nothing,  4, 16)
                 ,("Double", Nothing,  2, 16)
                 ]

-- | Description of all primitive types, expanded form
allPrimitiveTypes :: [TypeDesc]
allPrimitiveTypes = concatMap getPrimitiveTypes primitiveTypes
    where
        getPrimitiveTypes (base, mMemCount, startSize, endSize) = if startSize > endSize
            then []
            else (base, mMemCount, startSize) : getPrimitiveTypes (base, mMemCount, startSize * 2, endSize)

-- | Get the data type name form a type description
getDataName :: TypeDesc -> String
getDataName td@(_, _, size) = getBaseType td ++ "X" ++ show size

-- | Get the primitive type name form a type description
getPrimName :: TypeDesc -> String
getPrimName td@(_, _, 1) = getUnderlyingPrimType $ getBaseType td
getPrimName typeDesc     = getDataName typeDesc ++ "#"

getUnderlyingPrimType :: String -> String
getUnderlyingPrimType "Word8"  = "Word#"
getUnderlyingPrimType "Word16" = "Word#"
getUnderlyingPrimType "Word32" = "Word#"
getUnderlyingPrimType "Word64" = "RealWord64#"
getUnderlyingPrimType "Int8"   = "Int#"
getUnderlyingPrimType "Int16"  = "Int#"
getUnderlyingPrimType "Int32"  = "Int#"
getUnderlyingPrimType "Int64"  = "RealInt64#"
getUnderlyingPrimType t        = t ++ "#"

-- | Get the base name of the elements in the vector (Int8, ..., Word8, ..., Float, Double)
getBaseType :: TypeDesc -> String
getBaseType (base, mMemCount, _) = base ++ maybe "" show mMemCount

-- | Get the number of elements stored in this vector
getVectorSize :: TypeDesc -> Int
getVectorSize (_, _, size) = size

-- | Get the primitive type corresponding to the base type
getPrimBaseName :: TypeDesc -> String
getPrimBaseName typeDesc = filter (not . isLower) (getBaseType typeDesc) ++ "#"

-- | Get the size of a single element in a vector, in bytes
getElementSize :: TypeDesc -> Int
getElementSize (base, mMemCount, _) = fromMaybe floatSize mMemCount `quot` 8
    where
        floatSize = if base == "Double" then 64 else 32

-- | Create a tuple match or construction given a set of expressions or patterns.
matchTuple :: Bool -> [String] -> String
matchTuple p vars | length vars < maxTupleSize = "(" ++ intercalate ", " vars ++ ")"
                  | p                          = "(Tuple64 " ++ unwords (map addParens vars) ++ ")"
                  | otherwise                  = "Tuple64 " ++ unwords (map addParens vars)

-- | Surround a value with parents if required.
addParens :: String -> String
addParens s' = if needsParens s' then "(" ++ s' ++ ")" else s'
    where
        needsParens s = not $ all (\ c -> isAlphaNum c || c == '_') s

-- | Tuple type constructor of the given size.
tupleType :: Int -> String -> String
tupleType size t | size < maxTupleSize = "(" ++ intercalate ", " (replicate size t) ++ ")"
                 | otherwise           = "Tuple64 " ++ t

-- | Check if this is a Float or Double vector
isFloating :: TypeDesc -> Bool
isFloating (_, mMemCount, _) = isNothing mMemCount

-- | Check if this is a Word vector
isWord :: TypeDesc -> Bool
isWord (base, _, _) = base == "Word"

-- | Check if this is an Int vector
isInt :: TypeDesc -> Bool
isInt d = not $ isFloating d || isWord d

-- | Check if this is an 64 Int or Word vector
is64 :: TypeDesc -> Bool
is64 (_, mMemCount, _) = maybe False (64==) mMemCount

splitTypes :: Int -> TypeDesc -> (TypeDesc, Int)
splitTypes 64 (base, Just 8, 64)                    = splitTypes 32 (base, Just 8, 64) -- fix for maxTupleSize < 64, making some types unusable... (see https://ghc.haskell.org/trac/ghc/ticket/10648)
splitTypes maxCapability td@(base, mMemCount, size) = ((base, mMemCount, size `quot` fac), fac)
    where
        elemSize = getElementSize td * getVectorSize td
        fac | elemSize <= maxCapability = 1
            | maxCapability == 0        = size
            | otherwise                 = elemSize `quot` maxCapability

-- | Split a list into i parts of the same length
splitIntoPackets :: [a] -> Int -> [[a]]
splitIntoPackets l i = splitToParts (length l `quot` i) l

-- | Split a lists to chunks
splitToParts :: Int -> [a] -> [[a]]
splitToParts _ [] = []
splitToParts i xs = case splitAt i xs of
    (a, b) -> a : splitToParts i b

-- | Concat a list of strings, inserting ", " between the elements
unkomma :: [String] -> String
unkomma = intercalate ", "

-- | Generate the abs', signum', abs# and signum# functions. abs' and signum' are
--   equivalent to abs and signum, but do not inline further than abs# and signum#.
genSignumAbsFuncs :: TypeDesc -> String
genSignumAbsFuncs td = unlines
    ["abs' :: " ++ t ++ " -> " ++ t
    ,"abs' (" ++ p ++ " x) = " ++ p ++ " (abs# x)"
    ,""
    ,if isWord td then "{-# INLINE abs# #-}" else "{-# NOINLINE abs# #-}"
    ,"abs# :: " ++ getUnderlyingPrimType t ++ " -> " ++ getUnderlyingPrimType t
    ,"abs# x = case abs (" ++ p ++ " x) of"
    ,"    " ++ p ++ " y -> y"
    ,""
    ,"signum' :: " ++ t ++ " -> " ++ t
    ,"signum' (" ++ p ++ " x) = " ++ p ++ " (signum# x)"
    ,""
    ,"{-# NOINLINE signum# #-}"
    ,"signum# :: " ++ getUnderlyingPrimType t ++ " -> " ++ getUnderlyingPrimType t
    ,"signum# x = case signum (" ++ p ++ " x) of"
    ,"    " ++ p ++ " y -> y"
    ]
    where
        t = getBaseType td
        p = getPrimBaseName td

-- | Generate primitive functions if we emulate vectors.
genPrimHelperFuncs :: Int -> TypeDesc -> String
genPrimHelperFuncs maxCapability td | maxCapability /= 0 = ""
                                    | otherwise          = unlines $
    [unlines [broadcastFuncSig, broadcastFunc]
    ,unlines [packFuncSig, packFunc]
    ,unlines [unpackFuncSig, unpackFunc]
    ,unlines [insertFuncSig, insertFunc]
    ] ++
    [negateFunc | isInt td] ++
    [mathFunc "plus" "+" | baseType /= "Float#"] ++
    [mathFunc "minus" "-" | baseType /= "Float#"] ++
    [mathFunc "times" "*" | baseType /= "Float#"] ++
    [mathFunc "divide" "/" | baseType == "Double#"] ++
    [mathFunc "quot" "`quot`" | not $ isFloating td] ++
    [mathFunc "rem" "`rem`" | not $ isFloating td]
    where
        baseType = getBaseType td ++ "#"
        primBaseType = getUnderlyingPrimType $ getBaseType td
        constr = getPrimBaseName td
        broadcastFuncSig = "broadcast" ++ baseType ++ " :: " ++ primBaseType ++ " -> " ++ primBaseType
        broadcastFunc = "broadcast" ++ baseType ++ " v = v"
        packFuncSig = "pack" ++ baseType ++ " :: (# " ++ primBaseType ++ " #) -> " ++ primBaseType
        packFunc = "pack" ++ baseType ++ " (# v #) = v"
        unpackFuncSig = "unpack" ++ baseType ++ " :: " ++ primBaseType ++ " -> (# " ++ primBaseType ++ " #)"
        unpackFunc = "unpack" ++ baseType ++ " v = (# v #)"
        insertFuncSig = "insert" ++ baseType ++ " :: " ++ primBaseType ++ " -> " ++ primBaseType ++ " -> Int# -> " ++ primBaseType
        insertFunc = "insert" ++ baseType ++ " _ v _ = v"
        mathFunc name op = unlines
            [name ++ baseType ++ " :: " ++ primBaseType ++ " -> " ++ primBaseType ++ " -> " ++ primBaseType
            ,name ++ baseType ++ " a b = case " ++ constr ++ " a " ++ op ++ " " ++ constr ++ " b of " ++ constr ++ " c -> c"
            ]
        negateFunc = unlines
            ["negate" ++ baseType ++ " :: " ++ primBaseType ++ " -> " ++ primBaseType
            ,"negate" ++ baseType ++ " a = case negate (" ++ constr ++ " a) of " ++ constr ++ " b -> b"
            ]

fmapStr :: String
fmapStr = if versionBranch compilerVersion >= [7, 10] then "<$>" else "`liftM`"

-- | Get the data declaration for this type as well as the instance declarations
genTypeDecl :: Int -> TypeDesc -> String
genTypeDecl maxCapability typeDesc = unlines
    [dataDecl
    ,genPrimHelperFuncs maxCapability typeDesc
    ,genSignumAbsFuncs typeDesc
    ,eqInstance
    ,ordInstance
    ,showInstance
    ,numInstance
    ,boundedInstance
    ,fractionalInstance
    ,floatingInstance
    ,storableInstance
    ,vectorInstance
    ,intVectorInstance
    ,primInstance
    ,vectorArrayInstance
    ]
    where
        (splitType, splitCount) = splitTypes maxCapability typeDesc
        primName = getPrimName splitType
        dataName = getDataName typeDesc
        baseType = getBaseType typeDesc
        vecSize  = getVectorSize typeDesc
        vars1    = map (('x':) . show) [1 :: Int .. vecSize]
        vars2    = map (('y':) . show) [1 :: Int .. vecSize]
        dataDecl = "data " ++ dataName ++ " = " ++ dataName ++ " " ++ unwords (replicate splitCount primName) ++ " deriving Typeable\n"
        eqInstance          = unlines
            ["instance Eq " ++ dataName ++ " where"
            ,"    a == b = case unpack" ++ dataName ++ " a of"
            ,"        " ++ matchTuple False vars1 ++ " -> case unpack" ++ dataName ++ " b of"
            ,"            " ++ matchTuple False vars2 ++ " -> " ++ intercalate " && " (zipWith (\ v1 v2 -> v1 ++ " == " ++ v2) vars1 vars2)
            ]
        ordInstance          = unlines
            ["instance Ord " ++ dataName ++ " where"
            ,"    a `compare` b = case unpack" ++ dataName ++ " a of"
            ,"        " ++ matchTuple False vars1 ++ " -> case unpack" ++ dataName ++ " b of"
            ,"            " ++ matchTuple False vars2 ++ " -> " ++ intercalate " <> " (zipWith (\ v1 v2 -> v1 ++ " `compare` " ++ v2) vars1 vars2)
            ]
        showInstance        = unlines
            ["instance Show " ++ dataName ++ " where"
            ,"    showsPrec _ a s = case unpack" ++ dataName ++ " a of"
            ,"        " ++ matchTuple False vars1 ++ " -> \"" ++ dataName ++ " (\" ++ " ++ foldr helper initAcc vars1
            ]
            where
                initAcc = "\")\" ++ s"
                helper x acc | acc == initAcc = "shows " ++ x ++ " (" ++ acc ++ ")"
                             | otherwise      = "shows " ++ x ++ " (\", \" ++ " ++ acc ++ ")"
        numInstance         = unlines
            ["instance Num " ++ dataName ++ " where"
            ,"    (+) = plus" ++ dataName
            ,"    (-) = minus" ++ dataName
            ,"    (*) = times" ++ dataName
            ,"    negate = " ++ (if isWord typeDesc then "mapVector negate" else "negate" ++ dataName)
            ,"    abs    = mapVector abs'"
            ,"    signum = mapVector signum'"
            ,"    fromInteger = broadcastVector . fromInteger"
            ]
        boundedInstance     = if isFloating typeDesc
            then ""
            else unlines ["instance Bounded " ++ dataName ++ " where"
                         ,"    minBound = broadcastVector minBound"
                         ,"    maxBound = broadcastVector maxBound"
                         ]
        fractionalInstance     = if not $ isFloating typeDesc
            then ""
            else unlines ["instance Fractional " ++ dataName ++ " where"
                         ,"    (/)          = divide" ++ dataName
                         ,"    recip v      = broadcastVector 1 / v"
                         ,"    fromRational = broadcastVector . fromRational"
                         ]
        floatingInstance       = if not $ isFloating typeDesc
            then ""
            else unlines
                ["instance Floating " ++ dataName ++ " where"
                ,"    pi           = broadcastVector pi"
                ,"    exp          = mapVector exp"
                ,"    sqrt         = mapVector sqrt"
                ,"    log          = mapVector log"
                ,"    (**)         = zipVector (**)"
                ,"    logBase      = zipVector (**)"
                ,"    sin          = mapVector sin"
                ,"    tan          = mapVector tan"
                ,"    cos          = mapVector cos"
                ,"    asin         = mapVector asin"
                ,"    atan         = mapVector atan"
                ,"    acos         = mapVector acos"
                ,"    sinh         = mapVector sinh"
                ,"    tanh         = mapVector tanh"
                ,"    cosh         = mapVector cosh"
                ,"    asinh        = mapVector asinh"
                ,"    atanh        = mapVector atanh"
                ,"    acosh        = mapVector acosh"
                ]
        storableInstance    = unlines
            ["instance Storable " ++ dataName ++ " where"
            ,"    sizeOf x     = vectorSize x * elementSize x"
            ,"    alignment    = Foreign.Storable.sizeOf" -- for best performance align vectors as good as their size
            ,"    peek (Ptr a) = readOffAddr (Addr a) 0"
            ,"    poke (Ptr a) = writeOffAddr (Addr a) 0"
            ]
        vectorInstance      = unlines $
            ["instance SIMDVector " ++ dataName ++ " where"
            ,"    type Elem " ++ dataName ++ " = " ++ baseType
            ,"    type ElemTuple " ++ dataName ++ " = " ++ tupleType (getVectorSize typeDesc) baseType
            ,"    nullVector         = broadcastVector 0"
            ,"    vectorSize  _      = " ++ show (getVectorSize typeDesc)
            ,"    elementSize _      = " ++ show (getElementSize typeDesc)
            ,"    broadcastVector    = broadcast" ++ dataName
            ,"    generateVector     = generate" ++ dataName
            ,"    unsafeIndexVector  = unsafeIndex" ++ dataName
            ,"    unsafeInsertVector = unsafeInsert" ++ dataName
            ,"    packVector         = pack" ++ dataName
            ,"    unpackVector       = unpack" ++ dataName
            ,"    mapVector          = map" ++ dataName
            ,"    zipVector          = zip" ++ dataName
            ,"    foldVector         = fold" ++ dataName
            ] ++ ["    sumVector          = sum" ++ dataName | splitCount > 1 && maxCapability /= 0] ++
            ["    {-# INLINE nullVector #-}"
            ,"    {-# INLINE vectorSize #-}"
            ,"    {-# INLINE elementSize #-}"
            ,"    {-# INLINE broadcastVector #-}"
            ,"    {-# INLINE generateVector #-}"
            ,"    {-# INLINE unsafeIndexVector #-}"
            ,"    {-# INLINE unsafeInsertVector #-}"
            ,"    {-# INLINE packVector #-}"
            ,"    {-# INLINE unpackVector #-}"
            ,"    {-# INLINE mapVector #-}"
            ,"    {-# INLINE zipVector #-}"
            ,"    {-# INLINE foldVector #-}"
            ] ++ ["    {-# INLINE sumVector #-}" | splitCount > 1 && maxCapability /= 0]
        intVectorInstance   = if not $ isFloating typeDesc
            then unlines
                ["instance SIMDIntVector " ++ dataName ++ " where"
                ,"    quotVector = quot" ++ dataName
                ,"    remVector  = rem" ++ dataName
                ]
            else ""
        primInstance        = unlines
            ["instance Prim " ++ dataName ++ " where"
            ,"    sizeOf# a                   = let !(I# x) = sizeOf a in x"
            ,"    alignment# a                = let !(I# x) = alignment a in x"
            ,"    indexByteArray# ba i        = index" ++ dataName ++ "Array (ByteArray ba) (I# i)"
            ,"    readByteArray# mba i s      = let (ST r) = read" ++ dataName ++ "Array (MutableByteArray mba) (I# i) in r s"
            ,"    writeByteArray# mba i v s   = let (ST r) = write" ++ dataName ++ "Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }"
            ,"    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }"
            ,"    indexOffAddr# addr i        = index" ++ dataName ++ "OffAddr (Addr addr) (I# i)"
            ,"    readOffAddr# addr i s       = let (ST r) = read" ++ dataName ++ "OffAddr (Addr addr) (I# i) in r s"
            ,"    writeOffAddr# addr i v s    = let (ST r) = write" ++ dataName ++ "OffAddr" ++ " (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }"
            ,"    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }"
            ]
        vectorArrayInstance = unlines
            ["newtype instance UV.Vector " ++ dataName ++ " = V_" ++ dataName ++ " (PV.Vector " ++ dataName ++ ")"
            ,"newtype instance UV.MVector s " ++ dataName ++ " = MV_" ++ dataName ++ " (PMV.MVector s " ++ dataName ++ ")"
            ,""
            ,"instance Vector UV.Vector " ++ dataName ++ " where"
            ,"    basicUnsafeFreeze (MV_" ++ dataName ++ " v) = V_" ++ dataName ++ " " ++ fmapStr ++ " PV.unsafeFreeze v"
            ,"    basicUnsafeThaw (V_" ++ dataName ++ " v) = MV_" ++ dataName ++ " " ++ fmapStr ++ " PV.unsafeThaw v"
            ,"    basicLength (V_" ++ dataName ++ " v) = PV.length v"
            ,"    basicUnsafeSlice start len (V_" ++ dataName ++ " v) = V_" ++ dataName ++ "(PV.unsafeSlice start len v)"
            ,"    basicUnsafeIndexM (V_" ++ dataName ++ " v) = PV.unsafeIndexM v"
            ,"    basicUnsafeCopy (MV_" ++ dataName ++ " m) (V_" ++ dataName ++ " v) = PV.unsafeCopy m v"
            ,"    elemseq _ = seq"
            ,"    {-# INLINE basicUnsafeFreeze #-}"
            ,"    {-# INLINE basicUnsafeThaw #-}"
            ,"    {-# INLINE basicLength #-}"
            ,"    {-# INLINE basicUnsafeSlice #-}"
            ,"    {-# INLINE basicUnsafeIndexM #-}"
            ,"    {-# INLINE basicUnsafeCopy #-}"
            ,"    {-# INLINE elemseq #-}"
            ,""
            ,"instance MVector UV.MVector " ++ dataName ++ " where"
            ,"    basicLength (MV_" ++ dataName ++ " v) = PMV.length v"
            ,"    basicUnsafeSlice start len (MV_" ++ dataName ++ " v) = MV_" ++ dataName ++ "(PMV.unsafeSlice start len v)"
            ,"    basicOverlaps (MV_" ++ dataName ++ " v) (MV_" ++ dataName ++ " w) = PMV.overlaps v w"
            ,"    basicUnsafeNew len = MV_" ++ dataName ++ " " ++ fmapStr ++ " PMV.unsafeNew len"
            ,"#if MIN_VERSION_vector(0,11,0)"
            ,"    basicInitialize (MV_" ++ dataName ++ " v) = basicInitialize v"
            ,"#endif"
            ,"    basicUnsafeRead (MV_" ++ dataName ++ " v) = PMV.unsafeRead v"
            ,"    basicUnsafeWrite (MV_" ++ dataName ++ " v) = PMV.unsafeWrite v"
            ,"    {-# INLINE basicLength #-}"
            ,"    {-# INLINE basicUnsafeSlice #-}"
            ,"    {-# INLINE basicOverlaps #-}"
            ,"    {-# INLINE basicUnsafeNew #-}"
            ,"    {-# INLINE basicUnsafeRead #-}"
            ,"    {-# INLINE basicUnsafeWrite #-}"
            ,""
            ,"instance Unbox " ++ dataName
            ]

getTypeInfo :: TypeDesc -> (String, String, String)
getTypeInfo typeDesc = (getDataName typeDesc, getPrimBaseName typeDesc, getBaseType typeDesc)

getExtendedTypeInfo :: Int -> TypeDesc -> (String, String, String, TypeDesc, Int, String)
getExtendedTypeInfo maxCapability typeDesc = case getTypeInfo typeDesc of
    (dataName, primBaseName, baseType) -> case splitTypes maxCapability typeDesc of
        (splitType, splitCount) -> (dataName, primBaseName, baseType, splitType, splitCount, getDataName splitType)

-- | Generate a pattern synonym for a given vector size.
genPatSynonym :: Bool -> Int -> String
genPatSynonym patSigs n = unlines
    ["-- | Convenient way to match against and construct " ++ show n ++ "-ary vectors."
    ,patSigPrefix ++ "pattern Vec" ++ show n ++ " :: (ElemTuple v ~ " ++ tuplT ++ ", SIMDVector v)"
    ,patSigPrefix ++ "    => " ++ targs ++ " v"
    ,"pattern Vec" ++ show n ++ " " ++ args ++ " <- (unpackVector -> " ++ tuple ++ ") where"
    ,"    Vec" ++ show n ++ " " ++ args ++ " = packVector " ++ tuple
    ]
    where
        patSigPrefix = if patSigs then "" else "-- "
        tuplT, targs, args, tuple :: String
        tuplT = tupleType n "a"
        targs = concat $ replicate n "a -> "
        vars  = ['x' : show x | x <- [1..n]]
        args  = unwords vars
        tuple = matchTuple True vars

-- | Generate a function to broadcast a value to all elements of the vector
getBroadCastFunc :: Int -> TypeDesc -> String
getBroadCastFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, primBaseName, baseType, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "broadcast" ++ dataName
        primFuncName = "broadcast" ++ splitDataName ++ "#"
        funcDoc  = "-- | Broadcast a scalar to all elements of a vector."
        funcSig  = funcName ++ " :: " ++ baseType ++ " -> " ++ dataName
        funcImpl = funcName ++ " (" ++ primBaseName ++ " x) = " ++ funcImplRhs
        funcImplRhs | splitCount == 1 = dataName ++ " (" ++ primFuncName ++ " x)"
                    | otherwise       = "case " ++ primFuncName ++ " x of\n\tv -> " ++ unwords (dataName : replicate splitCount "v")

-- | Generate a function to pack values to a vector
getPackFunc :: Int -> TypeDesc -> String
getPackFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, primBaseName, _, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        vecSize  = getVectorSize typeDesc
        vars     = map (('x':) . show) [1 :: Int .. vecSize]
        primVars = map (\ v -> primBaseName ++ " " ++ v) vars
        splitVars = splitIntoPackets vars splitCount
        funcName = "pack" ++ dataName
        primFuncName = "pack" ++ splitDataName ++ "#"
        funcDoc  = "-- | Pack the elements of a tuple into a vector."
        funcSig  = funcName ++ " :: " ++ tupleType vecSize (getBaseType typeDesc) ++ " -> " ++ dataName
        funcImpl = funcName ++ " " ++ matchTuple True primVars ++ " = " ++ dataName ++ concatMap (\ v -> " (" ++ primFuncName ++ " (# " ++ unkomma v ++ " #))") splitVars

-- | Generate a function to unpack values from a vector
getUnpackFunc :: Int -> TypeDesc -> String
getUnpackFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, primBaseName, baseType, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        vecSize  = getVectorSize typeDesc
        primVars = map (\ v -> primBaseName ++ " " ++ v) vars
        vars     = map (('x':) . show) [1 :: Int .. vecSize]
        splitVars = splitIntoPackets vars splitCount
        matchVars = map (('m':) . show) [1..splitCount]
        funcName = "unpack" ++ dataName
        primFuncName = "unpack" ++ splitDataName ++ "#"
        funcDoc  = "-- | Unpack the elements of a vector into a tuple."
        funcSig  = funcName ++ " :: " ++ dataName ++ " -> " ++ tupleType vecSize baseType
        funcImpl = funcName ++ " (" ++ dataName ++ " " ++ unwords matchVars ++ ") = " ++ buildCaseMatches matchVars splitVars "" ++ matchTuple False primVars
        buildCaseMatches [] [] _ = ""
        buildCaseMatches (m:ms) (sl:sls) ind = "case " ++ primFuncName ++ " " ++ m ++ " of\n\t" ++ ind ++ "(# " ++ unkomma sl ++ " #) -> " ++ buildCaseMatches ms sls ('\t':ind)
        buildCaseMatches _ _ _ = error "no length match"

toCases :: [(String, String)] -> [String]
toCases [] = error "toCases: empty list"
toCases [(_, x)] = [x]
toCases l = foldr toCasesHelper [] l
    where
        toCasesHelper :: (String, String) -> [String] -> [String]
        toCasesHelper (_, x) [] = ["| otherwise " ++ x]
        toCasesHelper (c, x) xs = ("| " ++ c ++ " " ++ x) : xs

-- | Generate a function to index values out of a vector
getIndexFunc :: Int -> TypeDesc -> String
getIndexFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl, "    where", indexSubElem]
    where
        (dataName, primBaseName, baseType, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        incCount     = getVectorSize splitType
        funcName = "unsafeIndex" ++ dataName
        matchVars = map (('m':) . show) [1 .. splitCount]
        funcDoc  = "-- | Extract a scalar from the given position (starting from 0) out of a vector. "
            ++ "If the index is outside of the range, the behavior is undefined."
        funcSig  = funcName ++ " :: " ++ dataName ++ " -> Int -> " ++ baseType
        funcStart = funcName ++ " (" ++ dataName ++ " " ++ unwords matchVars ++ ") i "
        allStarts = funcStart : repeat wsStart
        wsStart   = replicate (length funcStart) ' '
        funcImpl  = intercalate "\n" $ zipWith (++) allStarts (toCases matchCases)
        matchCases = map buildMatchCase [0..splitCount-1]
        buildMatchCase :: Int -> (String, String)
        buildMatchCase ix = ("i < " ++ show ((ix + 1) * incCount), "= " ++ primBaseName ++ " (indexSubElem (i - " ++ show (ix * incCount) ++ ") " ++ (matchVars !! ix) ++ ")")
        indexArgName = if incCount == 1
            then getUnderlyingPrimType (getBaseType splitType)
            else splitDataName ++ "#"
        indexSubElemSig = "indexSubElem :: Int -> " ++ indexArgName ++ " -> " ++ getUnderlyingPrimType (getBaseType splitType)
        indexSubElem = unlines $ map (replicate 8 ' ' ++) $
            [indexSubElemSig
            ,"indexSubElem n vec# = case unpack" ++ splitDataName ++ "# vec# of"
            ,replicate 4 ' ' ++ "(# " ++ intercalate ", " indexSubElemVars ++ " #) -> case n of"
            ] ++ [replicate 8 ' ' ++ (if n == (incCount - 1) then "_" else show n) ++ " -> " ++ indexSubElemVars !! n | n <- [0 .. incCount - 1]]
        indexSubElemVars = map (('v':) . show) [1 .. incCount]

-- | Generate a function to insert values into a vector
getInsertFunc :: Int -> TypeDesc -> String
getInsertFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, primBaseName, baseType, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        primFuncName = "insert" ++ splitDataName ++ "#"
        incCount     = getVectorSize splitType
        funcName = "unsafeInsert" ++ dataName
        matchVars = map (('m':) . show) [1 .. splitCount]
        funcDoc  = "-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined."
        funcSig  = funcName ++ " :: " ++ dataName ++ " -> " ++ baseType ++ " -> Int -> " ++ dataName
        funcStart = funcName ++ " (" ++ dataName ++ " " ++ unwords matchVars ++ ") (" ++ primBaseName ++ " y) _i@(I# ip) "
        wsStart   = replicate (length funcStart) ' '
        allStarts = funcStart : repeat wsStart
        funcImpl  = unlines $ zipWith (++) allStarts (toCases matchCases)
        matchCases = map buildMatchCase [0..splitCount-1]
        buildMatchCase :: Int -> (String, String)
        buildMatchCase ix = ("_i < " ++ show ((ix + 1) * incCount ), "= " ++ dataName ++ " " ++ unwords (take ix matchVars ++ ["(" ++ primFuncName ++ " " ++ matchVars !! ix ++ " y (ip -# " ++ show (ix * incCount) ++ "#))"] ++ drop (ix + 1) matchVars))

-- | Generate a function to generate a vector from an index-function
getGenerateFunc :: TypeDesc -> String
getGenerateFunc typeDesc = unlines
    ["{-# INLINE " ++ funcName ++ " #-}"
    ,"-- | The vector that results from applying the given function to all indices in"
    ,"--   the range @0 .. " ++ show (getVectorSize typeDesc - 1) ++ "@."
    ,funcName ++ " :: (Int -> " ++ elemName ++ ") -> " ++ dataName
    ,funcName ++ " f = " ++ funcName ++ "# (\\ i -> case f (I# i) of { " ++ primName ++ " x -> x })"
    ,""
    ,"{-# INLINE[0] " ++ funcName ++ "# #-}"
    ,"-- | Unboxed helper function."
    ,funcName ++ "# :: (Int# -> " ++ getUnderlyingPrimType elemName ++ ") -> " ++ dataName
    ,funcName ++ "# f = inline pack" ++ dataName ++ " " ++ matchTuple True (map (\ i -> primName ++ " (f "++ i ++ "#)") indices)
    ]
    where
        (dataName, primName, elemName) = getTypeInfo typeDesc
        funcName = "generate" ++ dataName
        indices  = map show [0 .. getVectorSize typeDesc - 1]

-- | Generate a function to map a function over a vector
getMapFunc :: TypeDesc -> String
getMapFunc typeDesc = unlines
    ["{-# INLINE[1] " ++ funcName ++ " #-}"
    ,"-- | Apply a function to each element of a vector (unpacks and repacks the vector)"
    ,funcName ++ " :: (" ++ elemName ++ " -> " ++ elemName ++ ") -> " ++ dataName ++ " -> " ++ dataName
    ,funcName ++ " f = " ++ funcName ++ "# (\\ x -> case f (" ++ primName ++ " x) of { " ++ primName ++ " y -> y})"
    ,""
    ,"{-# RULES \"mapVector abs\" " ++ funcName ++ " abs = abs #-}"
    ,"{-# RULES \"mapVector signum\" " ++ funcName ++ " signum = signum #-}"
    ,if isWord typeDesc then "" else "{-# RULES \"mapVector negate\" " ++ funcName ++ " negate = negate #-}"
    ,"{-# RULES \"mapVector const\" forall x . " ++ funcName ++ " (const x) = const (broadcastVector x) #-}"
    ,"{-# RULES \"mapVector (x+)\" forall x v . " ++ funcName ++ " (\\ y -> x + y) v = broadcastVector x + v #-}"
    ,"{-# RULES \"mapVector (+x)\" forall x v . " ++ funcName ++ " (\\ y -> y + x) v = v + broadcastVector x #-}"
    ,"{-# RULES \"mapVector (x-)\" forall x v . " ++ funcName ++ " (\\ y -> x - y) v = broadcastVector x - v #-}"
    ,"{-# RULES \"mapVector (-x)\" forall x v . " ++ funcName ++ " (\\ y -> y - x) v = v - broadcastVector x #-}"
    ,"{-# RULES \"mapVector (x*)\" forall x v . " ++ funcName ++ " (\\ y -> x * y) v = broadcastVector x * v #-}"
    ,"{-# RULES \"mapVector (*x)\" forall x v . " ++ funcName ++ " (\\ y -> y * x) v = v * broadcastVector x #-}"
    ,if isFloating typeDesc then "{-# RULES \"mapVector (x/)\" forall x v . " ++ funcName ++ " (\\ y -> x / y) v = broadcastVector x / v #-}" else ""
    ,if isFloating typeDesc then "{-# RULES \"mapVector (/x)\" forall x v . " ++ funcName ++ " (\\ y -> y / x) v = v / broadcastVector x #-}" else ""
    ,if not $ isFloating typeDesc then "{-# RULES \"mapVector (`quot` x)\" forall x v . " ++ funcName ++ " (\\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}" else ""
    ,if not $ isFloating typeDesc then "{-# RULES \"mapVector (x `quot`)\" forall x v . " ++ funcName ++ " (\\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}" else ""
    ,""
    ,"{-# INLINE[0] " ++ funcName ++ "# #-}"
    ,"-- | Unboxed helper function."
    ,funcName ++ "# :: (" ++ getUnderlyingPrimType elemName ++ " -> " ++ getUnderlyingPrimType elemName ++ ") -> " ++ dataName ++ " -> " ++ dataName
    ,funcName ++ "# f = \\ v -> case unpack" ++ dataName ++ " v of"
    ,"    " ++ matchTuple False (map (\ v -> primName ++ " " ++ v) vars) ++ " -> pack" ++ dataName ++ " " ++ matchTuple True (map (\ v -> primName ++ " (f "++ v ++ ")") vars)
    ]
    where
        (dataName, primName, elemName) = getTypeInfo typeDesc
        funcName = "map" ++ dataName
        vars     = map (('x':) . show) [1 :: Int .. getVectorSize typeDesc]

-- | Generate a function to zip two vectors with a function
getZipFunc :: TypeDesc -> String
getZipFunc typeDesc = unlines ["{-# INLINE[1] " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl, zipRules]
    where
        (dataName, _, elemName) = getTypeInfo typeDesc
        funcName = "zip" ++ dataName
        vars1    = map (('x':) . show) [1 :: Int .. getVectorSize typeDesc]
        vars2    = map (('y':) . show) [1 :: Int .. getVectorSize typeDesc]
        funcDoc  = "-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)"
        funcSig  = funcName ++ " :: (" ++ elemName ++ " -> " ++ elemName ++ " -> " ++ elemName ++ ") -> " ++ dataName ++ " -> " ++ dataName ++ " -> " ++ dataName
        funcImpl = unlines [funcName ++ " f = \\ v1 v2 -> case unpack" ++ dataName ++ " v1 of"
                           ,"    " ++ matchTuple False vars1 ++ " -> case unpack" ++ dataName ++ " v2 of"
                           ,"        " ++ matchTuple False vars2 ++ " -> pack" ++ dataName ++ " " ++ matchTuple True (zipWith (\ v1 v2 -> "f "++ v1 ++ " " ++ v2) vars1 vars2)
                           ]
        baseZipRules = map (\ c -> genZipRule [c] ['(', c, ')']) "+-*"
        zipRules | isFloating typeDesc = intercalate "\n" $ baseZipRules ++ [genZipRule "/" "(/)"]
                 | otherwise           = intercalate "\n" $ baseZipRules ++ [genZipRule "`quotVector`" "quot", genZipRule "`remVector`" "rem"]
        genZipRule op opP = "{-# RULES \"zipVector " ++ op ++ "\" forall a b . " ++ funcName ++ " " ++ opP ++ " a b = a " ++ op ++ " b #-}"


-- | Generate a function to fold a vector with a function. See Note [Pipelining].
getFoldFunc :: Int -> TypeDesc -> String
getFoldFunc maxCapability typeDesc = unlines ["{-# INLINE[1] " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl, foldRule]
    where
        (dataName, _, elemName, _, splitCount, _) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "fold" ++ dataName
        vars     = map (('x':) . show) [1 :: Int .. getVectorSize typeDesc]
        funcDoc  = "-- | Fold the elements of a vector to a single value"
        funcSig  = funcName ++ " :: (" ++ elemName ++ " -> " ++ elemName ++ " -> " ++ elemName ++ ") -> " ++ dataName ++ " -> " ++ elemName
        funcImpl = unlines [funcName ++ " f' = \\ v -> case unpack" ++ dataName ++ " v of"
                           ,"    " ++ matchTuple False vars ++ " -> " ++ intercalate " `f` " vars
                           ,"    where f !x !y = f' x y"
                           ]
        foldRule = if splitCount == 1 || maxCapability == 0 then "" else "{-# RULES \"foldVector (+)\" " ++ funcName ++ " (+) = sumVector #-}"

-- | Generate a function to sum up a vector. See Note [Pipelining].
getSumFunc :: Int -> TypeDesc -> String
getSumFunc maxCapability typeDesc | splitCount == 1 || maxCapability == 0 = ""
                                  | otherwise = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, primBaseName, baseType, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "sum" ++ dataName
        primSumFunc = "plus" ++ splitDataName ++ "#"
        vars     = map (('x':) . show) [1 :: Int .. splitCount]
        matchVars = map (('y':) . show) [1 .. getVectorSize splitType]
        funcDoc  = "-- | Sum up the elements of a vector to a single value."
        funcSig  = funcName ++ " :: " ++ dataName ++ " -> " ++ baseType
        sumIt    = foldr1 (\ v s -> unwords [primSumFunc, v, addParens s]) vars
        funcImpl = unlines [funcName ++ " (" ++ dataName ++ " " ++ unwords vars ++ ") = case unpack" ++ splitDataName ++ "# " ++ addParens sumIt ++ " of"
                           ,"    (# " ++ unkomma matchVars ++ " #) -> " ++ intercalate " + " (map ((primBaseName++" ")++) matchVars)
                           ]

-- | Generate a function to perform arbitrary arithmetic with a vector
getArithmeticFunc :: Int -> Int -> String -> String -> TypeDesc -> String
getArithmeticFunc maxCapability argCount funcBaseName funcDoc typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = funcBaseName ++ dataName
        primFuncName = funcBaseName ++ splitDataName ++ "#"
        matchVars = map (('m':) . show) [1..splitCount]
        vars     = map (\ x -> map (\ m -> m ++ "_" ++ show x) matchVars) [1 :: Int .. argCount]
        argVars  = map (\ v -> "(" ++ dataName ++ " " ++ unwords v ++ ")") vars
        funcSig  = funcName ++ " :: " ++ dataName ++ concat (replicate argCount $ " -> " ++ dataName)
        funcImpl = funcName ++ " " ++ unwords argVars ++ " = " ++
            dataName ++ concatMap (\ v -> " (" ++ primFuncName ++ " " ++ unwords v ++ ")") (transpose vars)

getDivideFunc :: Int -> TypeDesc -> String
getDivideFunc maxCapability typeDesc = if isFloating typeDesc
    then getArithmeticFunc maxCapability 2 "divide" "-- | Divide two vectors element-wise." typeDesc
    else ""

getQuotFunc :: Int -> TypeDesc -> String
getQuotFunc maxCapability typeDesc = if not $ isFloating typeDesc
    then getArithmeticFunc maxCapability 2 "quot" "-- | Rounds towards zero element-wise." typeDesc
    else ""

getRemFunc :: Int -> TypeDesc -> String
getRemFunc maxCapability typeDesc = if not $ isFloating typeDesc
    then getArithmeticFunc maxCapability 2 "rem" "-- | Satisfies (quot x y) * y + (rem x y) == x." typeDesc
    else ""

getNegateFunc :: Int -> TypeDesc -> String
getNegateFunc maxCapability typeDesc = if not $ isWord typeDesc
    then getArithmeticFunc maxCapability 1 "negate" "-- | Negate element-wise." typeDesc
    else ""

unfoldReads :: String -> String -> [Int] -> String -> String
unfoldReads iName primFuncName offsets dataName = helper (zip matchVars offsets) 0 "\t"
    where
        matchVars :: [String]
        matchVars = map (('m':) . show) [1 :: Int .. length offsets]
        helper :: [(String, Int)] -> Int -> String -> String
        helper [] sIndex _ = "(# s" ++ show sIndex ++ ", " ++ dataName ++ " " ++ unwords matchVars ++ " #)"
        helper ((m,off):xs) sIndex indention = "case " ++
            primFuncName ++ " a (" ++ iName ++ " +# " ++ show off ++ "#) s" ++ show sIndex ++
            " of\n" ++
                indention ++ "(# s" ++ show (sIndex + 1) ++ ", " ++ m ++ " #) -> " ++ helper xs (sIndex + 1) ('\t':indention)

{-

GHC prim ops:

-- byte arrays

-- | Read a vector from specified index of immutable array.
indexVVVArray# :: ByteArray# -> Int# -> VVV#

-- | Read a vector from specified index of immutable array of scalars; offset is in scalar elements.
indexEEEArrayAsVVV# :: ByteArray# -> Int# -> VVV#

-- | Read a vector from specified index of mutable array.
readVVVArray# :: MutableByteArray# s -> Int# -> State# s -> (#State# s, VVV##)

-- | Read a vector from specified index of mutable array of scalars; offset is in scalar elements.
readEEEArrayAsVVV# :: MutableByteArray# s -> Int# -> State# s -> (#State# s, VVV##)

-- | Write a vector to specified index of mutable array.
writeVVVArray# :: MutableByteArray# s -> Int# -> VVV# -> State# s -> State# s

-- | Write a vector to specified index of mutable array of scalars; offset is in scalar elements.
writeEEEArrayAsVVV# :: MutableByteArray# s -> Int# -> VVV# -> State# s -> State# s

-- pointers

-- | Reads vector; offset in bytes.
indexVVVOffAddr# :: Addr# -> Int# -> VVV#

-- | Reads vector; offset in scalar elements.
indexEEEOffAddrAsVVV# :: Addr# -> Int# -> VVV#

-- | Reads vector; offset in bytes.
readVVVOffAddr# :: Addr# -> Int# -> State# s -> (#State# s, VVV##)

-- | Reads vector; offset in scalar elements.
readEEEOffAddrAsVVV# :: Addr# -> Int# -> State# s -> (#State# s, VVV##)

-- | Write vector; offset in bytes.
writeVVVOffAddr# :: Addr# -> Int# -> VVV# -> State# s -> State# s

-- | Write vector; offset in scalar elements.
writeEEEOffAddrAsVVV# :: Addr# -> Int# -> VVV# -> State# s -> State# s

-}

getIndexByteArrayFunc :: Int -> TypeDesc -> String
getIndexByteArrayFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "index" ++ dataName ++ "Array"
        primFuncName = "index" ++ splitDataName ++ "Array#"
        funcDoc  = "-- | Read a vector from specified index of the immutable array."
        funcSig  = funcName ++ " :: ByteArray -> Int -> " ++ dataName
        funcImpl = funcName ++ " (ByteArray a) (I# i) = " ++ dataName ++ funcImplBody
        funcImplBody | splitCount == 1 = " (" ++ primFuncName ++ " a i)"
                     | otherwise       = concatMap (\ i -> " (" ++ primFuncName ++ " a ((i *# " ++ show splitCount ++ "#) +# " ++ show i ++ "#))") [0..splitCount-1]

getReadByteArrayFunc :: Int -> TypeDesc -> String
getReadByteArrayFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "read" ++ dataName ++ "Array"
        primFuncName = "read" ++ splitDataName ++ "Array#"
        funcDoc  = "-- | Read a vector from specified index of the mutable array."
        funcSig  = funcName ++ " :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m " ++ dataName
        funcImpl = funcName ++ " (MutableByteArray a) (I# i) = primitive (\\ s0 -> " ++ unfoldReads ("(i *# " ++ show splitCount ++ "#)") primFuncName [0..splitCount-1] dataName ++ ")"

getWriteByteArrayFunc :: Int -> TypeDesc -> String
getWriteByteArrayFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, _, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "write" ++ dataName ++ "Array"
        primFuncName = "write" ++ splitDataName ++ "Array#"
        matchVars = map (('m':) . show) [1..splitCount]
        funcDoc  = "-- | Write a vector to specified index of mutable array."
        funcSig  = funcName ++ " :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> " ++ dataName ++ " -> m ()"
        funcImpl = funcName ++ " (MutableByteArray a) (I# i) (" ++ dataName ++ " " ++ unwords matchVars ++ ") = " ++ intercalate " >> " (zipWith helper matchVars [(0 :: Int)..])
        helper m i = "primitive_ (" ++ primFuncName ++ " a ((i *# " ++ show splitCount ++ "#) +# " ++ show i ++ "#) " ++ m ++ ")"

getIndexOffAddrFunc :: Int -> TypeDesc -> String
getIndexOffAddrFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "index" ++ dataName ++ "OffAddr"
        splitSize = getVectorSize splitType
        subVectorSize = splitSize * getElementSize typeDesc
        primFuncName = "index" ++ splitDataName ++ "OffAddr#"
        funcDoc  = "-- | Reads vector from the specified index of the address."
        funcSig  = funcName ++ " :: Addr -> Int -> " ++ dataName
        funcImpl = funcName ++ " (Addr a) (I# i) = " ++ dataName ++ funcImplBody
        funcImplBody | splitCount == 1 = " (" ++ primFuncName ++ " (plusAddr# a (i *# " ++ show subVectorSize ++ "#)) 0#)"
                     | otherwise       = concatMap (\ i -> " (" ++ primFuncName ++ " (plusAddr# a ((i *# " ++ show (splitCount * subVectorSize) ++ "#) +# " ++ show (i * subVectorSize) ++ "#)) 0#)") [0..splitCount-1]

getReadOffAddrFunc :: Int -> TypeDesc -> String
getReadOffAddrFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        splitSize = getVectorSize splitType
        subVectorSize = splitSize * getElementSize typeDesc
        funcName = "read" ++ dataName ++ "OffAddr"
        primFuncName = "(\\ addr i' -> read" ++ splitDataName ++ "OffAddr# (plusAddr# addr i') 0#)"
        funcDoc  = "-- | Reads vector from the specified index of the address."
        funcSig  = funcName ++ " :: PrimMonad m => Addr -> Int -> m " ++ dataName
        funcImpl = funcName ++ " (Addr a) (I# i) = primitive (\\ s0 -> " ++ funcImplBody ++ ")"
        funcImplBody = unfoldReads ("(i *# " ++ show (splitCount * subVectorSize) ++ "#)") primFuncName [0,subVectorSize..subVectorSize * (splitCount-1)] dataName

getWriteOffAddrFunc :: Int -> TypeDesc -> String
getWriteOffAddrFunc maxCapability typeDesc = unlines ["{-# INLINE " ++ funcName ++ " #-}", funcDoc, funcSig, funcImpl]
    where
        (dataName, _, _, splitType, splitCount, splitDataName) = getExtendedTypeInfo maxCapability typeDesc
        funcName = "write" ++ dataName ++ "OffAddr"
        primFuncName = "write" ++ splitDataName ++ "OffAddr#"
        splitSize = getElementSize splitType * getVectorSize splitType
        matchVars = map (('m':) . show) [1..splitCount]
        funcDoc  = "-- | Write vector to the specified index of the address."
        funcSig  = funcName ++ " :: PrimMonad m => Addr -> Int -> " ++ dataName ++ " -> m ()"
        funcImpl = funcName ++ " (Addr a) (I# i) (" ++ dataName ++ " " ++ unwords matchVars ++ ") = " ++ intercalate " >> " (zipWith helper matchVars [0..])
        helper m i = "primitive_ (" ++ primFuncName ++ " (plusAddr# a ((i *# " ++ show (splitCount * splitSize) ++ "#) +# " ++ show (i * splitSize) ++ "#)) 0# " ++ m ++ ")"

generatorFuncs :: Int -> [TypeDesc -> String]
generatorFuncs maxCapability =
    [getBroadCastFunc maxCapability
    ,getGenerateFunc
    ,getPackFunc maxCapability
    ,getUnpackFunc maxCapability
    ,getIndexFunc maxCapability
    ,getInsertFunc maxCapability
    ,getMapFunc
    ,getZipFunc
    ,getFoldFunc maxCapability
    ,getSumFunc maxCapability
    ,getArithmeticFunc maxCapability 2 "plus" "-- | Add two vectors element-wise."
    ,getArithmeticFunc maxCapability 2 "minus" "-- | Subtract two vectors element-wise."
    ,getArithmeticFunc maxCapability 2 "times" "-- | Multiply two vectors element-wise."
    ,getDivideFunc maxCapability
    ,getQuotFunc maxCapability
    ,getRemFunc maxCapability
    ,getNegateFunc maxCapability
    ,getIndexByteArrayFunc maxCapability
    ,getReadByteArrayFunc maxCapability
    ,getWriteByteArrayFunc maxCapability
    ,getIndexOffAddrFunc maxCapability
    ,getReadOffAddrFunc maxCapability
    ,getWriteOffAddrFunc maxCapability
    ]

generateTypeCode :: Int -> TypeDesc -> String
generateTypeCode maxCapability typeDesc = unlines [dataDoc, dataDecl, funcImpls]
    where
        dataDoc   = "-- ** " ++ getDataName typeDesc
        dataDecl  = genTypeDecl maxCapability typeDesc
        funcImpls = unlines $ filter (not . null) $ map ($ typeDesc) (generatorFuncs maxCapability)

data PatsMode = NoPats | NoPatSigs | Pats deriving Eq

classFile :: PatsMode -> Bool -> String
classFile genPatSyns doRules = unlines $
    ["{-# LANGUAGE TypeFamilies          #-}"
    ,"{-# LANGUAGE MultiParamTypeClasses #-}"
    ,"{-# LANGUAGE FlexibleContexts      #-}"
    ,if genPatSyns /= NoPats then "{-# LANGUAGE PatternSynonyms       #-}" else ""
    ,if genPatSyns /= NoPats then "{-# LANGUAGE ViewPatterns          #-}" else ""
    ,if doRules then "{-# LANGUAGE MagicHash             #-}" else ""
    ,if doRules then "{-# OPTIONS_GHC -fno-warn-orphans  #-}" else ""
    ,"module Data.Primitive.SIMD.Class where"
    ,""
    ,"-- This code was AUTOMATICALLY generated, DO NOT EDIT!"
    ,""
    ,"import Control.Monad.Primitive"
    ,"import Data.Primitive"
    ,""
    ,if doRules then "import GHC.Exts" else ""
    ,""
    ,if maxTupleSize < 64 then "-- | The compiler only supports tuples up to "
        ++ show maxTupleSize ++ " elements, so we have to use our own data type." else ""
    ,if maxTupleSize < 64 then "data Tuple64 a = Tuple64" ++ concat (replicate 64 " a") else ""
    ,""
    ,"-- * SIMD type classes"
    ,""
    ,"-- | This class provides basic operations to create and consume SIMD types."
    ,"--   Numeric operations on members of this class should compile to single"
    ,"--   SIMD instructions although not all operations are (yet) supported by"
    ,"--   GHC (e.g. 'sqrt', it is currently implemented as @mapVector sqrt@ which"
    ,"--   has to unpack the vector, compute the results and pack them again)."
    ,"class (Num v, Real (Elem v)) => SIMDVector v where"
    ,"    -- | Type of the elements in the vector"
    ,"    type Elem v"
    ,"    -- | Type used to pack or unpack the vector"
    ,"    type ElemTuple v"
    ,"    -- | Vector with all elements initialized to zero."
    ,"    nullVector       :: v"
    ,"    -- | Number of components (scalar elements) in the vector. The argument is not evaluated."
    ,"    vectorSize       :: v -> Int"
    ,"    -- | Size of each (scalar) element in the vector in bytes. The argument is not evaluated."
    ,"    elementSize      :: v -> Int"
    ,"    -- | Broadcast a scalar to all elements of a vector."
    ,"    broadcastVector  :: Elem v -> v"
    ,"    -- | The vector that results from applying the given function to all indices in"
    ,"    --   the range @0 .. 'vectorSize' - 1@."
    ,"    generateVector   :: (Int -> Elem v) -> v"
    ,"    -- | Extract a scalar from the given position (starting from 0) out of a vector."
    ,"    --   If the index is outside of the range an exception is thrown."
    ,"    {-# INLINE indexVector #-}"
    ,"    indexVector     :: v -> Int -> Elem v"
    ,"    indexVector v i | i < 0            = error $ \"indexVector: negative argument: \" ++ show i"
    ,"                    | i < vectorSize v = unsafeIndexVector v i"
    ,"                    | otherwise        = error $ \"indexVector: argument too large: \" ++ show i"
    ,"    -- | Extract a scalar from the given position (starting from 0) out of a vector."
    ,"    --   If the index is outside of the range the behavior is undefined."
    ,"    unsafeIndexVector     :: v -> Int -> Elem v"
    ,"    -- | Insert a scalar at the given position (starting from 0) in a vector."
    ,"    --   If the index is outside of the range an exception is thrown."
    ,"    {-# INLINE insertVector #-}"
    ,"    insertVector     :: v -> Elem v -> Int -> v"
    ,"    insertVector v e i | i < 0            = error $ \"insertVector: negative argument: \" ++ show i"
    ,"                       | i < vectorSize v = unsafeInsertVector v e i"
    ,"                       | otherwise        = error $ \"insertVector: argument too large: \" ++ show i"
    ,"    -- | Insert a scalar at the given position (starting from 0) in a vector."
    ,"    --   If the index is outside of the range the behavior is undefined."
    ,"    unsafeInsertVector     :: v -> Elem v -> Int -> v"
    ,"    -- | Apply a function to each element of a vector. Be very careful not to map"
    ,"    --   branching functions over a vector as they could lead to quite a bit of"
    ,"    --   code bloat (or make sure they are tagged with NOINLINE)."
    ,"    mapVector        :: (Elem v -> Elem v) -> v -> v"
    ,"    -- | Zip two vectors together using a combining function."
    ,"    zipVector        :: (Elem v -> Elem v -> Elem v) -> v -> v -> v"
    ,"    -- | Fold the elements of a vector to a single value. The order in which"
    ,"    --   the elements are combined is not specified."
    ,"    foldVector       :: (Elem v -> Elem v -> Elem v) -> v -> Elem v"
    ,"    -- | Sum up the components of the vector. Equivalent to @foldVector (+)@."
    ,"    sumVector        :: v -> Elem v"
    ,"    sumVector        = foldVector (+)"
    ,"    -- | Pack some elements to a vector."
    ,"    packVector       :: ElemTuple v -> v"
    ,"    -- | Unpack a vector."
    ,"    unpackVector     :: v -> ElemTuple v"
    ,""
    ,"-- | Provides vectorized versions of 'quot' and 'rem'. Implementing their"
    ,"--   type class is not possible for SIMD types as it would require"
    ,"--   implementing 'toInteger'."
    ,"class SIMDVector v => SIMDIntVector v where"
    ,"    -- | Rounds towards zero element-wise."
    ,"    quotVector :: v -> v -> v"
    ,"    -- | Satisfies @(quotVector x y) * y + (remVector x y) == x@."
    ,"    remVector  :: v -> v -> v"
    ,""
    ,"{-# INLINE setByteArrayGeneric #-}"
    ,"setByteArrayGeneric :: (Prim a, PrimMonad m) => MutableByteArray (PrimState m) -> Int -> Int -> a -> m ()"
    ,"setByteArrayGeneric mba off n v | n <= 0 = return ()"
    ,"                                | otherwise = do"
    ,"    writeByteArray mba off v"
    ,"    setByteArrayGeneric mba (off + 1) (n - 1) v"
    ,""
    ,"{-# INLINE setOffAddrGeneric #-}"
    ,"setOffAddrGeneric :: (Prim a, PrimMonad m) => Addr -> Int -> Int -> a -> m ()"
    ,"setOffAddrGeneric addr off n v | n <= 0 = return ()"
    ,"                               | otherwise = do"
    ,"    writeOffAddr addr off v"
    ,"    setOffAddrGeneric addr (off + 1) (n - 1) v"
    ,""
    ] ++ patSyns ++ rules
    where
        patSyns = if genPatSyns /= NoPats then map (genPatSynonym (genPatSyns == Pats)) [2, 4, 8, 16, 32, 64] else []
        rules = if doRules then map mkRule (filter isRealPrimitiveType allPrimitiveTypes) ++ [""] else []
        mkRule td = let
            p = getPrimName td
            in "{-# RULES \"unpack/pack " ++ p ++ "\" forall x . unpack" ++ p ++ " (pack" ++ p ++ " x) = x #-}\n" ++
               "{-# RULES \"pack/unpack " ++ p ++ "\" forall x . pack" ++ p ++ " (unpack" ++ p ++ " x) = x #-}"
        isRealPrimitiveType td = getPrimName td /= "DoubleX16#"

exposedFile :: PatsMode -> Int -> String
exposedFile genPatSyns maxCapability = unlines $
    ["-----------------------------------------------------------------------------"
    ,"-- |"
    ,"-- Module      :  Data.Primitive.SIMD"
    ,"-- Copyright   :  (c) 2015 - 2017 Anselm Jonas Scholl"
    ,"-- License     :  BSD3"
    ,"-- "
    ,"-- Maintainer  :  anselm.scholl@tu-harburg.de"
    ,"-- Stability   :  experimental"
    ,"-- Portability :  non-portable (uses GHC.Prim)"
    ,"--"
    ,"-- SIMD data types and functions."
    ,"--"
    ,"-----------------------------------------------------------------------------"
    ,if genPatSyns /= NoPats then "{-# LANGUAGE PatternSynonyms #-}" else ""
    ,"module Data.Primitive.SIMD ("
    ,"     -- * SIMD type classes"
    ,"     SIMDVector(..)"
    ,"    ,SIMDIntVector(..)"
    ,"     -- * SIMD data types"
    ] ++ tuple64 ++ types ++ patSyns ++
    ["     -- * Build information"
    ,"    ,VectorExtension(..)"
    ,"    ,getVectorExtensionSize"
    ,"    ,buildWithSSE"
    ,"    ) where"
    ,""
    ,"-- This code was AUTOMATICALLY generated, DO NOT EDIT!"
    ,""
    ,"import Data.Primitive.SIMD.Class"
    ] ++ imports ++
    [""
    ,"data VectorExtension = SSE128 | SSE256 | SSE512 deriving (Show, Eq, Ord, Enum)"
    ,""
    ,"-- | Get the number of bits usable with a vector extension."
    ,"getVectorExtensionSize :: VectorExtension -> Int"
    ,"getVectorExtensionSize SSE128 = 128"
    ,"getVectorExtensionSize SSE256 = 256"
    ,"getVectorExtensionSize SSE512 = 512"
    ,""
    ,"-- | If this library was build with vector instruction support,"
    ,"--   this will contain the enabled vector extension."
    ,"buildWithSSE :: Maybe VectorExtension"
    ,"buildWithSSE = " ++ vectorExtension
    ]
    where
        vectorExtension = case maxCapability * 8 of
            0 -> "Nothing"
            n -> "Just SSE" ++ show n
        tuple64 = if maxTupleSize < 64 then ["    ,Tuple64(..)"] else []
        types = map (\ td -> "    ," ++ getDataName td) allPrimitiveTypes
        patSyns = if genPatSyns /= NoPats then ["    ,pattern Vec" ++ show (n :: Int) | n <- [2, 4, 8, 16, 32, 64]] else []
        imports = map (\ td -> "import Data.Primitive.SIMD." ++ getDataName td) allPrimitiveTypes

fileHeader :: TypeDesc -> String
fileHeader td = unlines $
    ["{-# LANGUAGE UnboxedTuples         #-}"
    ,"{-# LANGUAGE MagicHash             #-}"
    ,"{-# LANGUAGE TypeFamilies          #-}"
    ,"{-# LANGUAGE DeriveDataTypeable    #-}"
    ,"{-# LANGUAGE BangPatterns          #-}"
    ,"{-# LANGUAGE MultiParamTypeClasses #-}"
    ,"{-# LANGUAGE CPP                   #-}"
    ,if versionBranch compilerVersion >= [8, 0] then "{-# OPTIONS_GHC -Wno-inline-rule-shadowing #-}" else ""
    ,""
    ,if is64 td then "#include \"MachDeps.h\"" else ""
    ,""
    ,"module Data.Primitive.SIMD." ++ getDataName td ++ " (" ++ getDataName td ++ ") where"
    ,""
    ,"-- This code was AUTOMATICALLY generated, DO NOT EDIT!"
    ,""
    ,if versionBranch compilerVersion < [7, 10] then "import Control.Monad" else ""
    ,"import Prelude"
    ,""
    ,"import Data.Primitive.SIMD.Class"
    ,""
    ,if isInt td then "import GHC.Int" else ""
    ,if isWord td then "import GHC.Word" else ""
    ,"import GHC.Exts"
    ,"import GHC.ST"
    ,""
    ,"import Foreign.Storable (Storable)"
    ,"import qualified Foreign.Storable"
    ,""
    ,"import Control.Monad.Primitive"
    ,""
    ,"import Data.Primitive.Types"
    ,"import Data.Primitive.ByteArray"
    ,"import Data.Primitive.Addr"
    ,if versionBranch compilerVersion < [8, 4] then "import Data.Monoid" else ""
    ,"import Data.Typeable"
    ,""
    ,"import qualified Data.Vector.Primitive as PV"
    ,"import qualified Data.Vector.Primitive.Mutable as PMV"
    ,"import Data.Vector.Unboxed (Unbox)"
    ,"import qualified Data.Vector.Unboxed as UV"
    ,"import Data.Vector.Generic (Vector(..))"
    ,"import Data.Vector.Generic.Mutable (MVector(..))"
    ,""
    ] ++ (if is64 td then
        ["#if WORD_SIZE_IN_BITS == 64"
        ,if isInt td then "type RealInt64# = Int#" else "type RealWord64# = Word#"
        ,"#elif WORD_SIZE_IN_BITS == 32"
        ,if isInt td then "type RealInt64# = Int64#" else "type RealWord64# = Word64#"
        ,"#else"
        ,"#error \"WORD_SIZE_IN_BITS is neither 64 or 32\""
        ,"#endif"
        ,""
        ] else [])

replaceTabs :: String -> String
replaceTabs = concatMap helper
    where
        helper '\t' = "    "
        helper c    = [c]

generateCode :: Int -> TypeDesc -> String
generateCode maxCapability td = replaceTabs $ fileHeader td ++ generateTypeCode maxCapability td

replaceX1 :: String -> String
replaceX1 ('X':'1':x:xs) | not (isDigit x) = replaceX1 (x:xs)
replaceX1 (x:xs) = x : replaceX1 xs
replaceX1 []     = []

groupLines :: String -> String
groupLines ('\n':'\n':'\n':x:xs) = groupLines ('\n':'\n':x:xs)
groupLines (x:xs) = x : groupLines xs
groupLines []     = []

genCode :: FilePath -> PatsMode -> Int -> IO ()
genCode fp genPatSyns maxCapability = do
    createDirectoryIfMissing True fp
    writeFile (fp ++ "/Class.hs") $ classFile genPatSyns (maxCapability /= 0)
    writeFile (fp ++ ".hs") $ exposedFile genPatSyns maxCapability
    forM_ allPrimitiveTypes $ \ td ->
        writeFile (fp ++ "/" ++ getDataName td ++ ".hs") (groupLines $ replaceX1 $ generateCode maxCapability td)
