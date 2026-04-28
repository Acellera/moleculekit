##
# File: BinaryCifWriter.py
#
# Vendored and slimmed down from py-mmcif (rcsb/py-mmcif, BinaryCifWriter.py)
# - Drops the DictionaryApi / DataCategoryTyped dependency: column data types
#   are supplied explicitly via a `typeHints` dict instead of being looked up
#   in a CIF dictionary. This avoids shipping the multi-megabyte mmcif_pdbx
#   dictionary with the package.
# - Drops the deprecated (pre-typed) encoder methods.
##

import logging
import struct

import msgpack

from moleculekit.pdbx.reader.BinaryCifReader import BinaryCifDecoders

logger = logging.getLogger(__name__)


class BinaryCifWriter(object):
    """Writer for the binary CIF format.

    Type hints are supplied explicitly:

        typeHints = {
            "atom_site": {"id": "integer", "Cartn_x": "float", ...},
            ...
        }

    Attributes not present in `typeHints[category]` default to "string".
    """

    def __init__(
        self,
        typeHints=None,
        storeStringsAsBytes=False,
        defaultStringEncoding="utf-8",
        useFloat64=False,
    ):
        self.__version = "0.3.0"
        self.__typeHints = typeHints or {}
        self.__storeStringsAsBytes = storeStringsAsBytes
        self.__defaultStringEncoding = defaultStringEncoding
        self.__useFloat64 = useFloat64

    def serialize(self, filePath, containerList):
        blocks = []
        for container in containerList:
            block = {
                self.__toBytes("header"): self.__toBytes(container.getName()),
                self.__toBytes("categories"): [],
            }
            categories = block[self.__toBytes("categories")]
            blocks.append(block)
            for catName in container.getObjNameList():
                cObj = container.getObj(catName)
                rowList = cObj.getRowList()
                rowCount = len(rowList)
                cols = []
                catHints = self.__typeHints.get(catName, {})
                for ii, atName in enumerate(cObj.getAttributeList()):
                    colDataList = [row[ii] for row in rowList]
                    dataType = catHints.get(atName, "string")
                    colMaskDict, encodedColDataList, encodingDictL = (
                        self.__encodeColumnData(colDataList, dataType)
                    )
                    cols.append(
                        {
                            self.__toBytes("name"): self.__toBytes(atName),
                            self.__toBytes("mask"): colMaskDict,
                            self.__toBytes("data"): {
                                self.__toBytes("data"): encodedColDataList,
                                self.__toBytes("encoding"): encodingDictL,
                            },
                        }
                    )
                categories.append(
                    {
                        self.__toBytes("name"): self.__toBytes("_" + catName),
                        self.__toBytes("columns"): cols,
                        self.__toBytes("rowCount"): rowCount,
                    }
                )
        data = {
            self.__toBytes("version"): self.__toBytes(self.__version),
            self.__toBytes("encoder"): self.__toBytes("moleculekit"),
            self.__toBytes("dataBlocks"): blocks,
        }
        with open(filePath, "wb") as ofh:
            msgpack.pack(data, ofh)
        return True

    def __encodeColumnData(self, colDataList, dataType):
        # "float64" overrides the writer-wide useFloat64 default for this column
        # only; used for fields where 32-bit precision isn't enough to round-trip
        # (e.g. cell dimensions like 54.89).
        useFloat64 = self.__useFloat64 or dataType == "float64"
        enc = BinaryCifEncoders(
            defaultStringEncoding=self.__defaultStringEncoding,
            storeStringsAsBytes=self.__storeStringsAsBytes,
            useFloat64=useFloat64,
        )
        typeEncoderD = {
            "string": "StringArrayMasked",
            "integer": "IntArrayMasked",
            "float": "FloatArrayMasked",
            "float64": "FloatArrayMasked",
        }
        colMaskList = enc.getMask(colDataList)
        dataEncType = typeEncoderD[dataType]
        colDataEncoded, colDataEncodingDictL = enc.encodeWithMask(
            colDataList, colMaskList, dataEncType
        )
        colMaskDict = None
        if colMaskList:
            # Mol* indicates that masks should be encoded as if uint_8
            colMaskListTyped = TypedArray(colMaskList, "unsigned_integer_8")
            maskEncoded, maskEncodingDictL = enc.encode(
                colMaskListTyped, ["RunLength", "ByteArray"], "integer"
            )
            colMaskDict = {
                self.__toBytes("data"): maskEncoded.data,
                self.__toBytes("encoding"): maskEncodingDictL,
            }
        return colMaskDict, colDataEncoded, colDataEncodingDictL

    def __toBytes(self, strVal):
        try:
            return (
                strVal.encode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else strVal
            )
        except (UnicodeDecodeError, AttributeError):
            logger.exception("Bad type for %r", strVal)
            return strVal


class TypedArray(object):
    """Small wrapper that pairs a data list with its declared bCIF type code."""

    __slots__ = ["dtype", "data"]

    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype

    def __repr__(self):
        return "<typed_array type %s data %s>" % (self.dtype, self.data)


class BinaryCifEncoders(object):
    """Column-oriented Binary CIF encoders implementing
    StringArray, ByteArray, IntegerPacking, Delta, RunLength,
    and FixedPoint encoders from the BinaryCIF specification described in
    https://github.com/molstar/BinaryCIF/blob/master/encoding.md
    """

    def __init__(
        self,
        defaultStringEncoding="utf-8",
        storeStringsAsBytes=False,
        useFloat64=False,
    ):
        self.__unknown = [".", "?"]
        self.__defaultStringEncoding = defaultStringEncoding
        self.__storeStringsAsBytes = storeStringsAsBytes
        self.__useFloat64 = useFloat64
        self.__bCifTypeCodeD = {v: k for k, v in BinaryCifDecoders.bCifCodeTypeD.items()}

    def encode(self, colDataList, encodingTypeList, dataType):
        encodingDictL = []
        legacy = False
        if isinstance(colDataList, list):
            colDataList = TypedArray(colDataList)
            legacy = True
        for encType in encodingTypeList:
            encDict = None
            if encType == "ByteArray":
                colDataList, encDict = self.byteArrayEncoderTyped(colDataList, dataType)
            elif encType == "Delta":
                colDataList, encDict = self.deltaEncoderTyped(colDataList)
            elif encType == "RunLength":
                colDataList, encDict = self.runLengthEncoderTyped(colDataList)
            elif encType == "IntegerPacking":
                colDataList, encDict = self.integerPackingEncoderTyped(colDataList)
            else:
                logger.info("unsupported encoding %r", encType)
            if encDict is not None:
                encodingDictL.append(encDict)
        if legacy:
            return colDataList.data, encodingDictL
        return colDataList, encodingDictL

    def encodeWithMask(self, colDataList, colMaskList, encodingType):
        if encodingType == "StringArrayMasked":
            return self.stringArrayMaskedEncoder(colDataList, colMaskList)
        if encodingType == "IntArrayMasked":
            return self.intArrayMaskedEncoder(colDataList, colMaskList)
        if encodingType == "FloatArrayMasked":
            return self.floatArrayMaskedEncoder(colDataList, colMaskList)
        logger.info("unsupported masked encoding %r", encodingType)
        return [], []

    def stringArrayMaskedEncoder(self, colDataList, colMaskList):
        integerEncoderList = ["Delta", "RunLength", "IntegerPacking", "ByteArray"]
        uniqStringIndex = {}
        uniqStringList = []
        indexList = []
        for i, strVal in enumerate(colDataList):
            if colMaskList is not None and colMaskList[i]:
                indexList.append(-1)
            else:
                tS = str(strVal)
                if tS not in uniqStringIndex:
                    uniqStringIndex[tS] = len(uniqStringIndex)
                    uniqStringList.append(tS)
                indexList.append(uniqStringIndex[tS])
        offsetList = [0]
        runningLen = 0
        for tS in uniqStringList:
            runningLen += len(tS)
            offsetList.append(runningLen)

        encodedOffsetList, offsetEncodingDictL = self.encode(
            offsetList, integerEncoderList, "integer"
        )
        encodedIndexList, indexEncodingDictL = self.encode(
            indexList, integerEncoderList, "integer"
        )
        encodingDict = {
            self.__toBytes("kind"): self.__toBytes("StringArray"),
            self.__toBytes("dataEncoding"): indexEncodingDictL,
            self.__toBytes("stringData"): self.__toBytes("".join(uniqStringList)),
            self.__toBytes("offsetEncoding"): offsetEncodingDictL,
            self.__toBytes("offsets"): encodedOffsetList,
        }
        return encodedIndexList, [encodingDict]

    def intArrayMaskedEncoder(self, colDataList, colMaskList):
        integerEncoderList = ["Delta", "RunLength", "IntegerPacking", "ByteArray"]
        if colMaskList:
            # Per spec, masked (missing) integer cells are encoded as 0.
            maskedColDataList = [
                0 if m else int(d) for m, d in zip(colMaskList, colDataList)
            ]
        else:
            maskedColDataList = [int(d) for d in colDataList]
        return self.encode(maskedColDataList, integerEncoderList, "integer")

    def floatArrayMaskedEncoder(self, colDataList, colMaskList):
        if colMaskList:
            maskedColDataList = [
                0.0 if m else float(d) for m, d in zip(colMaskList, colDataList)
            ]
        else:
            maskedColDataList = [float(d) for d in colDataList]
        return self.encode(maskedColDataList, ["ByteArray"], "float")

    def byteArrayEncoderTyped(self, colTypedDataList, dataType):
        if dataType == "float":
            byteArrayType = (
                self.__bCifTypeCodeD["float_64"]
                if self.__useFloat64
                else self.__bCifTypeCodeD["float_32"]
            )
        else:
            if colTypedDataList.dtype:
                byteArrayType = self.__bCifTypeCodeD[colTypedDataList.dtype]
            else:
                byteArrayType = self.__getIntegerPackingType(colTypedDataList.data)
        encodingD = {
            self.__toBytes("kind"): self.__toBytes("ByteArray"),
            self.__toBytes("type"): byteArrayType,
        }
        fmt = BinaryCifDecoders.bCifTypeD[
            BinaryCifDecoders.bCifCodeTypeD[byteArrayType]
        ]["struct_format_code"]
        encodedData = struct.pack(
            "<" + fmt * len(colTypedDataList.data), *colTypedDataList.data
        )
        return TypedArray(encodedData), encodingD

    def deltaEncoderTyped(self, colTypedDataList, minLen=40):
        if colTypedDataList.dtype and colTypedDataList.dtype not in (
            "integer_8",
            "integer_16",
            "integer_32",
        ):
            raise TypeError(
                "Only signed integer types can be encoded with delta encoder: %s"
                % colTypedDataList.dtype
            )
        if len(colTypedDataList.data) <= minLen:
            return colTypedDataList, None
        byteArrayType = colTypedDataList.dtype or "integer_32"
        encodingD = {
            self.__toBytes("kind"): self.__toBytes("Delta"),
            self.__toBytes("origin"): colTypedDataList.data[0],
            self.__toBytes("srcType"): self.__bCifTypeCodeD[byteArrayType],
        }
        encoded = [0] + [
            colTypedDataList.data[i] - colTypedDataList.data[i - 1]
            for i in range(1, len(colTypedDataList.data))
        ]
        return TypedArray(encoded, byteArrayType), encodingD

    def runLengthEncoderTyped(self, colTypedDataList, minLen=40):
        if len(colTypedDataList.data) <= minLen:
            return colTypedDataList, None
        srcType = colTypedDataList.dtype or "integer_32"
        encodingD = {
            self.__toBytes("kind"): self.__toBytes("RunLength"),
            self.__toBytes("srcType"): self.__bCifTypeCodeD[srcType],
            self.__toBytes("srcSize"): len(colTypedDataList.data),
        }
        encoded = []
        val = None
        repeat = 1
        for colVal in colTypedDataList.data:
            if colVal != val:
                if val is not None:
                    encoded.extend((val, repeat))
                val = colVal
                repeat = 1
            else:
                repeat += 1
        encoded.extend((val, repeat))
        # Bail out if RLE made the data bigger
        if len(encoded) > len(colTypedDataList.data):
            return colTypedDataList, None
        return TypedArray(encoded, "integer_32"), encodingD

    def integerPackingEncoderTyped(self, colTypedDataList):
        if colTypedDataList.dtype and colTypedDataList.dtype != "integer_32":
            raise TypeError(
                "Only integer-32 can be encoded with packing encoder: %s"
                % colTypedDataList.dtype
            )
        packing = self._determine_packing(colTypedDataList.data)
        nbytes = packing["bytes"]
        isSigned = packing["isSigned"]
        if nbytes == 4:
            return colTypedDataList, None
        encodingD = {
            self.__toBytes("kind"): self.__toBytes("IntegerPacking"),
            self.__toBytes("byteCount"): nbytes,
            self.__toBytes("srcSize"): len(colTypedDataList.data),
            self.__toBytes("isUnsigned"): not isSigned,
        }
        if isSigned:
            upper_limit = 0x7F if nbytes == 1 else 0x7FFF
        else:
            upper_limit = 0xFF if nbytes == 1 else 0xFFFF
        lower_limit = -upper_limit - 1
        encoded = []
        for colVal in colTypedDataList.data:
            if colVal >= 0:
                while colVal >= upper_limit:
                    encoded.append(upper_limit)
                    colVal -= upper_limit
            else:
                while colVal <= lower_limit:
                    encoded.append(lower_limit)
                    colVal -= lower_limit
            encoded.append(colVal)
        if nbytes == 1:
            byteArrayType = "integer_8" if isSigned else "unsigned_integer_8"
        else:
            byteArrayType = "integer_16" if isSigned else "unsigned_integer_16"
        return TypedArray(encoded, byteArrayType), encodingD

    def getMask(self, colDataList):
        """Return per-row mask: 0 = present, 1 = '.', 2 = '?'/None. None if no mask needed."""
        mask = None
        for ii, colVal in enumerate(colDataList):
            if colVal is not None and colVal not in self.__unknown:
                continue
            if not mask:
                mask = [0] * len(colDataList)
            mask[ii] = 2 if colVal is None or colVal == "?" else 1
        return mask

    def __getIntegerPackingType(self, colDataList):
        if not colDataList:
            return self.__bCifTypeCodeD["unsigned_integer_8"]
        try:
            minV = min(colDataList)
            maxV = max(colDataList)
            if minV >= 0:
                for typeName in (
                    "unsigned_integer_8",
                    "unsigned_integer_16",
                    "unsigned_integer_32",
                ):
                    if maxV <= BinaryCifDecoders.bCifTypeD[typeName]["max"]:
                        return self.__bCifTypeCodeD[typeName]
            else:
                for typeName in ("integer_8", "integer_16", "integer_32"):
                    info = BinaryCifDecoders.bCifTypeD[typeName]
                    if minV >= info["min"] and maxV <= info["max"]:
                        return self.__bCifTypeCodeD[typeName]
        except Exception as e:
            logger.exception("Failing with %s", str(e))
        raise TypeError("Cannot determine integer packing type")

    def _determine_packing(self, colDataList):
        """Pick the optimal IntegerPacking byte width.

        IntegerPacking can store values larger than the byte-width's max by emitting
        the max value repeatedly (acts as an "overflow continuation"), so the optimal
        choice is not just min/max — we have to estimate the encoded size.
        """

        def packing_size_signed(data, upper_limit):
            lower_limit = -upper_limit - 1
            size = 0
            for v in data:
                size += int(v / upper_limit) if v >= 0 else int(v / lower_limit)
            return size + len(data)

        def packing_size_unsigned(data, upper_limit):
            size = 0
            for v in data:
                size += int(v / upper_limit)
            return size + len(data)

        if not colDataList:
            # Empty column: pick a no-op packing (4-byte path returns the array as-is)
            return {"size": 0, "bytes": 4, "isSigned": False}
        minV = min(colDataList)
        is_signed = minV < 0
        size8 = (
            packing_size_signed(colDataList, 0x7F)
            if is_signed
            else packing_size_unsigned(colDataList, 0xFF)
        )
        size16 = (
            packing_size_signed(colDataList, 0x7FFF)
            if is_signed
            else packing_size_unsigned(colDataList, 0xFFFF)
        )
        dlen = len(colDataList)
        if dlen * 4 < size16 * 2:
            size, nbytes = dlen, 4
        elif size16 * 2 < size8:
            size, nbytes = size16, 2
        else:
            size, nbytes = size8, 1
        return {"size": size, "bytes": nbytes, "isSigned": is_signed}

    def __toBytes(self, strVal):
        try:
            return (
                strVal.encode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else strVal
            )
        except (UnicodeDecodeError, AttributeError):
            logger.exception("Bad type for %r", strVal)
            return strVal
