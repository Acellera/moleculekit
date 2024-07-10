##
# File: BinaryCifReader.py
# Date: 15-May-2021  jdw
#
#  Reader methods and decoders for binary CIF.
#
#  Updates:
##

# Comments by Stefan: bCIF decoder taken from py-mmcif repository since it won't compile on Windows
# Changed the imports to use pdbx instead of mmcif which is available here

from collections import OrderedDict
import gzip
import io
import logging
import struct
from contextlib import closing

import msgpack
import requests
from moleculekit.pdbx.reader.PdbxContainers import DataContainer, DataCategory

try:
    from urllib.parse import urlsplit
except Exception:
    from urlparse import urlsplit

logger = logging.getLogger(__name__)


class BinaryCifReader(object):
    """Reader methods for the binary CIF format."""

    def __init__(self, storeStringsAsBytes=False, defaultStringEncoding="utf-8"):
        """Create an instance of the binary CIF reader class.

        Args:
            storeStringsAsBytes (bool, optional): strings are stored as lists of bytes. Defaults to False.
            defaultStringEncoding (str, optional): default encoding for string data. Defaults to "utf-8".
        """
        self.__storeStringsAsBytes = storeStringsAsBytes
        self.__defaultStringEncoding = defaultStringEncoding

    def deserialize(self, locator, timeout=None):
        """Deserialize the input binary CIF file stored in the file/URL locator path.

        Args:
            locator (str): input file path or URL
            timeout (float): timeout for fetching a remote url

        Returns:
            list: list DataContainer objects
        """
        import os

        cL = []
        try:
            if os.path.exists(locator):  # FIXED BY STEFAN: if self.__isLocal(locator):
                with (
                    gzip.open(locator, mode="rb")
                    if locator[-3:] == ".gz"
                    else open(locator, "rb")
                ) as fh:
                    cL = self.__deserialize(
                        fh, storeStringsAsBytes=self.__storeStringsAsBytes
                    )
            else:
                if locator.endswith(".gz"):
                    customHeader = {"Accept-Encoding": "gzip"}
                    with closing(
                        requests.get(locator, headers=customHeader, timeout=timeout)
                    ) as fh:
                        ufh = gzip.GzipFile(fileobj=io.BytesIO(fh.content))
                        cL = self.__deserialize(
                            ufh, storeStringsAsBytes=self.__storeStringsAsBytes
                        )
                else:
                    with closing(requests.get(locator, timeout=timeout)) as fh:
                        cL = self.__deserialize(
                            io.BytesIO(fh.content),
                            storeStringsAsBytes=self.__storeStringsAsBytes,
                        )

        except Exception as e:
            logger.exception("Failing with %s", str(e))
        return cL

    def __deserialize(self, fh, storeStringsAsBytes=False):
        cL = []
        try:
            dec = BinaryCifDecoders(storeStringsAsBytes=storeStringsAsBytes)
            bD = msgpack.unpack(fh)
            #
            logger.debug("bD.keys() %r", bD.keys())
            logger.debug("bD['dataBlocks'] %s", bD[self.__toBytes("dataBlocks")])
            #
            for dataBlock in bD[self.__toBytes("dataBlocks")]:
                header = (
                    self.__fromBytes(dataBlock[self.__toBytes("header")])
                    if self.__toBytes("header") in dataBlock
                    else None
                )
                logger.debug("header %r", header)
                logger.debug("dataBlock %r", dataBlock)
                #
                dc = DataContainer(header)
                categoryList = (
                    dataBlock[self.__toBytes("categories")]
                    if self.__toBytes("categories") in dataBlock
                    else []
                )
                for category in categoryList:
                    catName = self.__fromBytes(category[self.__toBytes("name")])[1:]
                    colList = category[self.__toBytes("columns")]
                    logger.debug("catName %r columns %r", catName, colList)
                    colD = OrderedDict()
                    atNameList = []
                    for col in colList:
                        logger.debug("col.keys() %r", col.keys())
                        atName = self.__fromBytes(col[self.__toBytes("name")])
                        atData = col[self.__toBytes("data")]
                        logger.debug(
                            "atData encoding (%d) data (%d)",
                            len(atData[self.__toBytes("encoding")]),
                            len(atData[self.__toBytes("data")]),
                        )
                        atMask = col[self.__toBytes("mask")]
                        logger.debug("catName %r atName %r", catName, atName)
                        logger.debug(
                            " >atData.data    %r", atData[self.__toBytes("data")]
                        )
                        logger.debug(
                            " >atData.encoding (%d) %r",
                            len(atData[self.__toBytes("encoding")]),
                            atData[self.__toBytes("encoding")],
                        )
                        logger.debug(" >mask %r", atMask)
                        tVal = dec.decode(
                            col[self.__toBytes("data")][self.__toBytes("data")],
                            col[self.__toBytes("data")][self.__toBytes("encoding")],
                        )
                        if col[self.__toBytes("mask")]:
                            mVal = dec.decode(
                                col[self.__toBytes("mask")][self.__toBytes("data")],
                                col[self.__toBytes("mask")][self.__toBytes("encoding")],
                            )
                            tVal = [
                                "?" if m == 2 else "." if m == 1 else d
                                for d, m in zip(tVal, mVal)
                            ]
                        colD[atName] = tVal
                        atNameList.append(atName)
                    #
                    cObj = DataCategory(catName, attributeNameList=atNameList)
                    genL = [colGen for colGen in colD.values()]
                    for rowTup in zip(*genL):
                        row = list(rowTup)
                        logger.debug("row %r", row)
                        cObj.append(row)
                    #
                    dc.append(cObj)
                cL.append(dc)
        except Exception as e:
            logger.exception("Failing with %s", str(e))
        return cL

    def __isLocal(self, locator):
        """Returns true if input string can be interpreted as a local file path.

        Args:
            locator (str): url or path string

        Returns:
            bool: True if locator is a local path
        """
        try:
            locSp = urlsplit(locator)
            return locSp.scheme in ["", "file"]
        except Exception as e:
            logger.exception("For locator %r failing with %s", locator, str(e))
        return None

    def __toBytes(self, strVal):
        """Optional conversion of the input string to bytes according to the class setting (storeStringsAsBytes).

        Args:
            strVal (string): input string

        Returns:
            string or bytes: optionally converted string.
        """
        try:
            return (
                strVal.encode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else strVal
            )
        except (UnicodeDecodeError, AttributeError):
            logger.exception("Bad type for %r", strVal)
            return strVal

    def __fromBytes(self, byteVal):
        """Optional conversion of the input value according to the class setting (storeStringsAsBytes).

        Args:
            byteVal (string): input byte object

        Returns:
            string: optionally converted input value
        """
        try:
            return (
                byteVal.decode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else byteVal
            )
        except (UnicodeDecodeError, AttributeError):
            logger.exception("Bad type for %r", byteVal)
            return byteVal


class BinaryCifDecoders(object):
    """Column oriented Binary CIF decoders implementing
    StringArray, ByteArray, IntegerPacking, Delta, RunLength,
    FixedPoint, and  IntervalQuantization from the BinaryCIF
    specification described in:

    Sehnal D, Bittrich S, Velankar S, Koca J, Svobodova R, Burley SK, Rose AS.
    BinaryCIF and CIFTools-Lightweight, efficient and extensible macromolecular data management.
    PLoS Comput Biol. 2020 Oct 19;16(10):e1008247.
    doi: 10.1371/journal.pcbi.1008247. PMID: 33075050; PMCID: PMC7595629.

    and in the specification at https://github.com/molstar/BinaryCIF/blob/master/encoding.md

    and from the I/HM Python implementation at https://github.com/ihmwg/python-ihm[summary]

    """

    bCifCodeTypeD = {
        1: "integer_8",
        2: "integer_16",
        3: "integer_32",
        4: "unsigned_integer_8",
        5: "unsigned_integer_16",
        6: "unsigned_integer_32",
        32: "float_32",
        33: "float_64",
    }
    """Binary CIF protocol internal data type codes to integer and float types
    """
    bCifTypeD = {
        "integer_8": {"struct_format_code": "b", "min": -0x7F - 1, "max": 0x7F},
        "integer_16": {"struct_format_code": "h", "min": -0x7FFF - 1, "max": 0x7FFF},
        "integer_32": {
            "struct_format_code": "i",
            "min": -0x7FFFFFFF - 1,
            "max": 0x7FFFFFFF,
        },
        "unsigned_integer_8": {"struct_format_code": "B", "min": 0, "max": 0xFF},
        "unsigned_integer_16": {"struct_format_code": "H", "min": 0, "max": 0xFFFF},
        "unsigned_integer_32": {"struct_format_code": "I", "min": 0, "max": 0xFFFFFFFF},
        "float_32": {
            "struct_format_code": "f",
            "min": 1.175494351e-38,
            "max": 3.402823466e38,
        },
        "float_64": {
            "struct_format_code": "d",
            "min": 2.2250738585072014e-308,
            "max": 1.7976931348623158e308,
        },
    }
    """Binary CIF data type feature dictionary
    """

    def __init__(
        self, storeStringsAsBytes=False, defaultStringEncoding="utf-8", verbose=False
    ):
        """Create an instance of the binary CIF encoder class.

        Args:
            storeStringsAsBytes (bool, optional): express keys and strings as byte types otherwise follow the default encoding. Defaults to False.
            defaultStringEncoding (str, optional): default encoding for string types. Defaults to "utf-8".
            verbose(bool, optional): provide tracking of type conversion issues. Defaults to False.
        """
        self.__storeStringsAsBytes = storeStringsAsBytes
        self.__defaultStringEncoding = defaultStringEncoding
        self.__verbose = verbose
        #
        self.__encodersMethodD = {
            "StringArray": self.stringArrayDecoder,
            "ByteArray": self.byteArrayDecoder,
            "IntegerPacking": self.integerPackingDecoder,
            "Delta": self.deltaDecoder,
            "RunLength": self.runLengthDecoder,
            "FixedPoint": self.fixedPointDecoder,
            "IntervalQuantization": self.intervalQuantizationDecoder,
        }

    def decode(self, colDataList, encodingDictList):
        """Return the decoded input data column using the input list of encodings

        Args:
            colDataList (list): column of data to be decoded
            encodingDictList (list): list of dictionary holding binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of column data
        """
        for encoding in reversed(encodingDictList):
            encType = self.__fromBytes(encoding[self.__toBytes("kind")])
            colDataList = self.__encodersMethodD[encType](colDataList, encoding)
        return colDataList

    def stringArrayDecoder(self, colDataList, encodingDict):
        """Decode an array of strings stored as a concatenation of all unique
        strings, a list of offsets to construct the unique substrings, and indices into
        the offset array.

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of string data
        """
        offsetList = list(
            self.decode(
                encodingDict[self.__toBytes("offsets")],
                encodingDict[self.__toBytes("offsetEncoding")],
            )
        )
        lookupIndexIt = self.decode(
            colDataList, encodingDict[self.__toBytes("dataEncoding")]
        )

        stringData = self.__fromBytes(encodingDict[self.__toBytes("stringData")])
        uniqueStringList = []
        for iBegin, iEnd in zip(offsetList, offsetList[1:]):
            uniqueStringList.append(stringData[iBegin:iEnd])
            logger.debug("iBegin %d iEnd %d %r ", iBegin, iEnd, stringData[iBegin:iEnd])

        for ii in lookupIndexIt:
            yield uniqueStringList[ii] if ii >= 0 else None

    def byteArrayDecoder(self, colDataList, encodingDict):
        """Decode input byte list into a list of integers/floats

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of integer/float data
        """
        structKey = self.bCifCodeTypeD[encodingDict[self.__toBytes("type")]]
        structFormatCode = self.bCifTypeD[structKey]["struct_format_code"]
        count = len(colDataList) // struct.calcsize(structFormatCode)
        # struct.unpack() format string for little-endian  = < format_string code * counts
        return struct.unpack("<" + structFormatCode * count, colDataList)

    def __unsignedDecode(self, colDataList, encodingDict):
        upperLimit = (
            self.bCifTypeD["unsigned_integer_8"]["max"]
            if encodingDict[self.__toBytes("byteCount")] == 1
            else self.bCifTypeD["unsigned_integer_16"]["max"]
        )
        ii = 0
        while ii < len(colDataList):
            value = 0
            tVal = colDataList[ii]
            while tVal == upperLimit:
                value += tVal
                ii += 1
                tVal = colDataList[ii]
            yield value + tVal
            ii += 1

    def __signedDecode(self, colDataList, encodingDict):
        upperLimit = (
            self.bCifTypeD["integer_8"]["max"]
            if encodingDict[self.__toBytes("byteCount")] == 1
            else self.bCifTypeD["integer_16"]["max"]
        )
        lowerLimit = (
            self.bCifTypeD["integer_8"]["min"]
            if encodingDict[self.__toBytes("byteCount")] == 1
            else self.bCifTypeD["integer_16"]["min"]
        )
        ii = 0
        while ii < len(colDataList):
            value = 0
            tVal = colDataList[ii]
            while tVal == upperLimit or tVal == lowerLimit:
                value += tVal
                ii += 1
                tVal = colDataList[ii]
            yield value + tVal
            ii += 1

    def integerPackingDecoder(self, colDataList, encodingDict):
        """Decode a (32-bit) integer list packed into 8- or 16-bit values.

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of integer data

        """
        if encodingDict[self.__toBytes("isUnsigned")]:
            return self.__unsignedDecode(colDataList, encodingDict)
        else:
            return self.__signedDecode(colDataList, encodingDict)

    def deltaDecoder(self, colDataList, encodingDict):
        """Decode an integer list stored as a list of consecutive differences.

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of integer data
        """
        val = encodingDict[self.__toBytes("origin")]
        for diff in colDataList:
            val += diff
            yield val

    def runLengthDecoder(self, colDataList, encodingDict):
        """Decode an integer list stored as pairs of (value, number of repeats).

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of integer data
        """
        _ = encodingDict
        colDataList = list(colDataList)
        for ii in range(0, len(colDataList), 2):
            for _ in range(colDataList[ii + 1]):
                yield colDataList[ii]

    def fixedPointDecoder(self, colDataList, encodingDict):
        """Decode a floating point list stored as integers and a multiplicative factor.

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of float data
        """
        factor = float(encodingDict[self.__toBytes("factor")])
        for val in colDataList:
            yield float(val) / factor

    def intervalQuantizationDecoder(self, colDataList, encodingDict):
        """Decode a list of 32-bit integers quantized within a given interval into a list of floats.

        Args:
            colDataList (list): column of data to be decoded
            encodingDict (dict): dictionary of binary CIF encoding details
                                 elements described in the specification at
                                 https://github.com/molstar/BinaryCIF/blob/master/encoding.md

        Yields:
            list: decoded list of float data

        """
        delta = float(
            encodingDict[self.__toBytes("max")] - encodingDict[self.__toBytes("min")]
        ) / float(encodingDict[self.__toBytes("numSteps")] - 1.0)
        minVal = encodingDict[self.__toBytes("min")]
        for val in colDataList:
            yield minVal + delta * val

    def __toBytes(self, strVal):
        """Optional conversion of the input string to bytes according to the class setting (storeStringsAsBytes).

        Args:
            strVal (string): input string

        Returns:
            string or bytes: optionally converted string.
        """
        try:
            return (
                strVal.encode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else strVal
            )
        except (UnicodeDecodeError, AttributeError):
            if self.__verbose:
                logger.exception("Bad type for %r", strVal)
            return strVal

    def __fromBytes(self, byteVal):
        """Optional conversion of the input value according to the class setting (storeStringsAsBytes).

        Args:
            byteVal (string): input byte object

        Returns:
            string: optionally converted input value
        """
        try:
            return (
                byteVal.decode(self.__defaultStringEncoding)
                if self.__storeStringsAsBytes
                else byteVal
            )
        except (UnicodeDecodeError, AttributeError):
            if self.__verbose:
                logger.exception("Bad type for %r", byteVal)
            return byteVal
