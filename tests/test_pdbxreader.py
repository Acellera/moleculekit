import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _testReadSmallDataFile():
    """Test case -  read data file"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader

    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()


def _testReadBigDataFile():
    """Test case -  read data file"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader

    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1ffk.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()


def _testReadSFDataFile():
    """Test case -  read PDB structure factor data  file and compute statistics on f/sig(f)."""
    #
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader

    myContainerList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip-sf.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myContainerList)
    c0 = myContainerList[0]
    #
    catObj = c0.getObj("refln")
    if catObj is None:
        return False

    nRows = catObj.getRowCount()
    #
    # Get column name index.
    #
    itDict = {}
    itNameList = catObj.getItemNameList()
    for idxIt, itName in enumerate(itNameList):
        itDict[str(itName).lower()] = idxIt
        #
    idf = itDict["_refln.f_meas_au"]
    idsigf = itDict["_refln.f_meas_sigma_au"]
    minR = 100
    maxR = -1
    sumR = 0
    icount = 0
    for row in catObj.getRowList():
        try:
            f = float(row[idf])
            sigf = float(row[idsigf])
            ratio = sigf / f
            maxR = max(maxR, ratio)
            minR = min(minR, ratio)
            sumR += ratio
            icount += 1
        except Exception:
            continue

    ifh.close()
    print(
        "f/sig(f) min %f max %f avg %f count %d\n" % (minR, maxR, sumR / icount, icount)
    )
