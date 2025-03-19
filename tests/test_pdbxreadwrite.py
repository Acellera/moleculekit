import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _testSimpleInitialization(tmp_path):
    """Test case -  Simple initialization of a data category and data block"""
    #
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter
    from moleculekit.pdbx.reader.PdbxContainers import DataContainer, DataCategory

    attributeNameList = [
        "aOne",
        "aTwo",
        "aThree",
        "aFour",
        "aFive",
        "aSix",
        "aSeven",
        "aEight",
        "aNine",
        "aTen",
    ]
    rowList = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]
    nameCat = "myCategory"
    #
    #
    curContainer = DataContainer("myblock")
    aCat = DataCategory(nameCat, attributeNameList, rowList)
    aCat.printIt()
    curContainer.append(aCat)
    curContainer.printIt()
    #
    myContainerList = []
    myContainerList.append(curContainer)
    ofh = open(os.path.join(tmp_path, "test.cif"), "w")
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myContainerList)
    ofh.close()

    myContainerList = []
    ifh = open(os.path.join(tmp_path, "test.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myContainerList)
    ifh.close()
    for container in myContainerList:
        for objName in container.getObjNameList():
            name, aList, rList = container.getObj(objName).get()
            print("Recovered data category  %s\n" % name)
            print("Attribute list           %r\n" % repr(aList))
            print("Row list                 %r\n" % repr(rList))


def _testWriteDataFile(tmp_path):
    """Test case -  write data file"""
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter
    from moleculekit.pdbx.reader.PdbxContainers import DataContainer, DataCategory

    myDataList = []
    ofh = open(os.path.join(tmp_path, "test.cif"), "w")
    curContainer = DataContainer("myblock")
    aCat = DataCategory("pdbx_seqtool_mapping_ref")
    aCat.appendAttribute("ordinal")
    aCat.appendAttribute("entity_id")
    aCat.appendAttribute("auth_mon_id")
    aCat.appendAttribute("auth_mon_num")
    aCat.appendAttribute("pdb_chain_id")
    aCat.appendAttribute("ref_mon_id")
    aCat.appendAttribute("ref_mon_num")
    aCat.append([1, 2, 3, 4, 5, 6, 7])
    aCat.append([1, 2, 3, 4, 5, 6, 7])
    aCat.append([1, 2, 3, 4, 5, 6, 7])
    aCat.append([1, 2, 3, 4, 5, 6, 7])
    aCat.append([7, 6, 5, 4, 3, 2, 1])
    aCat.printIt()
    curContainer.append(aCat)
    curContainer.printIt()
    #
    myDataList.append(curContainer)
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myDataList)
    ofh.close()


def _testUpdateDataFile(tmp_path):
    """Test case -  update data file"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter
    from moleculekit.pdbx.reader.PdbxContainers import DataContainer, DataCategory

    # Create a initial data file --
    #
    myDataList = []

    curContainer = DataContainer("myblock")
    aCat = DataCategory("pdbx_seqtool_mapping_ref")
    aCat.appendAttribute("ordinal")
    aCat.appendAttribute("entity_id")
    aCat.appendAttribute("auth_mon_id")
    aCat.appendAttribute("auth_mon_num")
    aCat.appendAttribute("pdb_chain_id")
    aCat.appendAttribute("ref_mon_id")
    aCat.appendAttribute("ref_mon_num")
    aCat.append([9, 2, 3, 4, 5, 6, 7])
    aCat.append([10, 2, 3, 4, 5, 6, 7])
    aCat.append([11, 2, 3, 4, 5, 6, 7])
    aCat.append([12, 2, 3, 4, 5, 6, 7])

    # print("Assigned data category state-----------------\n")
    # aCat.dumpIt(fh=self.lfh)
    curContainer.append(aCat)
    myDataList.append(curContainer)
    ofh = open(os.path.join(tmp_path, "test.cif"), "w")
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myDataList)
    ofh.close()
    #
    #
    # Read and update the data -
    #
    myDataList = []
    ifh = open(os.path.join(tmp_path, "test.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()
    #
    myBlock = myDataList[0]
    myBlock.printIt()
    myCat = myBlock.getObj("pdbx_seqtool_mapping_ref")
    myCat.printIt()
    for iRow in range(0, myCat.getRowCount()):
        myCat.setValue("some value", "ref_mon_id", iRow)
        myCat.setValue(100, "ref_mon_num", iRow)
    ofh = open(os.path.join(tmp_path, "test2.cif"), "w")
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myDataList)
    ofh.close()


def _testReadDataFile():
    """Test case -  read data file"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader

    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()


def _testReadWriteDataFile(tmp_path):
    """Test case -  data file read write test"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter

    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()

    ofh = open(os.path.join(tmp_path, "test_output.cif"), "w")
    pWr = PdbxWriter(ofh)
    pWr.write(myDataList)
    ofh.close()
