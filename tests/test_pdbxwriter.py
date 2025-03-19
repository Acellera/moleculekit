import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _testWriteDataFile(tmp_path):
    """Test case -  write data file"""
    #
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
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    curContainer.append(aCat)
    myDataList.append(curContainer)
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myDataList)
    ofh.close()


def _testUpdateDataFile(tmp_path):
    """Test case -  write data file"""
    # Create a initial data file --
    #
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
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
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    aCat.append((1, 2, 3, 4, 5, 6, 7))
    curContainer.append(aCat)
    myDataList.append(curContainer)
    pdbxW = PdbxWriter(ofh)
    pdbxW.write(myDataList)
    ofh.close()
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

    #
    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()


def _testReadWriteDataFile(tmp_path):
    """Test case -  data file read write test"""
    from moleculekit.pdbx.reader.PdbxReader import PdbxReader
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter

    #
    myDataList = []
    ifh = open(os.path.join(curr_dir, "test_readers", "1kip.cif"), "r")
    pRd = PdbxReader(ifh)
    pRd.read(myDataList)
    ifh.close()

    ofh = open(os.path.join(tmp_path, "test_output.cif"), "w")
    pWr = PdbxWriter(ofh)
    pWr.write(myDataList)
    ofh.close()
