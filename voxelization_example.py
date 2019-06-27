from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.home import home
import os

# First let's read our two favourite protein and ligand structures
tut_data = home(dataDir='test-voxeldescriptors')
prot = Molecule(os.path.join(tut_data, '3PTB.pdb'))
slig = SmallMol(os.path.join(tut_data, 'benzamidine.mol2'))

# Atom typing for the protein requires the protein to be protonated and to include the atom bond information.
# Currently the Molecule contains does not contain all bond information as you can see by checking prot.bonds.
# `prepareProteinForAtomtyping` will perform most necessary operations such as removing non-protein atoms,
# adding hydrogens, guessing bonds and guessing protein chains but you can perform those manually as well.
# Take care however as the protonation will a) move atoms to optimize hydrogen networks b) add missing sidechains
# and c) the bond guessing can go wrong if atoms are very close to each other which can happen when adding sidechains.

# If your structure is fully protonated and contains all bond information in prot.bonds skip this step!
prot = prepareProteinForAtomtyping(prot)

# We would suggest visualizing the results with prot.view(guessBonds=False) to show only the bonds already stored in the Molecule. 
prot.view(guessBonds=False)

# Now we can calculate the voxel information for the protein
# By default getVoxelDescriptors will calculate the bounding box of the molecule and grid it into voxels.
# As we don't use point properties but smooth them out over space, we will add 1 A buffer space around the protein 
# for the voxelization grid so that the properties don't cut off at the edge of the grid.
prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, buffer=1)

# Now let's visualize the voxels. This will visualize each voxel channel as a separate VMD molecule so that you can
# inspect each individually. You can play around with the IsoValue in the IsoSurface representation of each channel
# to inspect it in more detail.
prot.view(guessBonds=False)
viewVoxelFeatures(prot_vox, prot_centers, prot_N)

# For the ligand since it's small we could increase the voxel resolution if we so desire to 0.5 A instead of the default 1 A.
lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, voxelsize=0.5, buffer=1)
slig.view(guessBonds=False)
viewVoxelFeatures(lig_vox, lig_centers, lig_N)

# Now let's use this voxel data in pytorch
import torch
prot_vox_t = torch.tensor(prot_vox)
# Reshape to (nsamples, nfeatures, nchannels) which is the usual format for machine learning. 
# In this case it's a single sample so we just add a first dimension to the array.
prot_vox_t = prot_vox_t.unsqueeze(0)

lig_vox_t = torch.tensor(lig_vox).unsqueeze(0)
