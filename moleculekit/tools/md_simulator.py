import argparse
import os
import shutil
import sys
import numpy as np
from glob import glob
from tqdm import trange, tqdm
from htmd.units import convert
from htmd.projections.metric import Metric
from time import sleep

sys.setrecursionlimit(25000)
import matplotlib
matplotlib.use('Agg')

ACEMD_EXE = 'acemd3'
TIMESTEP = 4
SIMTYPE_ID = {'globular': 0, 'membrane': 1}
TOT_STEP = [3, 5]
CURR_STEP = [{'Equilibration': 1, 'Production': 2, 'Analysis': 3}, {'Equilibration_1': 1, 'Equilibration_2': 2, 'Equilibration_3': 3, 'Production': 4, 'Analysis': 5}]

class MDSimulator:
    def __init__(self, args):

        for arg, value in args.items():
            if value == '':
                args[arg] = None

        self.vars = argparse.Namespace(**args)

    def run(self):
        from htmd.protocols.equilibration_v2 import Equilibration as EquilV2
        from htmd.protocols.production_v6 import Production as ProdV6

        self._equil_protocols = {2: EquilV2}
        self._protocols = {6: ProdV6}

        # Creates the output folder and writes the 'inopt.yml' file with all the options used
        if not os.path.isdir(self.vars.outdir):
            os.makedirs(self.vars.outdir)

        # Correctly initialize the input folder
        self.vars.inputdir = self._ensurelist(self.vars.inputdir)
        outdir = self.vars.outdir

        # Retrieve the simulation time for the equilibration time based on the simulation type
        equiltime, runtime = self._getSimulationTimes()

        # Return the number of builds
        numbuilds = self._checkInputdirs()

        # Returns the completed simulation. In this way, we will not have the ovewriting of the simulations
        sims_completed = self._checkStatus(outdir, TIMESTEP)
        eq_completed = sims_completed[:-1] if len(sims_completed) > 1 else [sims_completed[0]]
        prod_completed = sims_completed[-1]

        eq_tosim = self._prepareGlobularEquilibration(numbuilds, equiltime, eq_completed, protocol_version=2)
        self._executeSimulations(eq_tosim)
        sims_completed = self._checkStatus(outdir, TIMESTEP)
        eq_completed = sims_completed[:-1] if len(sims_completed) > 1 else [sims_completed[0]]
        prod_completed = sims_completed[-1]
        equilfolders = eq_tosim + [s for subruns in eq_completed for s in subruns]

        # Piece of code for running [Production]
        equilfolders = list(set(sorted(equilfolders)))
        prod_tosim = self._prepareProduction(equilfolders, runtime, prod_completed, protocol_version=6)

        self._executeSimulations(prod_tosim)

        sims_completed = self._checkStatus(outdir, TIMESTEP)
        eq_completed = sims_completed[:-1] if len(sims_completed) > 1 else [sims_completed[0]]
        prod_completed = sims_completed[-1]

        eqfolders, prodfolders = self._toAnalyze(self.vars.numruns, sims_completed)

        for n, (esims, psim) in enumerate(zip(eqfolders, prodfolders)):
            trj_id = os.path.basename(esims[0])
            refmol, eqsim, prodsim = self.AnalyzeTrajectories(esims, psim, trj_id)  # n+1)
            self.prepareOutputs(refmol, eqsim, prodsim, n+1)

        shutil.make_archive('output', 'zip', self.vars.outdir)


    def _toAnalyze(self, nruns, sims_completed):

        toanalyze_eq = []
        toanalyze_prod = []

        for n in range(nruns):
            tmp_eq = []
            tmp_prod = []
            idx = n + 1
            correct = True
            eq_expected = [os.path.join(self.vars.outdir, 'equil/{}'.format(idx))]
            eq_completed = sims_completed[0]

            for eq in eq_expected:
                if eq not in eq_completed:
                    correct = False
                    break
            if not correct:
                continue
            tmp_eq.extend(eq_expected)

            prod_expected = os.path.join( self.vars.outdir, 'prod/{}'.format(idx))
            prod_completed = sims_completed[1]
            if prod_expected not in prod_completed:
                continue
            tmp_prod.append(prod_expected)
            toanalyze_eq.append(tmp_eq)
            toanalyze_prod.append(tmp_prod)

        return toanalyze_eq, toanalyze_prod


    def prepareOutputs(self, refmol, eqsim, prodsim, nrun):
        from htmd.simlist import simmerge

        outfolder = self.vars.outdir

        equilfolders = glob(os.path.join(outfolder, 'equil*'))
        prodfolders = glob(os.path.join(outfolder, 'prod*'))
        generator = []
        figures = self._ensurelist(os.path.join(outfolder, 'figures'))
        clusters = self._ensurelist(os.path.join(outfolder, 'clusters'))
        outs = equilfolders + prodfolders + generator + figures + clusters

        allsimlists = simmerge(eqsim, prodsim)
        refmol.center()
        m = refmol
        refmol.write(os.path.join(self.vars.outdir, 'filtered_run{}.pdb'.format(nrun)))
        for s in allsimlists:
            m.read(s.trajectory[0], append=True)
            m.align('name CA')
        m.write(os.path.join(self.vars.outdir, 'trajectory_run{}.xtc'.format(nrun)))

        dirs_torm = [ d for d in glob('filtered*') if os.path.isdir(d)]
        for d in dirs_torm:
            shutil.rmtree(d)


    def _ensurelist(self, v):
        if not isinstance(v, list):
            return [v]

        return v


    def _getSimulationTimes(self):
        if self.vars.equiltime != 'auto':
            equiltime = float(self.vars.equiltime)
        else:
            equiltime =  3

        return equiltime, self.vars.runtime


    def _checkInputdirs(self):
        inputdirs = self.vars.inputdir
        numruns = self.vars.numruns
        numbuilds = 0

        for inputdir in inputdirs:
            if not os.path.isdir(inputdir):
                raise NotADirectoryError('{} of {}'.format(inputdir, inputdirs))
            else:
                numbuilds += 1

        if numruns < numbuilds:
            print('The number of runs ({}) is less than the number of builds provided ({}). '
                  'Defaulting numruns to {}'.format(numruns, numbuilds, numbuilds))
            self.vars.numruns = numbuilds
        return numbuilds

    def _checkStatus(self, outdir, timestep=4):
        # Checks for equil* or prod folders and subfolders. The simulation folders found are inspected to get if the
        # simulations were completed.
        #
        # Return a list of lists [[equil], [prod ]]
        from htmd.molecule.molecule import Molecule

        sims_found = [glob(os.path.join(outdir, 'equil*', '*')), glob(os.path.join(outdir, 'prod', '*'))]
        # here we compared the simulation time in the acemd input with the frames recorded in output.xtc found
        sims_completed = []
        for subruns in sims_found:
            tmp = []
            for s in subruns:
                input_md = os.path.join(s, 'input')
                time_input = None
                for line in open(input_md):
                #    if 'set numsteps' in line:
                    if 'run' in line:
                        n_steps = int(line.strip().split()[-1])
                        time_input = n_steps * timestep / 1000
                        break
                trj = os.path.join(s, 'output.xtc')
                if not os.path.isfile(trj):
                    continue
                m = Molecule(trj)
                time_md = m.time[-1]
                if time_md == time_input:
                    tmp.append(s)
            sims_completed.append(tmp)
        return sims_completed


    def _getRunsPerBuild(self, numbuilds, numruns):
        # Returns an array in which each number indicates the number of runs to perform for each built system provided

        runs_per_build = np.ones(numbuilds, dtype=int) * int(np.floor(numruns / numbuilds))
        diff = np.mod(numruns, numbuilds)
        idx_to_increment = np.random.choice(numbuilds, diff, replace=False)
        runs_per_build[idx_to_increment] += 1

        return runs_per_build


    def _prepareGlobularEquilibration(self, numbuilds, equiltime, completed_sims, protocol_version=2):

        from htmd.mdengine.acemd.acemd import AtomRestraint

        inputdirs = self.vars.inputdir
        simplerundir = self.vars.outdir
        numruns = self.vars.numruns
        completed_sims = [s for subruns in completed_sims for s in subruns]
        protocol = 'Equilibration'
        key = 'equil'
        # Determine mapping of equil to execute
        runs_per_build = self._getRunsPerBuild(numbuilds, numruns)
        print('{} Globular System equiltime: {} ns'.format(protocol, equiltime))

        runsteps = convert("ns", "timesteps", equiltime, timestep=TIMESTEP)
        restraints = [AtomRestraint('protein and name CA', 0, [(1, 0), (1, runsteps/2), (0, runsteps)]),
                      AtomRestraint('protein and noh and not name CA', 0, [(0.1, 0), (0.1, runsteps / 2), (0, runsteps)])]

        k = 1
        folders = []
        for i in range(numbuilds):
            print('Preparing equilibration for build #{}'.format(i))
            for j in trange(runs_per_build[i]):
                indir = inputdirs[i]
                outdir = os.path.join(simplerundir, key, '{}'.format(k))
                if outdir not in completed_sims:
                    md = self._equil_protocols[protocol_version](_version=3)
                    md.restraints = restraints
                    if self.vars.constraints == 'protein-ligand':
                        addRestrain = [AtomRestraint('resname {}'.format(self.vars.ligresname), 0, [(1, 0), (1, runsteps)])]
                        restraints = [] if md.restraints is None else md.restraints
                        md.restraints = restraints + addRestrain
                    self._makeSimulation(md, indir, outdir, equiltime)
                    folders.append(outdir)

                k += 1
        return folders


    def _prepareProduction(self, equilfolders, runtime, completed_sims, protocol_version=6):

        from htmd.mdengine.acemd.acemd import GroupRestraint

        simplerundir = self.vars.outdir
        #numruns = self.vars.numruns
        runtime = runtime

        #runsteps = convert("ns", "timesteps", runtime, timestep=TIMESTEP)

        protocol = 'Production'
        key = 'prod'


        print('{} runtime: {} ns'.format(protocol, runtime))

        folders = []

        for i, indir in enumerate(tqdm(equilfolders)):
            output_idx = os.path.basename(indir)
            outdir = os.path.join(simplerundir, key, '{}'.format(output_idx))

            if outdir not in completed_sims:

                md = self._protocols[protocol_version](_version=3)

                is_ready = self._makeSimulation(md, indir, outdir, runtime)
                if is_ready:
                    folders.append(outdir)

        return folders


    def _makeSimulation(self, md, indir, outdir, runtime):

        #runsteps = convert("ns", "timesteps", runtime, timestep=TIMESTEP)

        # Define simulation time
        md.runtime = runtime
        md.timeunits = 'ns'

        # Write the Equilibration directory
        try:
            md.write(indir, outdir)
            f = open(os.path.join(outdir, 'run.sh'), 'w')
            f.write('#!/bin/sh\n')
            DEVICE = "--platform CPU" if not self.vars.use_gpu else "--platform GPU"
            f.write('{} {} input > log.txt 2>&1'.format(ACEMD_EXE, DEVICE))
            f.close()
            return True
        except:
            print("Required files in {} not found. Simulation {} skipped".format(indir, outdir))
            return False


    def _executeSimulations(self, folders):
        if self.vars.use_gpu:
            from htmd.queues.localqueue import LocalGPUQueue
            execmode = LocalGPUQueue()
        else:
            from htmd.queues.localqueue import LocalCPUQueue
            execmode = LocalCPUQueue()
        execmode.submit(folders)
        execmode.wait()
        execmode.retrieve()

        return folders


    def AnalyzeTrajectories(self, equilfolders, prodfolders, nrun):
        from htmd.molecule.molecule import Molecule

        figure_folder = os.path.join(self.vars.outdir, 'figures')
        if not os.path.isdir(figure_folder):
            os.makedirs(figure_folder)
        clusters_folder = os.path.join(self.vars.outdir, 'clusters')
        if not os.path.isdir(clusters_folder):
            os.makedirs(clusters_folder)

        folders = equilfolders + prodfolders
        eq_names = ['eq{}_run{}'.format(n + 1, nrun) for n, e in enumerate(equilfolders)]
        prod_names = ['prod{}_run{}'.format(n + 1, nrun) for n, p in enumerate(prodfolders)]
        names = eq_names + prod_names

        refmol, eq_fsim, prod_fsim, refmol_sims, eq_sim, prod_sim = self._getSimsAndReference(folders, names)

        # compute RMSD --> CA
        print('\nComputing Protein RMSD\n')
        print('\tfor CA\n')
        dataRMSDCA_eq = self._computeRMSD('name CA', eq_fsim, refmol, onlyData=True)
        dataRMSDCA_prod = self._computeRMSD('name CA', prod_fsim, refmol, onlyData=True)


        # compute RSMD --> sidechain
        print('\tfor SideChain\n')
        dataRMSDSide_eq = self._computeRMSD('protein and not backbone and noh', eq_fsim, refmol, onlyData=True)
        dataRMSDSide_prod = self._computeRMSD('protein and not backbone and noh', prod_fsim, refmol, onlyData=True)

        if self.vars.ligresname is not None:
            # compute RMSD --> Ligand
            print('\tfor Ligand {}\n'.format(self.vars.ligresname))
            dataRMSDLigand_eq = self._computeRMSD('resname {} and noh'.format(self.vars.ligresname), eq_fsim, refmol, onlyData=True)
            dataRMSDLigand_prod = self._computeRMSD('resname {} and noh'.format(self.vars.ligresname), prod_fsim, refmol, onlyData=True)

        # plot RMSD
        print('\nPlotting RMSD\n')
        outname = os.path.join(figure_folder, 'RMSD_CA_Sidechain_run{}'.format(nrun))
        self._plotRMSD([dataRMSDCA_eq, dataRMSDSide_eq], [dataRMSDCA_prod, dataRMSDSide_prod], 'RMSD CA/Sidechain',
                       fname=outname, skip=10)

        if self.vars.ligresname is not None:
            outname = os.path.join(figure_folder, 'RMSD_Ligand_run{}'.format(nrun))
            self._plotRMSD([dataRMSDLigand_eq], [dataRMSDLigand_prod], 'RMSD Ligand',
                       fname=outname, skip=10)
        # MD info
        self._getMDInfo(refmol_sims, eq_sim, prod_sim, self.vars.outdir)

        return refmol, eq_fsim, prod_fsim


    def _getMDInfo(self, refmol, eqsim, prodsim, outfolder):
        import pandas as pd

        m = refmol.copy()
        m.read(eqsim[0].trajectory[0])
        teq = m.time[-1] / 1000
        m = refmol.copy()
        m.read(prodsim[0].trajectory[0])
        tprod = m.time[-1] / 1000

        numAtoms = refmol.numAtoms
        numWaters = np.sum(refmol.get('element', 'water') == 'O' )
        numResidues = np.sum( refmol.name == 'CA')

        columns = [ '# Atoms', '# Protein Residues', '# Waters', 'Equilibration Time (ns)', 'Production Time (ns)']
        types = ['int', 'int', 'int', 'float', 'float']
        datas = [int(numAtoms), numResidues, numWaters, teq, tprod]
        df = pd.DataFrame(columns=columns)
        df.loc[1] = datas
        df = df.astype(dtype={k:v for k, v in zip(columns, types)})

        fname = 'mdInfo.csv'
        fname = os.path.join(outfolder, fname)
        df.to_csv('{}'.format(fname), index=False)


    def _computeRMSD(self, selection, fsim, refmol, onlyData=False):
        from htmd.projections.metricrmsd import MetricRmsd
        metrRmsd = Metric(fsim)
        metrRmsd.set(MetricRmsd(refmol, selection, 'name CA'))
        dataRmsd = metrRmsd.project()

        if onlyData:
            dataRmsd = np.concatenate(dataRmsd.dat)
            if len(dataRmsd.shape) == 1:
                dataRmsd = dataRmsd.reshape(dataRmsd.shape[0], 1)

        return dataRmsd


    def _plotRMSD(self, datas_eq, datas_prod, title, fname='RMSD', skip=1):
        import matplotlib.pyplot as plt
        import pandas as pd

        data1_eq = datas_eq[0]
        data1_prod = datas_prod[0]

        if len(datas_eq) == 2:
            data2_eq = datas_eq[1]
            data2_prod = datas_prod[1]

        frames_eq = data1_eq.shape[0]
        time_eq = frames_eq * 0.1
        frames_prod = data1_prod.shape[0]
        time_prod = frames_prod * 0.1
        tot_frames = frames_eq + frames_prod
        tot_time = tot_frames * 0.1

        all_data1 = np.concatenate((np.array([[0.0]]), data1_eq, data1_prod))
        if len(datas_eq) == 2:
            all_data2 = np.concatenate((np.array([[0.0]]), data2_eq, data2_prod))
            y2 = all_data2.squeeze()

        x = np.arange(all_data1.shape[0]) * 0.1
        y1 = all_data1.squeeze()

        columns = ['Time', 'RMSD_CA', 'RMSD_Sidechain'] if len(datas_eq) == 2 else ['Time', 'RMSD']
        labels = ['CA', 'Sidechain'] if len(datas_eq) == 2 else ['Ligand']

        df = pd.DataFrame(columns=columns)
        df['Time'] = pd.Series(x)
        df[columns[1]] = pd.Series(y1)
        if len(datas_eq) == 2:
            df[columns[2]] = pd.Series(y2)
        df.to_csv('{}.csv'.format(fname))

        if skip != 1:
            all_data1 = all_data1[0::skip]
            if len(datas_eq) == 2:
                all_data2 = all_data2[0::skip]
                y2 = all_data2.squeeze()
            x = np.arange(all_data1.shape[0])  # * 0.1
            y1 = all_data1.squeeze()

        yText = 0.05
        xLabelEq = (time_eq / 2) - 0.05
        ## fixing prod label
        if skip != 1:
            xLabelProd = time_prod / 2 + time_eq - 0.05
        else:
            xLabelProd = ((tot_time - time_eq) / 2) + time_eq - 0.05

        plt.plot(x, y1, color='#1e01a8', marker='o', label=labels[0])
        if len(datas_eq) == 2:
            plt.plot(x, y2, color='orange', marker='o', label=labels[1])
        plt.ylabel('RMSD [Angstrom]')
        plt.xlabel('Time [ns]')
        plt.title(title)
        plt.grid(axis='y')
        plt.axvline(x=time_eq, linestyle='--', c='k')
        plt.text(xLabelEq, yText, "Equil")
        plt.text(xLabelProd, yText, "Prod")
        plt.legend()
        plt.yticks(np.arange(0, 16, 1), np.arange(0, 16, 1).astype(int))
        if skip != 1:
            xt, _ = plt.xticks()
            xtl = xt.copy()
            xeq = np.arange(0, time_eq, 1)
            xprod = np.arange(time_eq+1, tot_time + 1, 1)
            tics = np.concatenate((xeq, xprod))

            plt.xticks(tics, tics.astype(int))

        svg = plt.gcf()
        svg.tight_layout()
        svg.savefig('{}.svg'.format(fname), dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
        svg.clf()


    def _computeSS(self, fsim, onlyData=False):
        from htmd.projections.metricsecondarystructure import MetricSecondaryStructure
        metrSS = Metric(fsim)
        metrSS.set(MetricSecondaryStructure(sel='protein', ))
        dataSS = metrSS.project()

        return dataSS.dat[0] if onlyData else dataSS


    def _computeContacts(self, fsim, threshold=8, skip=2, onlyData=False):
        from htmd.projections.metricdistance import MetricSelfDistance
        from htmd.molecule.molecule import Molecule

        pdb = fsim[0].molfile
        m = Molecule(pdb)
        first = np.unique(m.resid)[0]
        last = np.unique(m.resid)[-1]
        reslist = np.arange(first, last, skip).astype(str).tolist()
        resstring = " ".join(reslist)

        metrContacts = Metric(fsim)
        metrContacts.set(
            MetricSelfDistance('name CA and resid {}'.format(resstring), groupsel='residue', metric='contacts',
                               threshold=threshold))
        dataContacts = metrContacts.project()

        return dataContacts.dat[0] if onlyData else dataContacts


    def _getSimsAndReference(self, simfolders, suffix=None):
        from htmd.simlist import simlist, simfilter, simmerge
        from htmd.molecule.molecule import Molecule
        names = ['filtered_{}'.format(n) for n in range(len(simfolders))] if suffix is None else \
            ['filtered_{}'.format(s) for s in suffix]
        fsims = []
        sims = []
        for simf, n in zip(simfolders, names):
            sim = simlist(simf, os.path.join(simf, 'structure.pdb'))
            fsim = simfilter(sim, n, 'not water and not lipids and not resname CHL1')
            fsims.append(fsim)
            sims.append(sim)

        refmol_sims = Molecule(sims[0][0].molfile)
        refmol = Molecule(fsims[0][0].molfile)
        eqsims = []
        eqfsims = []
        prodsims = []
        prodfsims = []

        if len(fsims) % 2 == 0:
            prodfsims = fsims.pop(-1)
            prodsims = sims.pop(-1)

        if len(fsims) == 1:
            eqfsims = fsims[0]
            eqsims = sims[0]

        elif len(fsims) == 3:
            eqfsims = fsims.pop(0)
            eqsims = sims.pop(0)
            for f, s in zip(fsims, sims):
                f[0].molfile = eqfsims[0].molfile
                s[0].molfile = eqsims[0].molfile
                eqfsims = simmerge(eqfsims, f)
                eqsims = simmerge(eqsims, s)

        return refmol, eqfsims, prodfsims, refmol_sims, eqsims, prodsims
