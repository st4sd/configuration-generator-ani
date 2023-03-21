#!/usr/bin/env python

# Copyright IBM Inc. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1-or-later
# Authors:
#   James L. McDonagh

# Utility script to generate GAMESS US inputs using ANI
# Copyright (C) 2020  IBM Inc
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

# !/usr/bin/env python

# python packages
import argparse
import logging
import os
from datetime import datetime
import numpy as np
import logging
import pandas as pd
import re

# atomic simulation environment
import ase
from ase import io as aseio
from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase import Atoms

# pytorch and torch ani
import torch
import torchani

#RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Geometry.rdGeometry import Point3D


__version__ = "0.1"
__authors__ = "James L. McDonagh"
__contact__ = "https://github.com/Jammyzx1"
__title__ = os.path.basename(__file__)
__copyright__ = "Copyright IBM Corp. 2020"


# functions
def setup_logger(cwd, loglev="INFO"):
    """
    Make logger setup
    INPUT :
        cwd : the working directory you want the log file to be saved in
    OUTPUT:
        FILE: log file
    """
    # set log level from user
    intloglev = getattr(logging, loglev)
    try:
        intloglev + 1
    except TypeError:
        print("ERROR - cannot convert loglev to numeric value using default of 20 = INFO")
        with open("error_logging.log", "w+") as logerr:
            logerr.write("ERROR - cannot convert loglev to numeric value using default of 20 = INFO")
        intloglev = 20

    # Format routine the message comes from, the leve of the information debug, info, warning, error, critical
    # writes all levels to teh log file Optimizer.log
    logging.raiseExceptions = True
    log = logging.getLogger()
    log.setLevel(intloglev)
    pathlog = os.path.join(cwd, "{}.log".format(__title__.split(".")[0]))

    # File logging handle set up
    filelog = logging.FileHandler("{}".format(pathlog), mode="w")
    filelog.setLevel(intloglev)
    fileformat = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    filelog.setFormatter(fileformat)

    # Setup handle for screen printing only prints info and above not debugging info
    screen = logging.StreamHandler()
    screen.setLevel(10)
    screenformat = logging.Formatter('%(message)s')
    screen.setFormatter(screenformat)

    # get log instance
    log.addHandler(screen)
    log.addHandler(filelog)

    log.info("The handlers {} logging level {} {}".format(log.handlers, loglev, intloglev))
    log.info('Started {}\n'.format(datetime.now()))

    return log


def sample_conformations(mol, n_conformers, r_seed=17591, RMS=False):
    """
    Generates n_conformers on the molecule mol
    :mol: RDKit molecule object
    :n_conformers: the number of conformers to generate
    :r_seed: random seedif not given the geometry will differ each time the call is made
    """

    if RMS is False:
        conformer_index = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, numThreads=0,randomSeed=r_seed)

    else:
        conformer_index = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, numThreads=0,
                                                 randomSeed=r_seed, maxAttempts=10000, pruneRmsThresh=0.25)
    # options to give above to only accept based on RMS thresh and max attempts maxAttempts=10000, pruneRmsThresh=0.25

    Chem.rdMolTransforms.CanonicalizeMol(mol, ignoreHs=False)
    return list(conformer_index)

def energy_minimize_all_confs(mol, max_int=2000):
    """
    energy minimize all conformations of a molecule which are defined in the molecule object
    :mol: molecule object
    :max_int: max number of minimization iterations
    return 1 not all converged OR 0 all converged
    """
    log = logging.getLogger(__name__)

    # Note numThreads=0 used max avaliable threads for processor
    # Minimize all conformations in the molecule object include inter-molecular potentials and terms
    try:
        log.info("Attempting conformer minimzation with MMFF")
        result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_int, numThreads=0,
                                                   ignoreInterfragInteractions=False)
    except Exception as err:
        log.info("Attempting conformer minimzation with UFF as MMFF encountered an err {}".format(err))
        result = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_int, numThreads=0,
                                                  ignoreInterfragInteractions=False)


    converged = [ent[0] for ent in result]
    energies = [ent[1] for ent in result]

    if any(ent == 1 for ent in converged):
        log.info("WARNING - At least some conformations have failed the minimization and are not converged")
        log.info("Converged? (0 = success, 1 = failed): {}".format(converged))
        log.info("Minimized energies: {}".format(energies))

    else:
        log.info("Energy minimization of all conformers successful")
        log.info("Minimized energy: {}\n".format(*["conformer {}: Energy {} Hartree\n".format(i, energy) for i,
                                                                                    energy in enumerate(energies)]))

    # Find lowest energy conformer
    min_ener = 0.0
    indx = 0
    for i, (conv, ener) in enumerate(zip(converged, energies)):
        if i == 0:
            min_ener = ener
            indx = i

        if conv == 0 and ener < min_ener:
            min_ener = ener
            indx = i

    # https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html
    log.info("\n".join(["index {} converged (0 true; 1/-1 false) {} energy {}".format(ind, con, en) for ind, (con, en)
                        in enumerate(zip(converged, energies))]))
    log.info("Minimum energy conformer determined to be index '{}' (zero based) with an "
             "energy {}".format(indx, min_ener))

    return indx

def get_mol_from_smiles(smiles, canonicalize=True):
    """
    Function to make a mol object based on smiles
    :param smiles: str - SMILES string
    :param canonicalize: True/False - use RDKit canonicalized smile or the input resprectively
    """

    log = logging.getLogger(__name__)

    if canonicalize is True:
        s = Chem.CanonSmiles(smiles, useChiral=True)
    else:
        s = smiles
    mol = Chem.MolFromSmiles(s)
    log.debug("Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
              "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(smiles, s))
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    return mol

def smi2coordinates(smi, n_conformers=25, random_seed=17591, minimize=True):
    """
    smiles to 3D coordinates
    :param smi: smiles string
    :return: rdkit molecule object with Hydrogens added
    """
    mol = get_mol_from_smiles(smi)
    mol_with_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)

    if n_conformers > 0:
        sample_conformations(mol_with_h, n_conformers, r_seed=random_seed)

    if minimize is True:
        indx = energy_minimize_all_confs(mol_with_h)
    else:
        indx = 0

    return mol_with_h, indx

def inchi2coordinates(inchi, n_conformers=25, random_seed=17591, minimize=True):
    """
    smiles to 3D coordinates
    :param smi: smiles string
    :return: rdkit molecule object with Hydrogens added
    """
    mol = Chem.MolFromInchi(inchi)
    mol_with_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)

    if n_conformers > 0:
        sample_conformations(mol_with_h, n_conformers, r_seed=random_seed)

    if minimize is True:
        indx = energy_minimize_all_confs(mol_with_h)
    else:
        indx = 0

    return mol_with_h, indx

def xyz_representation(molecule, n_conf=-1, smiles=None):
    """
    make an xyz file representation
    :molecule: is a molecule object which has had Hydrogens added, has been embeded and has been energy minimized
    :n_conf: the molecule conformation number to make an xyz representation of
    """
    coords = Chem.MolToMolBlock(molecule, n_conf)
    atomic_positions = []
    for ent in coords.split("\n"):
        elms = ent.split()
        try:
            float(elms[0])
            if len(elms) == 16:
                atomic_positions.append([elms[3], float(elms[0]), float(elms[1]), float(elms[2])])
        except ValueError:
            pass
        except IndexError:
            pass

    atomic_positions = pd.DataFrame(atomic_positions, columns=["element", "x", "y", "z"])
    coord_only = atomic_positions[["x", "y", "z"]]
    mf = rdMolDescriptors.CalcMolFormula(molecule)

    with open("rdkit_min_conformer.xyz", "w") as fout:
        fout.write("{}\n".format(molecule.GetNumAtoms()))
        if smiles is not None:
            fout.write("From RDKit generated on {}: Molecule: {}: SMILES {}\n".format(datetime.now(), mf, smiles))
        else:
            fout.write("From RDKit generated on {}: Molecule: {}\n".format(datetime.now(), mf))
        atomic_positions.to_csv(fout, header=None, index=None, sep=" ", mode="a")

def xyz(molecule, n_conf=-1):
    """
    Get elements and atomic coordinates
    :molecule: is a molecule object which has had Hydrogens added, has been embeded and has been energy minimized
    :n_conf: the molecule conformation number to make an xyz representation of
    return pandas dataframe of elements and coordinates
    """
    coords = Chem.MolToMolBlock(molecule, confId=n_conf)
    atomic_positions = []
    for ent in coords.split("\n"):
        elms = ent.split()
        try:
            float(elms[0])
            if len(elms) == 16:
                atomic_positions.append([elms[3], float(elms[0]), float(elms[1]), float(elms[2])])
        except ValueError:
            pass
        except IndexError:
            pass

    atomic_positions = pd.DataFrame(atomic_positions, columns=["element", "x", "y", "z"])

    return atomic_positions

def test_ani():
    """
    Function to test ani install is consistent
    """

    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
        [-0.83140486, 0.39370209, -0.26395324],
        [-0.66518241, -0.84461308, 0.20759389],
        [0.45554739, 0.54289633, 0.81170881],
        [0.66091919, -0.16799635, -0.91037834]]],
        requires_grad=True, device=device)

    species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative

    energy_same = True if 40.459790705366636 - abs(energy.item()) <= 1E-6 else False
    log.info("Energy: {} should be -40.459790705366636".format(energy.item))
    log.info("Energy as expected: {}".format(energy_same))
    #Energy: -40.459790705366636



    expected_force = torch.Tensor([[ 0.0478, -0.1304, -0.0551],
        [-0.1353,  0.1581, -0.0776],
        [ 0.0804, -0.0388,  0.0387],
        [ 0.0254,  0.0076,  0.0433],
        [-0.0183,  0.0035,  0.0508]])

    force_matches = torch.isclose(force, expected_force, rtol=5e-05, atol=5e-05)
    log.info("Force:\n{}\n\nshould be\n{}\n\ntolerance: {}".format(force, expected_force, 5e-05+5e-05))
    log.info("\nForce as expected: {}\n".format(force_matches))

    return True

def ase_read_xyz(xyz_filename, fmt="xyz"):
    """
    Function to use ASE to read a molecule from an xyz file
    :param xyz_filename: str - filename and path
    :param fmt: str - file format
    """
    return aseio.read(xyz_filename, format=fmt, do_not_split_by_at_sign=False)


def ase_write_xyz(atoms, xyz_filename, fmt="xyz"):
    """
    Function to use ASE to read a molecule from an xyz file
    :param atoms: ase atoms or list of atoms
    :param xyz_filename: str - filename and path
    :param fmt: str - file format
    """
    return aseio.write(xyz_filename, atoms, format=fmt, comment="optimized xyz with ANI")


def linear(atoms, threshold=1E-4):
    """
    Function to carry out a basic test linearity of a molecules based on consecutive triples of atoms
    :param atoms: ASE atoms object - ase atoms
    """
    log = logging.getLogger(__name__)

    a0 = atoms[0]
    angles = []
    i = 0
    while i + 2 < len(atoms):
        # save angles in deg from the ASE
        angles.append(atoms.get_angle(i, i + 1, i + 2, mic=True))
        i = i + 1

    log.debug("Angles:\n{}".format(angles))
    diffs = [a - 180.0 if abs(a) > 0.0 + threshold else a for a in angles]
    if all(0.0 <= abs(d) <= threshold for d in diffs):
        log.debug("Difference in angle deg:\n{}".format(diffs))
        return True
    else:
        log.debug("Difference in angle deg:\n{}".format(diffs))
        return False

def get_elements(mol):
    """
    Function to get the element symbols from an RDKit molecule object
    :param mol: RDKit molecule object - molecule
    """

    return [atom.GetSymbol() for atom in mol.GetAtoms()]

def get_coordinates(conf):
    """
    Function to get coordinates from RDKit conformer
    :param conf: RDkit conformer - conformer
    """

    xyzcoords = []
    for atom_num in range(0, conf.GetNumAtoms()):
        coords = conf.GetAtomPosition(atom_num)
        xyzcoords.append([coords.x, coords.y, coords.z])

    return xyzcoords

def gamess_input_from_template(mol, smi, name, template, indx=-1, spin=None, charge=None):
    """
    Function to make a GAMESS input from a template options file
    :param molecule: RDKit mol
    :param smi: smiles string
    :param name: molecule name
    :param template: template file to read and use the options from
    :param indx: molecule conformer index
    :return:
    """

    log = logging.getLogger(__name__)
    mf = rdMolDescriptors.CalcMolFormula(mol)
    coordinates = xyz(mol, n_conf=indx)
    atomic_masses = ["{:.1f}".format(round(atom.GetAtomicNum(), 0)) for atom in mol.GetAtoms()]
    log.info("Atomic mass: {}".format(atomic_masses))
    coordinates.insert(loc=1, column="masses", value=atomic_masses)
    coords_csv = coordinates.to_csv(header=False, index=False, sep=" ")

    with open(template, "r") as fin:
        temp_data = [line.strip() for line in fin if line]

    out_filename = "molecule.inp"

    # Try to get data for charge and multiplicity (NOTE: currently not sure of the multiplicity
    # will be correct by this method needs some testing)
    elec_data = count_electons(mol)
    if spin is not None:
        elec_data[1] = spin.strip()
        log.info("Spin set to user defined value {}".format(elec_data[1]))

    if charge is not None:
        elec_data[0] = charge.strip()
        log.info("Charge set to user defined value {}".format(elec_data[0]))

    log.info("Output file : {}".format(out_filename))

    with open(out_filename, "w") as fout:
        for line in temp_data:
            log.debug("Line for input files: {}".format(line))
            if "MULT" in line:
                line_split = ["MULT={}".format(elec_data[1]) if "MULT" in ent else ent for ent in line.split()]
                line = " ".join([str(elm) for elm in line_split])
                log.debug("MULT found set to {}".format(elec_data[1]))

            if "ICHARG" in line:
                line_split = ["ICHARG={}".format(elec_data[0]) if "ICHARG" in ent else ent for ent in line.split()]
                line = " ".join([str(elm) for elm in line_split])
                log.debug("ICHARG found set to {}".format(elec_data[0]))

            if str(elec_data[1]) != "1" and "SCFTYP=RHF" in line:
                line_split = ["SCFTYP=ROHF".format(elec_data[0]) if "SCFTYP" in ent else ent for ent in line.split()]
                line = " ".join([str(elm) for elm in line_split])
                log.debug("SCFTYP changes to ROHF due to openshell state")

            fout.write(" {}\n".format(line.strip()))

        # fout.write("\n")
        fout.write(" $DATA\n")
        fout.write("{} {}\n".format(name, re.sub(r"[^\w]", "", mf)))
        fout.write(" C1\n")
        for ent in coords_csv.split("\n"):
            if ent:
                fout.write(" {}\n".format(ent))
        fout.write(" $END\n")

    return out_filename

def count_electons(mol, indx=-1):
    """
    This is a function to count the total number of electrons in a molecule and the number of valence electrons
    mol : RDKit molecule object
    indx: int - confomrer index
    """
    log = logging.getLogger(__name__)

    mf = rdMolDescriptors.CalcMolFormula(mol)

    number_of_electrons = 0
    total_charge = 0
    number_unpaired_es = 0

    for atom in mol.GetAtoms():
        atom_n_e = atom.GetAtomicNum()
        atoms_sym = atom.GetSymbol()
        atom_chr = atom.GetFormalCharge()
        # TODO: assuming if the atom has a 'radical electron' it means unpaired accounting for bonding etc
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
        rade = atom.GetNumRadicalElectrons()

        number_of_electrons = number_of_electrons + atom_n_e - atom_chr
        total_charge = total_charge + atom_chr
        number_unpaired_es = number_unpaired_es + rade

    number_of_valence_electrons = Chem.Descriptors.NumValenceElectrons(mol)
    spin = number_unpaired_es + 1

    if (number_of_electrons % 2) == 0:
        radical = False
    else:
        radical = True
    log.info("Molecular formula: {}".format(mf))
    log.info("Total number of electrons: {}".format(number_of_electrons))
    log.info("Total charge: {}".format(total_charge))
    log.info("From RDKit number of valence electrons {}".format(number_of_valence_electrons))
    log.info("Radical molecule (only valid for organic moleules) {}".format(radical))
    log.info("Spin state is {}\n--------------------------\n".format(spin))

    return [total_charge, spin, number_of_valence_electrons, number_of_electrons, radical]

def run():
    """
    """

    try:
        usage = "python {} {}\n".format(__title__, " options .....")

        parser = argparse.ArgumentParser(description="Command line binary points script",
                                         usage=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-xyz", "--xyz_file", metavar="FILE",
                            action="store", help="xyz file to read",
                            default="none")

        parser.add_argument("-xyzp", "--xyz_path", metavar="PATH",
                            action="store", help="The path to the xyz file to read",
                            default=os.getcwd())

        parser.add_argument("-m", "--ani_model", metavar="STR",
                            action="store",
                            help="ANI model choice of ani1 for ANI1X, ani2 for ANI2X and anicc for ANI1ccx",
                            default="ani2")
        parser.add_argument("-i", "--max_iterations", metavar="INT",
                            action="store",
                            help="maximum number of times ASE can iterate over the optimzation using ANI", type=int,
                            default=5000)
        parser.add_argument("-o", "--optimizer", metavar="STR",
                            action="store", help="optimizer for the run one of bfgs, lbfgs, gp, fire, mdmin", type=str,
                            default="bfgs")
        parser.add_argument("--history_file", metavar="STR",
                            action="store", help="filename to store the optimization trajectory information to",
                            type=str,
                            default='trajectory.traj')
        parser.add_argument("--restart_file", metavar="STR",
                            action="store", help="filename to store the optimization restart data to", type=str,
                            default='trajectory.pckl')
        parser.add_argument("--test",
                            action="store_true", help="Run a test of ANI and ASE first",
                            default=False)
        parser.add_argument("--temperature",
                            action="store", help="Temperature in kelvin for thermochemical analysis",
                            type=float, default=298.15)
        parser.add_argument("--pressure",
                            action="store", help="Pressure in kelvin for thermochemical analysis",
                            type=float, default=101325.0)
        parser.add_argument("--force_tolerance",
                            action="store", help="optimizer force tolerance",
                            type=float, default=0.05)
        parser.add_argument("--spin", type=str,
                            action="store", help="molecules spin (multiplicity)",
                            default=None)
        parser.add_argument("--charge", type=str,
                            action="store", help="molecules charge state",
                            default=None)
        parser.add_argument("--outname", type=str,
                            action="store", help="name to save optimized molecules xyz file to",
                            default="opt")
        parser.add_argument("--loglev", action="store", default="INFO", help="log level")

        op = parser.parse_args()

    except argparse.ArgumentError as ee:
        print("\nERROR - command line arguments are ill defined please check the arguments\n")
        raise ee

    setup_logger(os.getcwd(), op.loglev)
    log = logging.getLogger(__name__)
    log.info("\nAuthors       : {}\nOrganiszation : IBM Research Europe\nCreated"
             "       : June 2020\nProgram       : {}\nVersion       : {}\n "
             "--------------------------------\n".format(__authors__, __title__, __version__))

    log.info("Command line input =\n\t{}".format(op))

    # check for extension to xyz
    if op.xyz_file[-4:] != ".xyz":
        op.xyz_file = op.xyz_file + ".xyz"

    # Check if the file path is given if it is prepend it to the file name
    if op.xyz_path.lower() != "none":
        op.xyz_file = os.path.join(op.xyz_path, op.xyz_file)

    # print the file name
    log.info("Reading the xyz file {}".format(op.xyz_file))

    # Enable flow variable of str None to be interpreted as python None
    if op.spin is not None:
        if op.spin.lower().strip() == "none":
            op.spin = None

    if op.charge is not None:
        if op.charge.lower().strip() == "none":
            op.charge = None

    # For flow input
    if op.xyz_file.lower() == "none":
        op.xyz_file = None

    # Run ASE and ANI test for consistency
    if op.test is True:
        ret = test_ani()
        if ret is True:
            log.info("Ani and ASE is consistent and runs as expected")
        else:
            log.warning("Ani and ASE install is inconsistent and does not run as expected")

    # Read in or create the molecule
    molecule = None

    if op.xyz_file is not None:
        log.info("XYZ file geometry used instead of RDKit from smiles")
        molecule = ase_read_xyz(op.xyz_file)

    # Get the ANI calculator object from ASE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if op.ani_model.lower() == "ani1":
        log.info("ANI model 1 will be used for potential energy evaluation")
        model = torchani.models.ANI1x(periodic_table_index=True).to(device).ase()
    elif op.ani_model.lower() == "anicc":
        log.info("ANI model CCX will be used for potential energy evaluation")
        model = torchani.models.ANI1ccx(periodic_table_index=True).to(device).ase()
    elif op.ani_model.lower():
        log.info("ANI model 2 will be used for potential energy evaluation")
        model = torchani.models.ANI2x(periodic_table_index=True).to(device).ase()
    else:
        log.warning("Unrecognised ANI model asked for {}. Using defualt ANI2X".format(op.ani_model))
        model = torchani.models.ANI2x(periodic_table_index=True).to(device).ase()

    # Build the molecule and calculator object system - NOTE molecule is the minimized geometry from RDKit above
    if molecule is not None and model is not None:
        system = Atoms(molecule, calculator=model)
    else:
        log.warning("Model or molecule is not set exiting as computer says no :( ..........")
        raise UserError

    # object optimizer with history and restart file writing NOTE: will automatically use restart files if they are present
    if op.optimizer.lower() == "bfgs":
        log.info("Optimizing with BFGS")
        opt = BFGS(system, trajectory=op.history_file, restart=op.restart_file)
    elif op.optimizer.lower() == "lbfgs":
        log.info("Optimizing with LBFGS")
        opt = LBFGS(system, trajectory=op.history_file, restart=op.restart_file)
    elif op.optimizer.lower() == "gp":
        log.info("Optimizing with GP")
        opt = GPMin(system, trajectory=op.history_file, restart=op.restart_file)
    elif op.optimizer.lower() == "fire":
        log.info("Optimizing with FIRE MD")
        opt = FIRE(system, trajectory=op.history_file, restart=op.restart_file)
    elif op.optimizer.lower() == "mdmin":
        log.info("Optimizing with MD")
        opt = MDMin(system, trajectory=op.history_file, restart=op.restart_file)

    # Run the optimization
    opt.run(fmax=op.force_tolerance, steps=op.max_iterations)
    min_energy = system.get_potential_energy()
    log.info("Energy minimized: {} eV".format(min_energy))
    ase_write_xyz(system, xyz_filename="{}.xyz".format(op.outname))

    # Run a vibrational analysis
    linear_molecule = linear(system)
    log.info("Is the molecule likely linear? {}".format(linear_molecule))
    vib = Vibrations(system)
    vib.run()

    frequencies = vib.get_frequencies()
    log.info("Vibrational frequencies:\n")
    log.info(frequencies)
    vib.summary()
    log.info("\n 3N-6 will be vibrations other are rotations and "
             "translations for non-linear molecules 3N-5 for linear, other than those none should be negative for a minima")

    # Run thermochemistry analysis
    log.info("Thermochemistry analysis starting .....")
    try:
        frequencies_energy = vib.get_energies()
        thermo = IdealGasThermo(vib_energies=frequencies_energy,
                                potentialenergy=min_energy,
                                atoms=system,
                                geometry='linear',
                                symmetrynumber=1,
                                spin=0)

        G = thermo.get_gibbs_energy(temperature=op.temperature, pressure=op.pressure)
        H = thermo.get_enthalpy(temperature=op.temperature)
        S = thermo.get_entropy(temperature=op.temperature, pressure=op.pressure)

        # Write output files
        aseio.write("optimized.xyz", system, format="xyz", comment="No negative frequencies")
        with open("energies.csv", "w") as fout:
            fout.write("energy_au,negative_frequencies,G,H,S\n")
            fout.write("{:.5f},False,{:.5f},{:.5f},{:.5f}\n".format(min_energy, G, H, S))

    except ValueError as verr:
        log.warning("{} Thermochemistry analysis is not avaliable".format(verr))

        # Write output files
        aseio.write("optimized.xyz", system, format="xyz", comment="{}".format(verr))
        with open("energies.csv", "w") as fout:
            fout.write("energy_au,negative_frequencies,G,H,S\n")
            fout.write("{:.5f},True,NaN,NaN,Nan\n".format(min_energy))


if __name__ == "__main__":
    run()

