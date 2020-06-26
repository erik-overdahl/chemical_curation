import os
import math
import logging
import pathlib
import re
import sys
import pandas

from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import SaltRemover
#from rdkit.Chem.MolStandardize import rdMolStandardize

from molvs import normalize, tautomer, metal

#from rdkit import rdBase

# from chemical_curation import curate

def setup_logging_to_console():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG)
    logger.addHandler(stream)


class Dataset:
    def __init__(self, filename, smiles_col='SMILES', mol_col='ROMol', target_cols=[]):
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.inchi_col = 'inchi'

        self.target_cols = target_cols
        
        self.data = self.load_data(filename)


    def calculate_inchis(self, inchi_col=None):
        if inchi_col is not None:
            self.inchi_col = inchi_col
        self._data[self.inchi_col] = self._data[self.mol_col].apply(Chem.MolToInchi)

        
    def load_data(self, filename):
        logging.info(f'Reading {filename}')
        file_ext = pathlib.Path(filename).suffix
        if file_ext == ".sdf":
            df = PandasTools.LoadSDF(filename, molColName = self.mol_col)
            if self.smiles_col not in df.columns:
                df[self.smiles_col] = df[self.mol_col].apply(Chem.MolToSmiles)

        elif file_ext in [".csv", ".tsv", ".smi"]:
            sep = ","
            if file_ext == ".tsv":
                sep = "\t"
            elif file_ext == ".smi":
                pass # what?
            df = pandas.read_csv(filename, sep = sep)
            PandasTools.AddMoleculeColumnToFrame(df, smilesCol=self.smiles_col)

        elif file_ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filename)
                
        elif file_ext == '':
            # TODO: Error: Cannot determine file type
            pass
        else:
            # TODO: Error: file type not supported
            pass
        
        return df


class MolCleaner:
    
    def __init__(self, organic_elements=[1,5,6,7,8,9,14,15,16,17,33,34,35,53], record_keeper=None,
                 salt_remover=None, normalizer=None, tautomerizer=None, metal_disconnector=None):

        self._cleaning_funcs_in_order = [self.filter_is_organic, self.filter_not_mixture, self.clean_strip_salts,
                                         self.clean_to_normalized, self.clean_to_canonical_tautomer, self.clean_disconnect_metals]
        self._organic_elements = set([6] + organic_elements)
        self.record_keeper = record_keeper
        if salt_remover is not None:
            self.salt_remover = salt_remover
        else:
            self.salt_remover = SaltRemover.SaltRemover()

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer(normalizations=normalize.NORMALIZATIONS,
                                                   max_restarts=normalize.MAX_RESTARTS)

        if tautomerizer is not None:
            self.tautomerizer = tautomerizer
        else:
            self.tautomerizer= tautomer.TautomerCanonicalizer(transforms=tautomer.TAUTOMER_TRANSFORMS,
                                                              scores=tautomer.TAUTOMER_SCORES,
                                                              max_tautomers=tautomer.MAX_TAUTOMERS)

        if metal_disconnector is not None:
            self.metal_disconnector = metal_disconnector
        else:
            self.metal_disconnector = metal.MetalDisconnector()

    @property
    def organic_elements(self):
        return self._organic_elements

    @organic_elements.setter
    def organic_elements(self, organic_elements):
        self._organic_elements = set([6] + organic_elements)
            
    @property
    def normalizations(self):
        return self.normalizer.normalizations

    @normalizations.setter
    def normalizations(self, normalizations):
        self.normalizer.normalizations = normalizations

    @property
    def normalizer_max_restarts(self):
        return self.normalizer.max_restarts

    @normalizer_max_restarts.setter
    def normalizer_max_restarts(self, max_restarts):
        self.normalizer.max_restarts = max_restarts

    @property
    def tautomer_transforms(self):
        return self.tautomerizer.transforms

    @tautomer_transforms.setter
    def tautomer_transforms(self, tautomer_transforms):
        self.tautomerizer.transforms = tautomer_transforms

    @property
    def tautomer_scores(self):
        return self.tautomerizer.scores

    @tautomer_scores.setter
    def tautomer_scores(self, tautomer_scores):
        self.tautomerizer.scores = tautomer_scores

    @property
    def max_tautomers(self):
        return self.tautomerizer.max_tautomers

    @max_tautomers.setter
    def tautomer_transforms(self, max_tautomers):
        self.tautomerizer.max_tautomers = max_tautomers


    def filter_is_organic(self, mol):
        mol_atoms = set([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        if mol_atoms.issubset(self._organic_elements):
            return mol
        return None

    
    def filter_not_mixture(self, mol):    
        mol_fragments = Chem.GetMolFrags(mol)
        if len(mol_fragments) == 1:
            return mol
        return None


    def clean_strip_salts(self, mol):
        res, deleted_salts = self.salt_remover.StripMolWithDeleted(mol)
        if res.GetNumAtoms == 0:
            mol = None
        if len(deleted_salts) > 0:
            logging.info(f'strip_salts changed {row[smiles_col]} to {new_smiles}')
            mol = res # update the molecule
        return mol

    
    def clean_to_normalized(self, mol):
        normal_mol = self.normalizer.normalize(mol)
        if len(normal_mol.GetAtoms()) == 0:
            normal_mol = None
        return normal_mol


    def clean_to_canonical_tautomer(self, mol):
        canonical_tautomer = self.tautomerizer(mol)
        if len(canonical_tautomer.GetAtoms()) == 0:
            canonical_tautomer = None
        return canonical_tautomer


    def clean_disconnect_metals(self, mol):
        disconnected_metals_mol = self.metal_disconnector(mol)
        if len(disconnected_metals_mol.GetAtoms()) == 0:
            disconnected_metals_mol = None
        return disconnected_metals_mol


    def add_mol_cleaning_function(self, func, step):
        self._cleaning_funcs_in_order.insert(step - 1, func)

        
    def remove_mol_cleaning_function(self, step):
        self._cleaning_funcs_in_order.pop(step - 1)
    

    def apply_all_transformations(self, mol):
        for step, func in enumerate(self._cleaning_funcs_in_order):
            new_mol = func(mol)
            if self.record_keeper is not None:
                if new_mol is None:
                    self.record_keeper.append(f'{func.__name__} removed {Chem.MolToSmiles(mol)}')
                elif new_mol != mol:
                    self.record_keeper.append(f'{func.__name__} changed {Chem.MolToSmiles(mol)} to {Chem.MolToSmiles(new_mol)}')
            if new_mol is None:
                break
        return new_mol
            
    
class Curator:
    
    def __init__(self, mol_cleaner=None):
        self.record_keeper = []
        if mol_cleaner is None:
            self.mol_cleaner = MolCleaner(record_keeper=self.record_keeper)
        else:
            self.mol_cleaner = mol_cleaner
            

    def clean_mols(self, dataset):
        df = dataset.data.copy()
        df[dataset.mol_col] = df[dataset.mol_col].apply(self.mol_cleaner.apply_all_transformations)
        retained = df.loc[df[dataset.mol_col].notna()]
        self.record_keeper.extend(self.mol_cleaner.record_keeper)
        # if we to know which rows didn't make it, we would use this
        #removed = df.loc[~df.index.isin(retained.index)]
        return retained


if __name__=='__main__':
    # To get an incomprehensible mess of garbage logged directly to stdout
    # whenever you run something, run this function
    setup_logging_to_console()
    
    # example for the MolCleaner
    mol_cleaner = MolCleaner(record_keeper=[])
    
    testA = Dataset('./resources/dima.csv', smiles_col='smiles')

    testA.data[testA.mol_col] = testA.data[testA.mol_col].apply(mol_cleaner.apply_all_transformations)

    # Example for the Curator
    # clean_mols() is basically just nicer syntax for the MolCleaner example
    # It removes anything that became None as well
    curator = Curator()
    testA.data = curator.clean_mols(testA)

    # to inspect the record keeper, which is just a list of strings
    # (I hope to have a better solution soon)
    curator.record_keeper
