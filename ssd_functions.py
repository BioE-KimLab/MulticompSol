import os
import json 
import sys
from datetime import datetime
import random
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit,GroupKFold,GroupShuffleSplit,LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

rand_seed = 0
random.seed(rand_seed)
np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa

import nfp

import rdkit
import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcLabuteASA
from rdkit.Chem.GraphDescriptors import Chi0v, Chi1v, BalabanJ

### data processing functions
def get_solvent_system(row):
    if type(row['can_smiles_solvent2']) == float:
        smi_2_placeholder = 'None'
    else:
        smi_2_placeholder = row['can_smiles_solvent2']

    if type(row['can_smiles_solvent3']) == float:
        smi_3_placeholder = 'None'
    else:
        smi_3_placeholder = row['can_smiles_solvent3']
    
    sorted_solvents = sorted([row['can_smiles_solvent1'], smi_2_placeholder, smi_3_placeholder])
    return f"{sorted_solvents[0]}/{sorted_solvents[1]}/{sorted_solvents[2]}"

def process_all_data(data):
    data.reset_index(inplace=True)
    print("\nData after loading:\n\t",data.shape, data.columns.shape,list(data.columns))

    cols_chosen = ['can_smiles_solute', 'can_smiles_solvent1', 'can_smiles_solvent2', 'can_smiles_solvent3', 'mol_frac_solvent1', 'mol_frac_solvent2', 'mol_frac_solvent3', 'T_K', 'DGsolv', 'DGsolv_converted']
    data = data[cols_chosen]

    print(f"Columns chosen:{list(data.columns)}")

    data['tag'] = data['DGsolv'].apply(lambda x: 'converted' if pd.isna(x) else 'exp')

    cols_to_dropna = ['can_smiles_solute', 'can_smiles_solvent1', 'mol_frac_solvent1']

    print(f"Dropping datapoints w/ NaN SMILES or mole fractions...")
    print(f"Columns to dropna: {cols_to_dropna}")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset = cols_to_dropna) 
    print("\tAfter:\t",data.shape)

    dummy_smi = 'C'
    print(f"Using dummy SMILES of '{dummy_smi}'...")

    nan_smiles2 = data.loc[data.can_smiles_solvent2.isna() == True, :]
    if nan_smiles2.shape[0] > 0:
        print(f"\tReplacing {nan_smiles2.shape[0]} missing solvent 2 SMILES with '{dummy_smi}'")
        data.loc[(data.can_smiles_solvent2.isna()), 'can_smiles_solvent2'] = dummy_smi
        print(f"Done.")
    else:
        print("Could not find any NaN SMILES to replace for solvent 2. Shape:",nan_smiles2.shape)

    nan_smiles3 = data.loc[data.can_smiles_solvent3.isna() == True,:]
    if nan_smiles3.shape[0] > 0:
        print(f"\tReplacing {nan_smiles3.shape[0]} missing solvent 3 SMILES with '{dummy_smi}'")
        data.loc[(data.can_smiles_solvent3.isna()), 'can_smiles_solvent3'] = dummy_smi
        print(f"Done.")
    else:
        print("Could not find any NaN SMILES to replace for solvent 3. Shape:",nan_smiles3.shape)

    data['target'] = data.apply(lambda row: row['DGsolv_converted'] if row['tag'] == 'converted' else row['DGsolv'], axis=1)

    target = 'target'
    print(f"Dropping datapoints w/ NaN {target}...")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset=[target])
    print("\tAfter:\t", data.shape)

    target_const = tf.constant(list(data[target]))
    target_str = f"{target}_constant"
    data[target_str] = target_const

    data['solvent_system'] = data.apply(get_solvent_system, axis=1)

    return data

def process_cosmors_data(data):
    data.reset_index(inplace=True)
    print("\nData after loading:\n\t",data.shape, data.columns.shape,list(data.columns))

    cols_chosen = ['can_smiles_solute', 'can_smiles_solvent1', 'can_smiles_solvent2', 'can_smiles_solvent3', 'mol_frac_solvent1', 'mol_frac_solvent2', 'mol_frac_solvent3', 'T_K', 'DGsolv_cosmors']
    data = data[cols_chosen]

    data.mol_frac_solvent2.fillna(0, inplace=True)
    data.mol_frac_solvent3.fillna(0, inplace=True)
    
    print(f"Columns chosen:{list(data.columns)}")

    data['tag'] = 'cosmors' # exp / converted / cosmors
    # data['DGsolv'].apply(lambda x: 'converted' if pd.isna(x) else 'exp')

    cols_to_dropna = ['can_smiles_solute', 'can_smiles_solvent1', 'mol_frac_solvent1']

    print(f"Dropping datapoints w/ NaN SMILES or mole fractions...")
    print(f"Columns to dropna: {cols_to_dropna}")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset = cols_to_dropna) 
    print("\tAfter:\t",data.shape)

    dummy_smi = 'C'
    print(f"Using dummy SMILES of '{dummy_smi}'...")

    nan_smiles2 = data.loc[data.can_smiles_solvent2.isna() == True, :]
    if nan_smiles2.shape[0] > 0:
        print(f"\tReplacing {nan_smiles2.shape[0]} missing solvent 2 SMILES with '{dummy_smi}'")
        data.loc[(data.can_smiles_solvent2.isna()), 'can_smiles_solvent2'] = dummy_smi
        print(f"Done.")
    else:
        print("Could not find any NaN SMILES to replace for solvent 2. Shape:",nan_smiles2.shape)

    nan_smiles3 = data.loc[data.can_smiles_solvent3.isna() == True,:]
    if nan_smiles3.shape[0] > 0:
        print(f"\tReplacing {nan_smiles3.shape[0]} missing solvent 3 SMILES with '{dummy_smi}'")
        data.loc[(data.can_smiles_solvent3.isna()), 'can_smiles_solvent3'] = dummy_smi
        print(f"Done.")
    else:
        print("Could not find any NaN SMILES to replace for solvent 3. Shape:",nan_smiles3.shape)

    data['target'] = data['DGsolv_cosmors']
    # data.apply(lambda row: row['DGsolv_converted'] if row['tag'] == 'converted' else row['DGsolv'], axis=1)

    target = 'target'

    print(f"Target is DGsolv_cosmors")
    print(f"Dropping datapoints w/ NaN {target}...")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset=[target])
    print("\tAfter:\t", data.shape)

    target_const = tf.constant(list(data[target]))
    target_str = f"{target}_constant"
    data[target_str] = target_const

    data['solvent_system'] = data.apply(get_solvent_system, axis=1)

    return data

### Train Test Split Functions
def atom_features(atom):
    atom_type = namedtuple('Atom', ['totalHs', 'symbol', 'aromatic', 'fc', 'ring_size'])
    return str((atom.GetTotalNumHs(),
                atom.GetSymbol(),
                atom.GetIsAromatic(),
                atom.GetFormalCharge(), # 220829
                nfp.preprocessing.features.get_ring_size(atom, max_size=6)
               ))

def bond_features(bond, flipped=False):
    bond_type = namedtuple('Bond', ['bond_type', 'ring_size', 'symbol_1', 'symbol_2'])

    if not flipped:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

    else:
        atom1 = bond.GetEndAtom()
        atom2 = bond.GetBeginAtom()

    return str((bond.GetBondType(),
                nfp.preprocessing.features.get_ring_size(bond, max_size=6),
                atom1.GetSymbol(),
                atom2.GetSymbol()
               ))

def global_features(smiles, row, solute_or_solvent):
    mol = rdkit.Chem.MolFromSmiles(smiles)

    if solute_or_solvent == 'solute':
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         row['T_K'], # REMOVED TEMPORARILY 2024.03.20
                            
                         #row['atomcharges_min_solute'],
                         #row['atomcharges_max_solute'],
                         #row['dispersionenergies_avg_solute'],
                         #row['homo_lumo_gap_solute'] REMOVED 4.25.23
                         ])
    else:
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         row['T_K'], # REMOVED TEMPORARILY 2024.03.20
                            
                         #row['atomcharges_min_solvent'],
                         #row['atomcharges_max_solvent'],
                         #row['dispersionenergies_avg_solvent'],
                         #row['homo_lumo_gap_solvent'] REMOVED 4.25.23. Update hardcoded lengths as needed
                         ])
    
class CustomPreprocessor_NFPx2_ternary(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor_NFPx2_ternary, self).construct_feature_matrices(smiles, train)
        #features['mol_features'] = global_features(smiles)
        return features
    
    #output_signature = {**nfp.SmilesPreprocessor.output_signature,
    #                 **{'mol_features': tf.TensorSpec(shape=(1,), dtype=tf.float32) }}

    output_signature = {'atom_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solute': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        #'ratio_solute': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solute': tf.TensorSpec(shape=(5,), dtype=tf.float32), #! Change shape as needed
  

                        'atom_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent1': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent1': tf.TensorSpec(shape=(5,), dtype=tf.float32), #! Change shape as needed
 
                        'atom_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent2': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent2': tf.TensorSpec(shape=(5,), dtype=tf.float32),
                        
                        'atom_solvent3': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent3': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent3': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent3': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent3': tf.TensorSpec(shape=(5,), dtype=tf.float32),
                       
                        'connectivity_edges': tf.TensorSpec(shape=(None, 2), dtype=tf.int32), # Shape, name, dtype need to match layers.Input in main.py
                        #'weight_edges': tf.TensorSpec(shape=(None, 128), dtype=tf.float32), # Shape, name, dtype need to match layers.Input in main.py
                        'weight_edges': tf.TensorSpec(shape=(None, 4), dtype=tf.float32), # Shape, name, dtype need to match layers.Input in main.py
                        
                        
                        'temp_val': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'num_solvents': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                       } #! Change shape as needed
    
def create_tf_dataset_NFPx2_ternary_ShareWeights(df, preprocessor, sample_weight = 1.0, train=True, output_val_col = "DGsolv_constant"): 
    for _, row in df.iterrows():
        inputs_solute = preprocessor.construct_feature_matrices(row['can_smiles_solute'], train=train)
        inputs_solvent1 = preprocessor.construct_feature_matrices(row['can_smiles_solvent1'], train=train)
        inputs_solvent2 = preprocessor.construct_feature_matrices(row['can_smiles_solvent2'], train=train)
       
        if not train:
            one_data_sample_w = 1.0
        else:
            try:
                #one_data_sample_w = np.exp( -np.abs(row['DGsolv'] - row['DGsolv_cosmo']) / ( 1.9872E-3 * 298.15  ))
                one_data_sample_w = 1.0 
            except: #Exp dataset
                one_data_sample_w = 1.0
                
        mol_solute = rdkit.Chem.MolFromSmiles(row['can_smiles_solute'])
        mol_solvent1 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent1'])
        mol_solvent2 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent2'])

        
        output_val = tf.constant(row[output_val_col])
        
        
        """ 
          How do we choose which edge weights to represent?
          Possible solution: change order of edge connections (instead of 0->1, 0->2, 1->2)
              0->1, 1->2, 2->0.
              This way we can use each source node's properties as a sequential list, which makes sense in a way
        """
        if row['mol_frac_solvent3'] > 0:
            inputs_solvent3 = preprocessor.construct_feature_matrices(row['can_smiles_solvent3'], train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent3'])

            edge_weight = tf.constant([
                                        # Solute is source node
                                        [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                 MolWt(mol_solute),
                                        ], 

                                        # Solvent 1 is source node 
                                        [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                 MolWt(mol_solvent1),
                                        ],

                                        # Solvent 2 is source node 
                                        [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent2),
                                        ],
                
                                        # Solvent 3 is source node 
                                        [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent3),
                                        ],
                                      ])

            #edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]) # bidirectional
            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,2)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,3)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                
                                                
                                                # Solvent 1 is source node (1,0)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,2)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,3)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],


                                                # Solvent 2 is source node (2,0)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,1)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,3)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                
                
                                                # Solvent 3 is source node (3,0)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,1)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,2)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],

                                              ])
            stoich_vec_6edge = tf.constant([[1.0],[1.0],[1.0],
                                            [row['mol_frac_solvent1']], [row['mol_frac_solvent1']], [row['mol_frac_solvent1']],
                                            [row['mol_frac_solvent2']], [row['mol_frac_solvent2']], [row['mol_frac_solvent2']],
                                            [row['mol_frac_solvent3']], [row['mol_frac_solvent3']], [row['mol_frac_solvent3']],
                                                          ])
        elif row['mol_frac_solvent2'] > 0:
            solv3_fake_smi = 'C'
            #print("USING FAKE SMILES",solv3_fake_smi)
            inputs_solvent3 = preprocessor.construct_feature_matrices(solv3_fake_smi, train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(solv3_fake_smi)

            edge_weight = tf.constant([
                                        # Solute is source node
                                        [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                 MolWt(mol_solute),
                                        ], 

                                        # Solvent 1 is source node 
                                        [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                 MolWt(mol_solvent1),
                                        ],

                                        # Solvent 2 is source node 
                                        [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent2),
                                        ],
                
                                        # Solvent 3 is source node 
                                        [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent3),
                                        ],
                                      ])

            #edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]) # bidirectional
            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,2)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,3)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                
                                                
                                                # Solvent 1 is source node (1,0)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,2)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,3)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],


                                                # Solvent 2 is source node (2,0)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,1)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,3)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                
                
                                                # Solvent 3 is source node (3,0)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,1)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,2)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],

                                              ])
            stoich_vec_6edge = tf.constant([[1.0],[1.0],[1.0],
                                            [row['mol_frac_solvent1']], [row['mol_frac_solvent1']], [row['mol_frac_solvent1']],
                                            [row['mol_frac_solvent2']], [row['mol_frac_solvent2']], [row['mol_frac_solvent2']],
                                            [row['mol_frac_solvent3']], [row['mol_frac_solvent3']], [row['mol_frac_solvent3']],
                                                          ])
        else: # If not multi-solvent
            solv2_fake_smi = 'C'
            #print("USING FAKE SMILES 2",solv2_fake_smi)
            inputs_solvent2 = preprocessor.construct_feature_matrices(solv2_fake_smi, train=train)
            mol_solvent2 = rdkit.Chem.MolFromSmiles(solv2_fake_smi)
            
            solv3_fake_smi = 'C'
            #print("USING FAKE SMILES 3",solv3_fake_smi)
            inputs_solvent3 = preprocessor.construct_feature_matrices(solv3_fake_smi, train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(solv3_fake_smi)

            edge_weight = tf.constant([
                                        # Solute is source node
                                        [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                 MolWt(mol_solute),
                                        ], 

                                        # Solvent 1 is source node 
                                        [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                 MolWt(mol_solvent1),
                                        ],

                                        # Solvent 2 is source node 
                                        [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent2),
                                        ],
                
                                        # Solvent 3 is source node 
                                        [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                 MolWt(mol_solvent3),
                                        ],
                                      ])

            #edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]) # bidirectional
            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,2)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,3)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                
                                                
                                                # Solvent 1 is source node (1,0)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,2)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                                # Solvent 1 is source node (1,3)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],


                                                # Solvent 2 is source node (2,0)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,1)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,3)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                
                
                                                # Solvent 3 is source node (3,0)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,1)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],
                                                # Solvent 3 is source node (3,2)
                                                [CalcNumHBA(mol_solvent3), CalcNumHBD(mol_solvent3), BalabanJ(mol_solvent3), 
                                                         MolWt(mol_solvent3),
                                                ],

                                              ])
            stoich_vec_6edge = tf.constant([[1.0],[1.0],[1.0],
                                            [row['mol_frac_solvent1']], [row['mol_frac_solvent1']], [row['mol_frac_solvent1']],
                                            [row['mol_frac_solvent2']], [row['mol_frac_solvent2']], [row['mol_frac_solvent2']],
                                            [row['mol_frac_solvent3']], [row['mol_frac_solvent3']], [row['mol_frac_solvent3']],
                                                          ])
        edge_weight_bidirect_stoichbias = stoich_vec_6edge*edge_weight_bidirect #UNUSED!!
        #;;if _ == 1:
        #;;    print("DIVIDING BY EDGE WEIGHTS")
        #;;edge_weight = edge_weight*row['T_K']
        
        yield ({
                'atom_solute': inputs_solute['atom'],
                'bond_solute': inputs_solute['bond'],
                'connectivity_solute': inputs_solute['connectivity'],
                #'ratio_solute': tf.constant([row['ratio_solute']]),
                'mol_features_solute': global_features(row['can_smiles_solute'], row,'solute'),
                

                'atom_solvent1': inputs_solvent1['atom'],
                'bond_solvent1': inputs_solvent1['bond'],
                'connectivity_solvent1': inputs_solvent1['connectivity'],
                'ratio_solvent1': tf.constant([row['mol_frac_solvent1']]),
                'mol_features_solvent1': global_features(row['can_smiles_solvent1'], row,'solvent'),


                'atom_solvent2': inputs_solvent2['atom'],
                'bond_solvent2': inputs_solvent2['bond'],
                'connectivity_solvent2': inputs_solvent2['connectivity'],
                'ratio_solvent2': tf.constant([row['mol_frac_solvent2']]),
                'mol_features_solvent2': global_features(row['can_smiles_solvent2'], row, 'solvent'),
            
            
                'atom_solvent3': inputs_solvent3['atom'],
                'bond_solvent3': inputs_solvent3['bond'],
                'connectivity_solvent3': inputs_solvent3['connectivity'],
                'ratio_solvent3': tf.constant([row['mol_frac_solvent3']]),
                'mol_features_solvent3': global_features(row['can_smiles_solvent3'], row, 'solvent'),
            
            
                'connectivity_edges': edge_connectivity_bidirect, #was edge_connectivity
                'weight_edges': edge_weight_bidirect, #was edge_weight or edge_weight_bidirect
            
                'temp_val': tf.constant([row['T_K']]),
                'num_solvents':  tf.constant([2]),
                },
               #row['logS_constant'], 
               output_val, 
               one_data_sample_w) #! CHANGE. WHAT DOES one_data??
        
def get_train_test_NFPx2_AltSplits_Ternary(data, sample_weight, batch_size, fold_number, rand_seed, split_type='shuffle', output_val_col = "DGsolv_constant"):
    print(f"Using output val col of '{output_val_col}'.")
    # Change below if ternary or NOT deltaG!
    X_data = data[['can_smiles_solute', 'can_smiles_solvent1',
                       'can_smiles_solvent2', 'can_smiles_solvent3',
                       'mol_frac_solvent1', 'mol_frac_solvent2', 'mol_frac_solvent3',
                       'T_K', ]]
    y_data = data[output_val_col]
    
    num_splits = 5
    if split_type == 'shuffle':
        # Separates training+validation from test
        index_train_valid, index_test, dummy_train_valid, dummy_test = train_test_split(data.index,
                                    data.index, test_size = 0.1, random_state = rand_seed) 
        test_exp = data[data.index.isin(index_test)]
        train_valid = data[data.index.isin(index_train_valid)] # BOTH training and validation set
        kfold = KFold(n_splits = num_splits, shuffle = True, random_state = rand_seed) # Split training and validation into 10
        train_valid_split = list(kfold.split(train_valid))[fold_number] # Let's you choose chunk which is validation set (0-9)
        train_index, valid_index = train_valid_split
        train_exp = train_valid.iloc[train_index]
        valid_exp = train_valid.iloc[valid_index]

        train = train_exp
        valid = valid_exp
        test = test_exp
    elif split_type == 'group_kfold_solute':
        print("Performing Group KFold on Solute Smiles!")

        group_kfold = GroupShuffleSplit(n_splits=num_splits, train_size=0.9, random_state=0)
        groups_solute = list(data.can_smiles_solute)

        folds_dict = {}
        for i, (train_valid_index, test_index) in enumerate(group_kfold.split(X_data, y_data, groups_solute)):
                #print(f"\nFold {i}:")
                #NOTE tv = train_valid
                X_data_test = X_data.iloc[test_index, :]
                X_data_tv = X_data.iloc[train_valid_index, :]
                y_data_tv = y_data.iloc[train_valid_index]
                train_valid_size = X_data_tv.shape[0]
                X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_tv, y_data_tv, 
                                                                                          test_size=0.1, random_state=0)
                train_size = X_data_train.shape[0]
                valid_size = X_data_valid.shape[0]
                test_size = X_data_test.shape[0]
                
                train_ratio = train_size/(X_data.shape[0])
                valid_ratio = valid_size/(X_data.shape[0])
                test_ratio = test_size/(X_data.shape[0])
                
                idx_X_train = list(X_data_train.index)
                idx_X_valid = list(X_data_valid.index)
                idx_X_test = list(X_data_test.index)
                folds_dict[f"fold_{i}"] = {
                                        "train_ratio":train_ratio,
                                        "valid_ratio":valid_ratio,
                                        "test_ratio":test_ratio,
                                        "train_idx":idx_X_train, # USE LOC TO RETRIEVE FROM X_data!
                                        "valid_idx":idx_X_valid, # USE LOC TO RETRIEVE FROM X_data!
                                        "test_idx":idx_X_test, # USE LOC TO RETRIEVE FROM X_data!
                                        }


        fold_params = folds_dict[f"fold_{fold_number}"]
        print(f"  Using Fold {fold_number}")
        print(f"  Ratios: {fold_params['train_ratio']:.2f}/{fold_params['valid_ratio']:.2f}/{fold_params['test_ratio']:.2f}")
        train = data.loc[fold_params['train_idx'],:]
        valid = data.loc[fold_params['valid_idx'],:]
        test = data.loc[fold_params['test_idx'],:]
        train_exp = train
        valid_exp = valid
        test_exp = test
    elif split_type == 'group_kfold_solvent_system':
        print("Performing Group KFold on Solvent Systems!")

        group_kfold = GroupShuffleSplit(n_splits=num_splits, train_size=0.9, random_state=0)
        groups_solv_system = list(data.solvent_system)
        

        folds_dict = {}
        for i, (train_valid_index, test_index) in enumerate(group_kfold.split(X_data, y_data, groups_solv_system)):
                #print(f"\nFold {i}:")
                #NOTE tv = train_valid
                X_data_test = X_data.iloc[test_index, :]
                X_data_tv = X_data.iloc[train_valid_index, :]
                y_data_tv = y_data.iloc[train_valid_index]
                train_valid_size = X_data_tv.shape[0]
                X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_tv, y_data_tv, 
                                                                                          test_size=0.1, random_state=0)
                train_size = X_data_train.shape[0]
                valid_size = X_data_valid.shape[0]
                test_size = X_data_test.shape[0]
                
                train_ratio = train_size/(X_data.shape[0])
                valid_ratio = valid_size/(X_data.shape[0])
                test_ratio = test_size/(X_data.shape[0])
                
                idx_X_train = list(X_data_train.index)
                idx_X_valid = list(X_data_valid.index)
                idx_X_test = list(X_data_test.index)
                folds_dict[f"fold_{i}"] = {
                                        "train_ratio":train_ratio,
                                        "valid_ratio":valid_ratio,
                                        "test_ratio":test_ratio,
                                        "train_idx":idx_X_train, # USE LOC TO RETRIEVE FROM X_data!
                                        "valid_idx":idx_X_valid, # USE LOC TO RETRIEVE FROM X_data!
                                        "test_idx":idx_X_test, # USE LOC TO RETRIEVE FROM X_data!
                                        }
        fold_params = folds_dict[f"fold_{fold_number}"]
        print(f"  Using Fold {fold_number}")
        print(f"  Ratios: {fold_params['train_ratio']:.2f}/{fold_params['valid_ratio']:.2f}/{fold_params['test_ratio']:.2f}")
        train = data.loc[fold_params['train_idx'],:]
        valid = data.loc[fold_params['valid_idx'],:]
        test = data.loc[fold_params['test_idx'],:]
        train_exp = train
        valid_exp = valid
        test_exp = test
    elif split_type == 'group_kfold_leave_one_solute':
        print("Performing 'Leave One Solute Out' splitting!")

        logo_split = LeaveOneGroupOut()
        groups_solute = list(data_in.can_smiles_solute)

        folds_dict = {}
        # Iterate over all possible solutes left out (69 of them)
        # For each, create test by 
        for i, (train_valid_test_index, logo_index) in enumerate(logo_split.split(X_data, y_data, groups_solute)):
                #print("LOGO INDEX IS",logo_index)
                X_data_logo = X_data.iloc[logo_index, :]
                y_data_logo = y_data.iloc[logo_index]
                #group_kfold_1fold = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=0)
                X_data_tvt = X_data.iloc[train_valid_test_index, :]
                y_data_tvt = y_data.iloc[train_valid_test_index]
                train_valid_test_size = X_data_tvt.shape[0]
                
                # Split off test from train and validation (shuffle split)
                X_data_tv, X_data_test_nologo, y_data_tv, y_data_test_nologo = train_test_split(X_data_tvt, y_data_tvt, test_size=0.1, random_state=0)
                train_valid_size = X_data_tv.shape[0]
                
                # Split off validation from train (shuffle split)
                X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_tv, y_data_tv, test_size=0.1, random_state=0)
                
                X_data_test = pd.concat([X_data_test_nologo, X_data_logo])
                y_data_test = pd.concat([y_data_test_nologo, y_data_logo])

                logo_size = X_data_logo.shape[0]
                train_size = X_data_train.shape[0]
                valid_size = X_data_valid.shape[0]
                test_size = X_data_test.shape[0]

                logo_ratio = logo_size/(X_data.shape[0])
                train_ratio = train_size/(X_data.shape[0])
                valid_ratio = valid_size/(X_data.shape[0])
                test_ratio = test_size/(X_data.shape[0])

                idx_X_logo = list(X_data_logo.index)
                idx_X_train = list(X_data_train.index)
                idx_X_valid = list(X_data_valid.index)
                idx_X_test = list(X_data_test.index)
                folds_dict[f"fold_{i}"] = {
                                        "logo_ratio":logo_ratio,
                                        "train_ratio":train_ratio,
                                        "valid_ratio":valid_ratio,
                                        "test_ratio":test_ratio,
                                        "logo_idx":idx_X_logo,
                                        "train_idx":idx_X_train, # USE LOC TO RETRIEVE FROM X_data!
                                        "valid_idx":idx_X_valid, # USE LOC TO RETRIEVE FROM X_data!
                                        "test_idx":idx_X_test, # USE LOC TO RETRIEVE FROM X_data!
                                        }
        fold_params = folds_dict[f"fold_{fold_number}"]
        print(f"  Using Fold {fold_number} - LOGO!")
        print(f"  Ratios: {fold_params['train_ratio']:.2f}/{fold_params['valid_ratio']:.2f}/{fold_params['test_ratio']:.2f} logo: {fold_params['logo_ratio']:.2f}")
        print(f"  LOGO Indices:\t {fold_params['logo_idx']}")
        train = data.loc[fold_params['train_idx'],:]
        valid = data.loc[fold_params['valid_idx'],:]
        test = data.loc[fold_params['test_idx'],:]
        train_exp = train
        valid_exp = valid
        test_exp = test
    elif split_type == 'group_kfold_NumSolvShuffle':
        print("Performing Group Kfold Shuffle Split based on Single/Binary/Ternary label!")

        group_kfold = GroupShuffleSplit(n_splits=num_splits, train_size=0.9, random_state=0)
        groups_sin_bin_tern = list(data.sin_bin_term)

        folds_dict = {}
        for i, (train_valid_index, test_index) in enumerate(group_kfold.split(X_data, y_data, groups_sin_bin_tern)):
                #print(f"\nFold {i}:")
                #NOTE tv = train_valid
                X_data_test = X_data.iloc[test_index, :]
                X_data_tv = X_data.iloc[train_valid_index, :]
                y_data_tv = y_data.iloc[train_valid_index]
                train_valid_size = X_data_tv.shape[0]
                X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_tv, y_data_tv, 
                                                                                          test_size=0.1, random_state=0)
                train_size = X_data_train.shape[0]
                valid_size = X_data_valid.shape[0]
                test_size = X_data_test.shape[0]
                
                train_ratio = train_size/(X_data.shape[0])
                valid_ratio = valid_size/(X_data.shape[0])
                test_ratio = test_size/(X_data.shape[0])
                
                idx_X_train = list(X_data_train.index)
                idx_X_valid = list(X_data_valid.index)
                idx_X_test = list(X_data_test.index)
                folds_dict[f"fold_{i}"] = {
                                        "train_ratio":train_ratio,
                                        "valid_ratio":valid_ratio,
                                        "test_ratio":test_ratio,
                                        "train_idx":idx_X_train, # USE LOC TO RETRIEVE FROM X_data!
                                        "valid_idx":idx_X_valid, # USE LOC TO RETRIEVE FROM X_data!
                                        "test_idx":idx_X_test, # USE LOC TO RETRIEVE FROM X_data!
                                        }


        fold_params = folds_dict[f"fold_{fold_number}"]
        print(f"  Using Fold {fold_number}")
        print(f"  Ratios: {fold_params['train_ratio']:.2f}/{fold_params['valid_ratio']:.2f}/{fold_params['test_ratio']:.2f}")
        train = data.loc[fold_params['train_idx'],:]
        valid = data.loc[fold_params['valid_idx'],:]
        test = data.loc[fold_params['test_idx'],:]
        train_exp = train
        valid_exp = valid
        test_exp = test

        
    #! Ignore
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(len(train_exp), len(train) - len(train_exp), len(train))
    print(len(valid_exp), len(valid) - len(valid_exp), len(valid))
    print(len(test_exp), len(test) - len(test_exp), len(test))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # Set labels - after training, concatenate train/valid/test
    last_col_idx = len(list(train.columns)) - 1
    train.insert(loc=last_col_idx, column='Train/Valid/Test', value='Train')
    valid.insert(loc=last_col_idx, column='Train/Valid/Test', value='Valid')
    test.insert(loc=last_col_idx, column='Train/Valid/Test', value='Test')        

                                    
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(data.shape[0],"total datapoints")
    print("Train/valid/test datapoints:",train.shape[0], valid.shape[0], test.shape[0])
    print("Total datapoints vs train/valid/test sum:",data.shape[0] - (train.shape[0]+valid.shape[0]+test.shape[0])) 
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # Set labels - after training, concatenate train/valid/test
    train['Train/Valid/Test'] = 'Train'
    valid['Train/Valid/Test'] = 'Valid'
    test['Train/Valid/Test'] = 'Test'


    #### == Construct Preprocessor ==
    #* Beginning of actual GNN Part
    #* Preprocessor in GNN.py - converts SMILES into Atom and Bond Feature vectors
    #* output_signature: describes shape of input vectors
    #* tf.TensorSpec: dummy placeholder to specify shape (if shape is None, can be variable). If array given doesnt make shape (if shape not None), error
    #* Construct_feature_matrices: from NFP package (by peter st. john and YJ) https://github.com/NREL/nfp
    #* Converts SMILES to mol, gets atoms and bonds, and gets specific features requested for each.
    #* CustomPreprocessor inherits SMILESPreprocessor, and MolPreprocessor is superclass of SMILESPreprocessor
    #* Only purpose of CustomPreprocessor is to allow for global features (mol_features). However, this is *different* now! it is in create_tf_dataset.
    #* Could just use SMILESPreprocessor
    #* To use customized atom and bond features, can define customized feature function. Specify them when constructing CustomPreprocessor
    #* Other than atom +bond features, we have mol features. global_features function takes SMILES as input and calculates various rdkit features. 
    #* QM features are brought from dataframe rows. Can use different global features for solute and solvent
    preprocessor = CustomPreprocessor_NFPx2_ternary(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features,
        # MODIFIED
        # TO ADD GLOBAL FEATURES, LOOK AT gnn.py AND MODIFY MOL_FEATURES SHAPE AND VALS
        # DOESNT WORK NO MOLglobal_features=tf.constant([CalcNumHBA(g),
        # DOESNT WORK NO MOL                 CalcNumHBD(mol), 
        # DOESNT WORK NO MOL                 CalcLabuteASA(mol),
        # DOESNT WORK NO MOL                 CalcTPSA(mol)])
    )
    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")
    
    # Raul: to speed up, create preprocessor once and then load it


    train_all_smiles = list( set(list(train['can_smiles_solvent1']) + 
                                list(train['can_smiles_solvent2']) + 
                                list(train['can_smiles_solvent3']) +
                                 list(train['can_smiles_solute']) ) ) #! CHANGED: 
    
    print("TRAIN ALL SMILES", train_all_smiles)
    #* Initially preprocessor has no info about atom and bond types, so we iterate over all SMILES to get atom and bond classes
    #* Also have bond_tokenizer - shortening/classifying atom and bond feature info. Class #1/2/x will be converted to 64dim vector
    for smiles in train_all_smiles:
        print("SMI",smiles,"| Mol Obj:", rdkit.Chem.MolFromSmiles(smiles))
        preprocessor.construct_feature_matrices(smiles, train=True)
    
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    # Below must match output signature specified in gnn.py
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32), #! CHANGE
                        tf.TensorSpec(shape=(), dtype=tf.float32) #! Code   breaks if removed (??)
                        ) # deprecated. May need to modify other parts if removed
    

    #* Generates input data (incl. all atom,bond,global features defined in preprocessor)
    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(train, preprocessor, sample_weight, True, output_val_col = output_val_col), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    print("\nTRAIN DATA\n",train_data) # Seems fine based on main_copoly results

    valid_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(valid, preprocessor, sample_weight, False, output_val_col = output_val_col), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(test, preprocessor, sample_weight, False, output_val_col = output_val_col), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    train_data_final = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(train, preprocessor, sample_weight, False, output_val_col = output_val_col), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    dataframes = [train,valid,test]
    datasets = [train_data_final, train_data,valid_data,test_data]
    return preprocessor, output_signature, datasets, dataframes

### GNN Functions
def create_GNN_NFPx2_ShareWeights_ternary(model_name_in, train_data, valid_data, test_data, 
                       train_df, valid_df, test_df,
                       td_final,
                       preprocessor, output_signature, batch_size, sample_weight,
                       num_hidden, num_messages, learn_rate, num_epochs,
                       node_aggreg_op,
                       do_stoich_multiply,
                        dropout = 1.0e-10,
                        fold_number = 0,
                     output_val_col = "DGsolv_constant",
                     share_weights = "noshare",
                     split_type = "shuffle"
                      ):
    ##################
    #* Beginning of ACTUAL GNN OPERATORS
    features_dim = num_hidden
    print(f"\nFold number is: {fold_number} - previously defaulted to 0, check to match data split\n")
    #! Define input for solute/solvent
    # solute
    #* layers.Input is a placeholder to receive dict w/ atom_feature_matrix, bond_feature_matrix, connectivity, and global features
    # layers.Input is NOT a model layer, but a function to construct Tensors!
    #* In graph: connectivity gives source atom and target atom
    atom_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='atom_solute')
    bond_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='bond_solute')
    connectivity_Input_solute = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solute')
    global_Input_solute = layers.Input(shape=[5], dtype=tf.float32, name='mol_features_solute') #! Change shape as needed to fit global features
    #fake_ratio_Input_solute = layers.Input(shape=[1], dtype=tf.float32, name='fake_ratio_solute1')

    # Solvent 1
    atom_Input_solvent1 = layers.Input(shape=[None], dtype=tf.int32, name='atom_solvent1')
    bond_Input_solvent1 = layers.Input(shape=[None], dtype=tf.int32, name='bond_solvent1')
    connectivity_Input_solvent1 = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solvent1')
    ratio_Input_solvent1 = layers.Input(shape=[1], dtype=tf.float32, name='ratio_solvent1') # Necessary
    global_Input_solvent1 = layers.Input(shape=[5], dtype=tf.float32, name='mol_features_solvent1') #! Change shape as needed to fit global features

    # Solvent 2
    atom_Input_solvent2 = layers.Input(shape=[None], dtype=tf.int32, name='atom_solvent2')
    bond_Input_solvent2 = layers.Input(shape=[None], dtype=tf.int32, name='bond_solvent2')
    connectivity_Input_solvent2 = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solvent2')
    ratio_Input_solvent2 = layers.Input(shape=[1], dtype=tf.float32, name='ratio_solvent2') # Necessary
    global_Input_solvent2 = layers.Input(shape=[5], dtype=tf.float32, name='mol_features_solvent2') #! Change shape as needed to fit global features
    
    # Solvent 3
    atom_Input_solvent3 = layers.Input(shape=[None], dtype=tf.int32, name='atom_solvent3')
    bond_Input_solvent3 = layers.Input(shape=[None], dtype=tf.int32, name='bond_solvent3')
    connectivity_Input_solvent3 = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solvent3')
    ratio_Input_solvent3 = layers.Input(shape=[1], dtype=tf.float32, name='ratio_solvent3') # Necessary
    global_Input_solvent3 = layers.Input(shape=[5], dtype=tf.float32, name='mol_features_solvent3') #! Change shape as needed to fit global features


    connectivity_Input_edges = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_edges') # None here allows for variable # of edge connections
    #weight_Input_edges = layers.Input(shape=[None, 128], dtype=tf.float32, name='weight_edges') # None here allows for variable # of edge weights
    weight_Input_edges = layers.Input(shape=[None, 4], dtype=tf.float32, name='weight_edges') # None here allows for variable # of edge weights
    print("WEIGHT INPUT EDGES",weight_Input_edges)
    ######
    
    temp_Input = layers.Input(shape=[1], dtype=tf.float32, name='temp_val')
    num_solvents_Input = layers.Input(shape=[1], dtype=tf.float32, name='num_solvents')
    
    """ 
    TO ADD NEW Embedded feature (which is used at any point)
    1. Create layers.Input (make sure name matches dictionary in gnn create_tf_dataset function)
        This NEEDS to be part of the preprocessor output signature, create_tf_dataset dictionary, and list passing to model instantiation
    2. Create layers.Embedding (don't forget regularizer). 
        This embedding layer does NOT need to be part of the preprocessor output signature, create_tf_dataset dictionary, or model instantiation.
    3. 
    """
    
    #! Define embedding and dense layers for solute/solvent
    #* Embedding part converts atom/bond/global info into 64 dim vectors
    #* Take a group of vectors or strings, convert into a 64/128 dim vector
    #* Global feature: already numbers, so we only use a 1 dense layer
    #* ALSO adjusting these input embedding during training. 
    #* Embeddings are also trainable parameters. (e.g. relative weights 
    #* or importance of each vector/string for embedding)
    # Solute
    atom_state_solute = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solute)
    bond_state_solute = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solute)
    global_state_solute = layers.Dense(features_dim, activation='relu')(global_Input_solute)
   
    # Solvent 1
    atom_state_solvent1 = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solvent1', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solvent1)
    bond_state_solvent1 = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solvent1', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solvent1)
    global_state_solvent1 = layers.Dense(features_dim, activation='relu')(global_Input_solvent1)

    # Solvent 2
    atom_state_solvent2 = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solvent2', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solvent2)
    bond_state_solvent2 = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solvent2', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solvent2)
    global_state_solvent2 = layers.Dense(features_dim, activation='relu')(global_Input_solvent2)
      
    # Solvent 3
    atom_state_solvent3 = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solvent3', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solvent3)
    bond_state_solvent3 = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solvent3', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solvent3)
    global_state_solvent3 = layers.Dense(features_dim, activation='relu')(global_Input_solvent3)
    
    
    
    # Edge Weight Feature Embedding
    #NOTE Do we need to StandardScale the data here? Embedding was doing it.
    weight_embedding_edges = layers.Dense(features_dim, name='weight_Embedding_edges')(weight_Input_edges)
    #weight_embedding_edges = layers.Embedding(3, features_dim,
    #                             name='weight_Embedding_edges', mask_zero=True,
    #                             embeddings_regularizer='l2')(weight_Input_edges)
    print("WEIGHT EMBEDDING EDGES",weight_embedding_edges)
    
    global_state_GNN2_initial = layers.Dense(features_dim, activation='relu')(temp_Input)
    
    
    
    #* Create message passing blocks
    #! This is the shaded box in YJ's GNN figure.
    #* If curious about layers, go to GNN.py and look at message_block function
    #* Message blocks for solute and solvent independtly
    #* Embedding layer vs dense layer: Embedding operation has no info about neighboring atoms/bonds.
    #* When they pass through 5x message passing layers, the atom/bond/global features interact
    #* Atom/bond/global features transfer messages to each other
    #* This is a message passing neural network
    if share_weights.count("all") > 0:
        #    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

        print("USING SHARED WEIGHTS (SIAMESE) GNN FOR COMBINED FIRST GNN")
        print("\n !! GNN1: Sharing Solute and Solvents Weights.\n")
        Layers_In = [
                    #    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],
        ]

        for i in range(num_messages):
            surv_prob_i = 1.0
            # If on first loop, print atom/bond/global/connectivity states
            if i == 0:
                print('atom:\n\t',atom_state_solute,'\nbond:\n\t',bond_state_solute,'\nglobal:\n\t',
                        global_state_solute,'\nconnectivity\n\t',connectivity_Input_solute)
            atom_states_all_in =   [atom_state_solute, atom_state_solvent1, atom_state_solvent2, atom_state_solvent3]
            bond_states_all_in =   [bond_state_solute, bond_state_solvent1, bond_state_solvent2, bond_state_solvent3]
            global_states_all_in = [global_state_solute, global_state_solvent1, global_state_solvent2, global_state_solvent3]
            connectivity_Input_all_in = [connectivity_Input_solute, connectivity_Input_solvent1, connectivity_Input_solvent2, connectivity_Input_solvent3]


            atom_states_out, bond_states_out, global_states_out = message_block_solu_solv_shared_ternary(atom_states_all_in, 
                                                                                bond_states_all_in, 
                                                                                global_states_all_in, 
                                                                                connectivity_Input_all_in, 
                                                                                features_dim, i, 
                                                                                #dropout, 
                                                                                Layers_In)

            atom_state_solute, atom_state_solvent1, atom_state_solvent2, atom_state_solvent3 = atom_states_out
            bond_state_solute, bond_state_solvent1, bond_state_solvent2, bond_state_solvent3 = bond_states_out
            global_state_solute, global_state_solvent1, global_state_solvent2, global_state_solvent3 = global_states_out
    elif share_weights.count("all_solt_last") > 0:
        #    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

        print("USING SHARED WEIGHTS (SIAMESE) GNN FOR COMBINED FIRST GNN")
        print("\n !! GNN1: Sharing Solute and Solvents Weights. Solute goes last! \n")
        Layers_In = [
                    #    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],
        ]

        for i in range(num_messages):
            surv_prob_i = 1.0
            # If on first loop, print atom/bond/global/connectivity states
            if i == 0:
                print('atom:\n\t',atom_state_solute,'\nbond:\n\t',bond_state_solute,'\nglobal:\n\t',
                        global_state_solute,'\nconnectivity\n\t',connectivity_Input_solute)
            atom_states_all_in =   [atom_state_solute, atom_state_solvent1, atom_state_solvent2, atom_state_solvent3]
            bond_states_all_in =   [bond_state_solute, bond_state_solvent1, bond_state_solvent2, bond_state_solvent3]
            global_states_all_in = [global_state_solute, global_state_solvent1, global_state_solvent2, global_state_solvent3]
            connectivity_Input_all_in = [connectivity_Input_solute, connectivity_Input_solvent1, connectivity_Input_solvent2, connectivity_Input_solvent3]


            atom_states_out, bond_states_out, global_states_out = message_block_solu_solv_shared_ternary_SoluteLast(atom_states_all_in, 
                                                                                bond_states_all_in, 
                                                                                global_states_all_in, 
                                                                                connectivity_Input_all_in, 
                                                                                features_dim, i, 
                                                                                #dropout, 
                                                                                Layers_In)

            atom_state_solute, atom_state_solvent1, atom_state_solvent2, atom_state_solvent3 = atom_states_out
            bond_state_solute, bond_state_solvent1, bond_state_solvent2, bond_state_solvent3 = bond_states_out
            global_state_solute, global_state_solvent1, global_state_solvent2, global_state_solvent3 = global_states_out
    elif share_weights.count("solvs") > 0:
        print("USING SHARED WEIGHTS (SIAMESE) GNN FOR COMBINED FIRST GNN")
        print("\n !!GNN 1: Solvents share weights, Solute has independent weights.\n")
        Layers_In = [
                    #    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],

                    [layers.GlobalAveragePooling1D(),                         #    atom_av,
                                layers.Dense(features_dim, activation='relu'),   #    global_embed_dense1,
                                layers.Dense(features_dim), layers.Add(),        #    global_embed_dense2, global_residcon,
                                nfp.EdgeUpdate(dropout = dropout), layers.Add(), #    nfp_edgeupdate, bond_residcon, 
                                nfp.NodeUpdate(dropout = dropout), layers.Add()  #    nfp_nodeupdate, atom_residcon
                    ],
        ]
        for i in range(num_messages):
            surv_prob_i = 1.0
            # If on first loop, print atom/bond/global/connectivity states 
            if i == 0:
                print('atom:\n\t',atom_state_solute,'\nbond:\n\t',bond_state_solute,'\nglobal:\n\t',
                        global_state_solute,'\nconnectivity\n\t',connectivity_Input_solute)

            # SOLUTE
            atom_state_solute, bond_state_solute, global_state_solute = message_block(atom_state_solute, 
                                                        bond_state_solute, 
                                                        global_state_solute, 
                                                        connectivity_Input_solute, 
                                                        features_dim, i, dropout, surv_prob_i)

            # SOLVENTS
            atom_states_solvents =   [atom_state_solvent1, atom_state_solvent2, atom_state_solvent3]
            bond_states_solvents =   [bond_state_solvent1, bond_state_solvent2, bond_state_solvent3]
            global_states_solvents = [global_state_solvent1, global_state_solvent2, global_state_solvent3]
            connectivity_Input_solvents = [connectivity_Input_solvent1, connectivity_Input_solvent2, connectivity_Input_solvent3]
            atom_states_out, bond_states_out, global_states_out = message_block_solv_shared_only_ternary(atom_states_solvents, 
                                                                                bond_states_solvents, 
                                                                                global_states_solvents, 
                                                                                connectivity_Input_solvents, 
                                                                                features_dim, i, 
                                                                                #dropout, 
                                                                                Layers_In)
            atom_state_solvent1, atom_state_solvent2, atom_state_solvent3 = atom_states_out
            bond_state_solvent1, bond_state_solvent2, bond_state_solvent3 = bond_states_out
            global_state_solvent1, global_state_solvent2, global_state_solvent3 = global_states_out
    
    elif share_weights.count("noshare") > 0:#
        print("\n !!GNN 1: NOT Sharing any weights!\n")
        for i in range(num_messages):
            surv_prob_i = 1.0
            if i == 0:
                print('atom:\n\t',atom_state_solute,'\nbond:\n\t',bond_state_solute,'\nglobal:\n\t',
                        global_state_solute,'\nconnectivity\n\t',connectivity_Input_solute)
            atom_state_solute, bond_state_solute, global_state_solute = message_block(atom_state_solute, 
                                                                                        bond_state_solute, 
                                                                                        global_state_solute, 
                                                                                        connectivity_Input_solute, 
                                                                                        features_dim, i, dropout, surv_prob_i) #dropout at readout only?
                                                                                        #features_dim, i, args.dropout, surv_prob_i)
            atom_state_solvent1, bond_state_solvent1, global_state_solvent1 = message_block(atom_state_solvent1, 
                                                                                        bond_state_solvent1, 
                                                                                        global_state_solvent1, 
                                                                                        connectivity_Input_solvent1, 
                                                                                        features_dim, i, dropout, surv_prob_i) #dropout at readout only?
                                                                                        #features_dim, i, args.dropout, surv_prob_i)


            atom_state_solvent2, bond_state_solvent2, global_state_solvent2 = message_block(atom_state_solvent2,  
                                                                                    bond_state_solvent2,  
                                                                                    global_state_solvent2,  
                                                                                    connectivity_Input_solvent2, 
                                                                                    features_dim, i, dropout, surv_prob_i) #dropout at readout only?
            atom_state_solvent3, bond_state_solvent3, global_state_solvent3 = message_block(atom_state_solvent3,  
                                                                                bond_state_solvent3, 
                                                                                global_state_solvent3, 
                                                                                connectivity_Input_solvent3, 
                                                                                features_dim, i, dropout, surv_prob_i) #dropout at readout only?
                                                                                #features_dim, i, args.dropout, surv_prob_i)
    else:
        print(f"Could not find a valid share weights option (Given: {share_weights})")
        print("Valid options are ['all', 'all_solt_last', 'solvs', 'noshare']")


    
    

    node_states_initial = tf.stack([global_state_solute, 
                                           global_state_solvent1, 
                                           global_state_solvent2,
                                           global_state_solvent3], axis=1)
    print("NODE STATES INITIAL",node_states_initial)
    print("!!! Stoichiometry Fix Level: FIXED")
    stoich_vec_6edge = tf.stack([ratio_Input_solvent1 + ratio_Input_solvent2 + ratio_Input_solvent3,
                                 ratio_Input_solvent1 + ratio_Input_solvent2 + ratio_Input_solvent3,
                                 ratio_Input_solvent1 + ratio_Input_solvent2 + ratio_Input_solvent3,
                                 
                                 ratio_Input_solvent1, 
                                 ratio_Input_solvent1,
                                 ratio_Input_solvent1,
                             
                                 ratio_Input_solvent2, 
                                 ratio_Input_solvent2,
                                 ratio_Input_solvent2,
                                 
                                 ratio_Input_solvent3,
                                 ratio_Input_solvent3,
                                 ratio_Input_solvent3,
                                              ], 
                                 axis=1)
    print("STOICH VEC 6 EDGES",stoich_vec_6edge)
    edge_states_initial = weight_embedding_edges #*stoich_vec_6edge
    print("EDGE STATES INITIAL",edge_states_initial)
    edge_connectivity = connectivity_Input_edges
    

    # Multiple node state by stoichiometry 
    # Note that this is extremely similar to the previous version of model with concatenation
    if do_stoich_multiply == 'before_dense':
        stoich_multiply_tensor = tf.stack([ratio_Input_solvent1 + ratio_Input_solvent2 + ratio_Input_solvent3, 
                                            ratio_Input_solvent1,
                                            ratio_Input_solvent2, 
                                            ratio_Input_solvent3], axis=1)
        print("\nSTOICH MULTIPLY TENSOR (BEFORE DENSE)",stoich_multiply_tensor)
        node_states_in = node_states_initial * stoich_multiply_tensor
    else:
        print("Not multiplying stoich before...")
        node_states_in = node_states_initial
        
    node_state, edge_state = message_block_SolvGraph(node_states_in, 
                                                     edge_states_initial,
                                                    edge_connectivity,)
        
    #;;node_state, edge_state, global_state_out = message_block_SolvGraph_global(node_states_in, edge_states_initial, 
    #;;                                                                          global_state_GNN2_initial, 
    #;;                                                                          edge_connectivity, features_dim=128,
    #;;                                                                           i=1) # Features_dim
    
    
    if do_stoich_multiply == 'after_dense':
        stoich_multiply_tensor = tf.stack([ratio_Input_solvent1 + ratio_Input_solvent2 + ratio_Input_solvent3, 
                                            ratio_Input_solvent1, 
                                            ratio_Input_solvent2, 
                                            ratio_Input_solvent3], axis=1)
        print("\nSTOICH MULTIPLY TENSOR (AFTER DENSE)",stoich_multiply_tensor)
        node_state = node_state * stoich_multiply_tensor
    else:
        if do_stoich_multiply != 'before_dense':
            print("WARNING: Stoichiometry multiplication of node states not given.\n\tNOT INCORPORATING STOICHIOMETRY!!!")
            
            
    #node_state = tf.math.reduce_sum(node_state)
    print("NODE STATE",node_state)
    

    
    # Average or sum node state over solute-solvent1-solvent2 nodes
    # if not specified to be sum, does average.
    if node_aggreg_op == 'sum':
        node_state = tf.math.reduce_sum(node_state, axis=1)
    else:  # node_aggreg_op == 'mean':
        node_state = tf.math.reduce_mean(node_state, axis=1)
    
    
    prediction = layers.Dense(1)(node_state)




    # DO NOT DELETE BELOW COMMENTS!
    # NOTE THAT ORDER MATTERS HERE. MUST MATCH GNN (create_tensor_dataset func yield)
    # NOTE I USE A DIFFERENT ORDER THAN YJ FOR GLOBAL FEATURES!! 
    input_tensors = [
                    # SOLUTE
                    atom_Input_solute,
                    bond_Input_solute, 
                    connectivity_Input_solute, 
                    global_Input_solute,

                    # SOLVENT 1
                    atom_Input_solvent1,
                    bond_Input_solvent1, 
                    connectivity_Input_solvent1, 
                    ratio_Input_solvent1,
                    global_Input_solvent1,

                    # SOLVENT 2
                    atom_Input_solvent2,
                    bond_Input_solvent2,
                    connectivity_Input_solvent2,
                    ratio_Input_solvent2,
                    global_Input_solvent2,
            
                    # SOLVENT 3
                    atom_Input_solvent3,
                    bond_Input_solvent3,
                    connectivity_Input_solvent3,
                    ratio_Input_solvent3,
                    global_Input_solvent3,
    
                    # EDGE MATRICES
                    connectivity_Input_edges,
                    weight_Input_edges,
        
                    # GNN 2 GLOBAL
                    temp_Input,
                    num_solvents_Input,
                    ]
    print("INPUT TENSORS\n",*input_tensors,sep='\n\t-\t')
    model = tf.keras.Model(input_tensors, [prediction])

    
    #! KEEP model.summary()
    ###############################
    ## MODEL COMPILATION AND TRAINING 

    # Below from R Perez Soto
    #;;@tf.function
    #;;def rsquare(y_true,y_pred): 
    #;;    sse = tf.reduce_sum(tf.math.square(y_true - y_pred),axis=1)
    #;;    tse = tf.reduce_sum(tf.math.square(y_true - tf.reduce_mean(y_true, axis=1,
    #;;                                                        keepdims=True)), axis=1) 
    #;;    r2_score = tf.reduce_mean(1 - tf.math.divide(sse, tse))
    #;;    return r2_score

    model_metrics = [
            #rsquare,
           tf.keras.metrics.mean_absolute_error,
           tf.keras.metrics.mean_squared_error,
           #tf.keras.metrics.RootMeanSquaredError, # uses y_true and y_pred, not prediction and labels
            ]
        
    model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(), 
            optimizer=tf.keras.optimizers.Adam(learn_rate), 
            metrics=model_metrics,
            )

    model_path = "model_files/"+model_name_in+"/best_model.h5"
    checkpoint = ModelCheckpoint(model_path, monitor="val_loss",\
                                 verbose=2, save_best_only = True, mode='auto', period=1,
                                #custom
                                )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                patience=5, min_lr=1e-6)
    tensorboard = keras.callbacks.TensorBoard(log_dir=("model_files/"+model_name_in), histogram_freq=1)
    csv_log_callback = keras.callbacks.CSVLogger("model_files/" + model_name_in + "/all_training_metrics.log")

    callbacks_lst = [checkpoint,
                        reduce_lr,
                        tensorboard,
                        csv_log_callback
                        ]
    hist = model.fit(train_data,
                     validation_data=valid_data,
                     epochs=num_epochs,
                     verbose=1, 
                     callbacks = callbacks_lst)
                     #use_multiprocessing = True, workers = 24

    model.load_weights(model_path)
    
    #train_data_final = tf.data.Dataset.from_generator(
    #    lambda: create_tf_dataset(train, preprocessor, sample_weight, False), output_signature=output_signature)\
    #    .cache()\
    #    .padded_batch(batch_size=batch_size)\
    #    .prefetch(tf.data.experimental.AUTOTUNE)
    print("Using argument of td_final...",)

    #train_data_final = train_data_final.as_numpy_iterator() # CHANGED!
    train_results = model.predict(td_final).squeeze()
    print("Shape train results",model.predict(td_final).shape,)
    valid_results = model.predict(valid_data).squeeze()
    test_results = model.predict(test_data).squeeze()

    
    #NOTE Check for accuracy
    #! CHANGE BELOW
    #! Removed tf.argmax from all calls below -01.26.24
    train_labels = tf.convert_to_tensor(list(train_df[output_val_col]), dtype=tf.float32)
    print("Train labels",train_labels)
    #print("Train labels from train directly",train[output_val_col])
    valid_labels = tf.convert_to_tensor(list(valid_df[output_val_col]), dtype=tf.float32)
    test_labels = tf.convert_to_tensor(list(test_df[output_val_col]), dtype=tf.float32)



    train_df['predicted'] = train_results
    valid_df['predicted'] = valid_results
    test_df['predicted'] =  test_results

    diff_pred_train = train_labels - train_results
    print("DIFF TRAIN VS PRED TRAIN",diff_pred_train)
    

    
    mae_train = np.abs(train_labels - train_results).mean()
    mae_valid = np.abs(valid_labels - valid_results).mean()
    mae_test = np.abs(test_labels - test_results).mean()
    # I should be doing this with numpy
    rmse_train = sklearn.metrics.mean_squared_error(y_true=train_labels, y_pred=train_results, squared=False) # Will fail in scitkit-learn >1.3
    rmse_valid = sklearn.metrics.mean_squared_error(y_true=valid_labels, y_pred=valid_results, squared=False) # Will fail in scikit-learn >1.3
    rmse_test = sklearn.metrics.mean_squared_error(y_true=test_labels, y_pred=test_results, squared=False) # Will fail in scikit-learn > 1.3
    r2_train = sklearn.metrics.r2_score(y_true=train_labels,
                                        y_pred=train_results)
    r2_valid = sklearn.metrics.r2_score(y_true=valid_labels,
                                        y_pred=valid_results)
    r2_test = sklearn.metrics.r2_score(y_true=test_labels,
                                        y_pred=test_results)

    #fold_number = 0
    print(f"Fold number is: {fold_number} with {split_type} - previously defaulted to 0, check to match data split")
    print(len(train_df),len(valid_df),len(test_df))
    mae_string = f"{mae_train:.2f},{mae_valid:.2f},{mae_test:.2f}"
    rmse_string = f"{rmse_train:.2f},{rmse_valid:.2f},{rmse_test:.2f}"
    r2_string = f"{r2_train:.2f},{r2_valid:.2f},{r2_test:.2f}"
    print("MAEs:\t",mae_string)
    print("RMSEs:\t",rmse_string)
    print("R2s:\t",r2_string)
    with open("model_files/" + model_name_in + "/results.txt",'w') as f:
        f.write("MAEs:\t" + mae_string)
        f.write('\n')
        f.write("RMSEs:\t" + rmse_string)
        f.write('\n')
        f.write("R2s:\t" + r2_string)
    pd.concat([train_df, valid_df, test_df], ignore_index=True).to_csv('model_files/' + model_name_in +'/kfold_'+str(fold_number)+'.csv',index=False)
    preprocessor.to_json("model_files/"+ model_name_in  +"/preprocessor.json")
    return model, pd.concat([train_df,valid_df,test_df])  

def message_block(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, dropout = 0.0, surv_prob = 1.0):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    global_state = original_global_state
    
    global_state_update = layers.GlobalAveragePooling1D()(atom_state)

    global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)
    '''
    if dropout > 0:
        global_state_update = layers.Dropout(dropout)(global_state_update)
    '''

    global_state_update = layers.Dense(features_dim)(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)
    '''
    if dropout > 0:
        global_state_update = layers.Dropout(dropout)(global_state_update)
    '''

    global_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_global_state, global_state_update])
    '''
    if surv_prob == 1.0:
        global_state = layers.Add()([original_global_state, global_state_update])
    else:
        global_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_global_state, global_state_update])
    '''

    #################
    new_bond_state = nfp.EdgeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([original_bond_state, new_bond_state])
    '''
    if surv_prob == 1.0:
        bond_state = layers.Add()([original_bond_state, new_bond_state])
    else:
        bond_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_bond_state, new_bond_state])
    '''

    #################
    new_atom_state = nfp.NodeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([original_atom_state, new_atom_state])
    '''
    if surv_prob == 1.0:
        atom_state = layers.Add()([original_atom_state, new_atom_state])
    else:
        atom_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_atom_state, new_atom_state])
    '''
    
    return atom_state, bond_state, global_state

def message_block_SolvGraph(original_node_state, original_edge_state,
                            #original_global_state, 
                            connectivity, #;;features_dim, i, 
                            dropout = 0.0, surv_prob = 1.0):
    
    node_state = original_node_state
    edge_state = original_edge_state
    #;;global_state = original_global_state
    print("ORIGINAL NODE STATE",original_node_state)
    print("ORIGINAL EDGE STATE",original_edge_state)
    #;;global_state_update = layers.GlobalAveragePooling1D()(node_state)
    #;;global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    #;;global_state_update = layers.Dropout(dropout)(global_state_update)
    #;;global_state_update = layers.Dense(features_dim)(global_state_update)
    #;;global_state_update = layers.Dropout(dropout)(global_state_update)
    #;;global_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_global_state, global_state_update])

    #################
    #new_edge_state = nfp.EdgeUpdate(dropout = dropout)([node_state, edge_state, connectivity, global_state])
    new_edge_state = nfp.EdgeUpdate(dropout = dropout)([node_state, edge_state, connectivity])
    edge_state = layers.Add()([original_edge_state, new_edge_state])

    #################
    #new_node_state = nfp.NodeUpdate(dropout = dropout)([node_state, edge_state, connectivity, global_state])
    new_node_state = nfp.NodeUpdate(dropout = dropout)([node_state, edge_state, connectivity])
    node_state = layers.Add()([original_node_state, new_node_state])

    
    return node_state, edge_state

def message_block_solv_shared_only_ternary(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solv1, atom_state_solv2, atom_state_solv3 = original_atom_state
    bond_state_solv1, bond_state_solv2, bond_state_solv3 = original_bond_state
    global_state_solv1, global_state_solv2, global_state_solv3 = original_global_state
    connectivity_solv1, connectivity_solv2, connectivity_solv3 = connectivity

    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]
   
    #solvent 1
    global_state_update = atom_av(atom_state_solv1)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solv1 = global_residcon([global_state_solv1, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solv1, bond_state_solv1, connectivity_solv1, global_state_solv1])
    bond_state_solv1 = bond_residcon([bond_state_solv1, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solv1, bond_state_solv1, connectivity_solv1, global_state_solv1])
    atom_state_solv1 = atom_residcon([atom_state_solv1, new_atom_state])
    
    #solvent 2
    global_state_update = atom_av(atom_state_solv2)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solv2 = global_residcon([global_state_solv2, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solv2, bond_state_solv2, connectivity_solv2, global_state_solv2])
    bond_state_solv2 = bond_residcon([bond_state_solv2, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solv2, bond_state_solv2, connectivity_solv2, global_state_solv2])
    atom_state_solv2 = atom_residcon([atom_state_solv2, new_atom_state])

    #solvent 3
    global_state_update = atom_av(atom_state_solv3)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solv3 = global_residcon([global_state_solv3, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solv3, bond_state_solv3, connectivity_solv3, global_state_solv3])
    bond_state_solv3 = bond_residcon([bond_state_solv3, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solv3, bond_state_solv3, connectivity_solv3, global_state_solv3])
    atom_state_solv3 = atom_residcon([atom_state_solv3, new_atom_state])

    #Return 
    atom_state =   [atom_state_solv1, atom_state_solv2, atom_state_solv3]
    bond_state =   [bond_state_solv1, bond_state_solv2, bond_state_solv3]
    global_state = [global_state_solv1, global_state_solv2, global_state_solv3]

    return atom_state, bond_state, global_state