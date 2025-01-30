import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
from collections import namedtuple

from tensorflow.keras import layers
import nfp

import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcLabuteASA
from rdkit.Chem.GraphDescriptors import Chi0v, Chi1v, BalabanJ
import pandas as pd
import json

import tensorflow_addons as tfa
from itertools import combinations
from itertools import permutations


# Define general custom preprocessors as a subclass of nfp SmilesPreprocessor
# Output signature should match the layers.Input in train_... for the model.
# None shape here allows for flexible sizing of tensors
class CustomPreprocessor_NFPx2(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor_NFPx2, self).construct_feature_matrices(smiles, train)
        return features
    
    # NOTE: For all below, the Shape, name, dtype need to match layers.Input in train_....py
    output_signature = {'atom_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solute': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'mol_features_solute': tf.TensorSpec(shape=(5,), dtype=tf.float32), 
  

                        'atom_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent1': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent1': tf.TensorSpec(shape=(5,), dtype=tf.float32), 
 
                        'atom_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent2': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent2': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent2': tf.TensorSpec(shape=(5,), dtype=tf.float32),
                       
                        'connectivity_edges': tf.TensorSpec(shape=(None, 2), dtype=tf.int32), # Shape, name, dtype need to match layers.Input in train_....py
                        'weight_edges': tf.TensorSpec(shape=(None, 4), dtype=tf.float32), # Shape, name, dtype need to match layers.Input in main.py
                        
                        
                        'temp_val': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'num_solvents': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                       } 
    

# Define ternary custom preprocessors as a subclass of nfp SmilesPreprocessor
# Same as CustomPreprocessor, but adds solvent3 features (atom_solvent3, bond_solvent3, etc.)
# Output signature should match the layers.Input in train_... for the model.
# None shape here allows for flexible sizing of tensors
class CustomPreprocessor_NFPx2_ternary(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor_NFPx2_ternary, self).construct_feature_matrices(smiles, train)
        return features
    

    output_signature = {'atom_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solute': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'mol_features_solute': tf.TensorSpec(shape=(5,), dtype=tf.float32), 
  

                        'atom_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solvent1': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'ratio_solvent1': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'mol_features_solvent1': tf.TensorSpec(shape=(5,), dtype=tf.float32), 
 
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
                       
                        'connectivity_edges': tf.TensorSpec(shape=(None, 2), dtype=tf.int32), # Shape, name, dtype need to match layers.Input in train_....py
                        'weight_edges': tf.TensorSpec(shape=(None, 4), dtype=tf.float32), # Shape, name, dtype need to match layers.Input in train_....py
                        
                        
                        'temp_val': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        'num_solvents': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                       } 
    
    
    

# Construct atom features for all GNN models
def atom_features(atom):
    atom_type = namedtuple('Atom', ['totalHs', 'symbol', 'aromatic', 'fc', 'ring_size'])
    return str((atom.GetTotalNumHs(),
                atom.GetSymbol(),
                atom.GetIsAromatic(),
                atom.GetFormalCharge(),
                nfp.preprocessing.features.get_ring_size(atom, max_size=6)
               ))

# Construct bond features for all GNN models
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

# Construct global features for all GNN models
def global_features(smiles, row, solute_or_solvent):
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Allows for different descriptors used for solute/solvent (currently the same)
    if solute_or_solvent == 'solute':
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         row['T_K'], 
                         ])
    else:
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         row['T_K'], 
                         ])

    
    

        
        
# Input data creation function for binary models without weight sharing
def create_tf_dataset_NFPx2(df, preprocessor, sample_weight = 1.0, train=True, output_val_col = "DGsolv_constant"): 
    for _, row in df.iterrows():
        inputs_solute = preprocessor.construct_feature_matrices(row['can_smiles_solute'], train=train)
        inputs_solvent1 = preprocessor.construct_feature_matrices(row['can_smiles_solvent1'], train=train)
        inputs_solvent2 = preprocessor.construct_feature_matrices(row['can_smiles_solvent2'], train=train)
        if not train:
            one_data_sample_w = 1.0
        else:
            try:
                one_data_sample_w = 1.0 
            except: 
                one_data_sample_w = 1.0
                
                
        mol_solute = rdkit.Chem.MolFromSmiles(row['can_smiles_solute'])
        mol_solvent1 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent1'])
        mol_solvent2 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent2'])
        
        output_val = tf.constant(row[output_val_col])
        
        if row['mol_frac_solvent2'] > 0:
            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]) # bidirectional graph connectivity
            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,2)
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


                                                # Solvent 2 is source node (2,0)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,1)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                              ])
            # Below: 
            stoich_vec_6edge = tf.constant([[1.0],[1.0],
                                            [row['mol_frac_solvent1']], [row['mol_frac_solvent1']],
                                            [row['mol_frac_solvent2']], [row['mol_frac_solvent2']],
                                                          ])
        else: 
            edge_connectivity_bidirect = tf.constant([(0, 1), 
                                                      (1, 0), 
                                                     ])
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 

                                                # Solvent 1 is source node (1,0)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                              ])
            stoich_vec_6edge = tf.constant([[1.0],
                                            [row['mol_frac_solvent1']], 
                                                          ])
        
        yield ({
                'atom_solute': inputs_solute['atom'],
                'bond_solute': inputs_solute['bond'],
                'connectivity_solute': inputs_solute['connectivity'],
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
                
                'connectivity_edges': edge_connectivity_bidirect, 
                'weight_edges': edge_weight_bidirect, 
            
                'temp_val': tf.constant([row['T_K']]),
                'num_solvents':  tf.constant([2]),
                },
               output_val, 
               one_data_sample_w)

# Input data creation function for ternary models without weight sharing
def create_tf_dataset_NFPx2_ternary(df, preprocessor, sample_weight = 1.0, train=True, output_val_col = "DGsolv_constant"): 
    for _, row in df.iterrows():
        inputs_solute = preprocessor.construct_feature_matrices(row['can_smiles_solute'], train=train)
        inputs_solvent1 = preprocessor.construct_feature_matrices(row['can_smiles_solvent1'], train=train)
        inputs_solvent2 = preprocessor.construct_feature_matrices(row['can_smiles_solvent2'], train=train)
        inputs_solvent3 = preprocessor.construct_feature_matrices(row['can_smiles_solvent3'], train=train)
        if not train:
            one_data_sample_w = 1.0
        else:
            try:
                one_data_sample_w = 1.0 
            except: #Exp dataset
                one_data_sample_w = 1.0
                
        mol_solute = rdkit.Chem.MolFromSmiles(row['can_smiles_solute'])
        mol_solvent1 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent1'])
        mol_solvent2 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent2'])
        mol_solvent3 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent3'])
        
        output_val = tf.constant(row[output_val_col])
        edge_connectivity = tf.constant([(0, 1), (1, 2), (2, 0)]) #monodirectional
        
        
        if row['mol_frac_solvent3'] > 0:


            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 
            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
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
            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]) # bidirectional
            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 
                                                # Solute is source node (0,2)
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


                                                # Solvent 2 is source node (2,0)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                                # Solvent 2 is source node (2,1)
                                                [CalcNumHBA(mol_solvent2), CalcNumHBD(mol_solvent2), BalabanJ(mol_solvent2), 
                                                         MolWt(mol_solvent2),
                                                ],
                                              ])
            stoich_vec_6edge = tf.constant([[1.0],[1.0],
                                            [row['mol_frac_solvent1']], [row['mol_frac_solvent1']],
                                            [row['mol_frac_solvent2']], [row['mol_frac_solvent2']],
                                                          ])
        else: # If not multi-solvent
            edge_connectivity_bidirect = tf.constant([(0, 1), 
                                                      (1, 0), 
                                                     ]) # bidirectional
            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
            edge_weight_bidirect = tf.constant([
                                                # Solute is source node (0,1)
                                                [CalcNumHBA(mol_solute), CalcNumHBD(mol_solute), BalabanJ(mol_solute),
                                                         MolWt(mol_solute),
                                                ], 

                                                # Solvent 1 is source node (1,0)
                                                [CalcNumHBA(mol_solvent1), CalcNumHBD(mol_solvent1), BalabanJ(mol_solvent1), 
                                                         MolWt(mol_solvent1),
                                                ],
                                              ])
            stoich_vec_6edge = tf.constant([[1.0],
                                            [row['mol_frac_solvent1']], 
                                                          ])
        
        
        yield ({
                'atom_solute': inputs_solute['atom'],
                'bond_solute': inputs_solute['bond'],
                'connectivity_solute': inputs_solute['connectivity'],
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
            
            
                'connectivity_edges': edge_connectivity_bidirect, 
                'weight_edges': edge_weight_bidirect, 
            
                'temp_val': tf.constant([row['T_K']]),
                'num_solvents':  tf.constant([2]),
                },
               output_val, 
               one_data_sample_w) 
    
# Input data creation function for ternary models with weight sharing
def create_tf_dataset_NFPx2_ternary_ShareWeights(df, preprocessor, sample_weight = 1.0, train=True, output_val_col = "DGsolv_constant"): 
    for _, row in df.iterrows():
        inputs_solute = preprocessor.construct_feature_matrices(row['can_smiles_solute'], train=train)
        inputs_solvent1 = preprocessor.construct_feature_matrices(row['can_smiles_solvent1'], train=train)
        inputs_solvent2 = preprocessor.construct_feature_matrices(row['can_smiles_solvent2'], train=train)
       
        if not train:
            one_data_sample_w = 1.0
        else:
            try:
                one_data_sample_w = 1.0 
            except: #Exp dataset
                one_data_sample_w = 1.0
                
        mol_solute = rdkit.Chem.MolFromSmiles(row['can_smiles_solute'])
        mol_solvent1 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent1'])
        mol_solvent2 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent2'])

        
        output_val = tf.constant(row[output_val_col])
        
        
        if row['mol_frac_solvent3'] > 0:
            inputs_solvent3 = preprocessor.construct_feature_matrices(row['can_smiles_solvent3'], train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(row['can_smiles_solvent3'])

            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 

            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
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
            solv3_dummy_smi = 'C'
            inputs_solvent3 = preprocessor.construct_feature_matrices(solv3_dummy_smi, train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(solv3_dummy_smi)

            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 
            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
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
            solv2_dummy_smi = 'C'
            inputs_solvent2 = preprocessor.construct_feature_matrices(solv2_dummy_smi, train=train)
            mol_solvent2 = rdkit.Chem.MolFromSmiles(solv2_dummy_smi)
            
            solv3_dummy_smi = 'C'
            inputs_solvent3 = preprocessor.construct_feature_matrices(solv3_dummy_smi, train=train)
            mol_solvent3 = rdkit.Chem.MolFromSmiles(solv3_dummy_smi)

            edge_connectivity_bidirect = tf.constant([(0, 1), (0, 2), (0, 3), # bidirectional, 4 component
                                                          (1, 0), (1, 2), (1, 3), 
                                                          (2, 0), (2, 1), (2, 3), 
                                                          (3, 0), (3, 1), (3, 2)]) 

            # edge_weight_bidirect is the initial node state for all nodes in the solute-solvent graph before embedding
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
               
        yield ({
                'atom_solute': inputs_solute['atom'],
                'bond_solute': inputs_solute['bond'],
                'connectivity_solute': inputs_solute['connectivity'],
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
            
            
                'connectivity_edges': edge_connectivity_bidirect,
                'weight_edges': edge_weight_bidirect,
            
                'temp_val': tf.constant([row['T_K']]),
                'num_solvents':  tf.constant([2]),
                },
               output_val, 
               one_data_sample_w)


        




# Generic message block for a given atom/bond/global state and connectivity, with optional parameters 
def message_block(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, dropout = 0.0, surv_prob = 1.0):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    global_state = original_global_state
    
    global_state_update = layers.GlobalAveragePooling1D()(atom_state)

    global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)

    global_state_update = layers.Dense(features_dim)(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)

    global_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_global_state, global_state_update])

    #################
    new_bond_state = nfp.EdgeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([original_bond_state, new_bond_state])

    #################
    new_atom_state = nfp.NodeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([original_atom_state, new_atom_state])
    
    return atom_state, bond_state, global_state


# Intermolecular GNN message block. Global update is unused here.
def message_block_SolvGraph(original_node_state, original_edge_state,
                            #original_global_state, 
                            connectivity, #;;features_dim, i, 
                            dropout = 0.0, surv_prob = 1.0):
    
    node_state = original_node_state
    edge_state = original_edge_state
    print("ORIGINAL NODE STATE",original_node_state)
    print("ORIGINAL EDGE STATE",original_edge_state)

    #################
    new_edge_state = nfp.EdgeUpdate(dropout = dropout)([node_state, edge_state, connectivity])
    edge_state = layers.Add()([original_edge_state, new_edge_state])

    #################
    new_node_state = nfp.NodeUpdate(dropout = dropout)([node_state, edge_state, connectivity])
    node_state = layers.Add()([original_node_state, new_node_state])

    
    return node_state, edge_state

#### BINARY SOLVENT MESSAGE BLOCKS ####
# Intramolecular GNN message block, with shared solute and solvent weights  (binary)
def message_block_solu_solv_shared(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solute, atom_state_solv1, atom_state_solv2 = original_atom_state
    bond_state_solute, bond_state_solv1, bond_state_solv2 = original_bond_state
    global_state_solute, global_state_solv1, global_state_solv2 = original_global_state
    connectivity_solute, connectivity_solv1, connectivity_solv2 = connectivity

    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

    #solute
    global_state_update = atom_av(atom_state_solute)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solute = global_residcon([global_state_solute, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    bond_state_solute = bond_residcon([bond_state_solute, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    atom_state_solute = atom_residcon([atom_state_solute, new_atom_state])
   
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

    #Return 
    atom_state =   [atom_state_solute, atom_state_solv1, atom_state_solv2]
    bond_state =   [bond_state_solute, bond_state_solv1, bond_state_solv2]
    global_state = [global_state_solute, global_state_solv1, global_state_solv2]

    return atom_state, bond_state, global_state


# Intramolecular GNN message block, with shared solvent weights (binary solvents)
def message_block_solv_shared_only(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solv1, atom_state_solv2 = original_atom_state
    bond_state_solv1, bond_state_solv2 = original_bond_state
    global_state_solv1, global_state_solv2 = original_global_state
    connectivity_solv1, connectivity_solv2 = connectivity

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

    #Return 
    atom_state =   [atom_state_solv1, atom_state_solv2]
    bond_state =   [bond_state_solv1, bond_state_solv2]
    global_state = [global_state_solv1, global_state_solv2]

    return atom_state, bond_state, global_state

#### TERNARY SOLVENT MESSAGE BLOCKS ####
# Intramolecular GNN message block, with shared solute and solvent weights (ternary solvents)
def message_block_solu_solv_shared_ternary(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solute, atom_state_solv1, atom_state_solv2, atom_state_solv3 = original_atom_state
    bond_state_solute, bond_state_solv1, bond_state_solv2, bond_state_solv3 = original_bond_state
    global_state_solute, global_state_solv1, global_state_solv2, global_state_solv3 = original_global_state
    connectivity_solute, connectivity_solv1, connectivity_solv2, connectivity_solv3 = connectivity

    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

    #solute
    global_state_update = atom_av(atom_state_solute)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solute = global_residcon([global_state_solute, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    bond_state_solute = bond_residcon([bond_state_solute, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    atom_state_solute = atom_residcon([atom_state_solute, new_atom_state])
   
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
    atom_state =   [atom_state_solute, atom_state_solv1, atom_state_solv2, atom_state_solv3]
    bond_state =   [bond_state_solute, bond_state_solv1, bond_state_solv2, bond_state_solv3]
    global_state = [global_state_solute, global_state_solv1, global_state_solv2, global_state_solv3]

    return atom_state, bond_state, global_state

# Intramolecular GNN message block, with shared solute and solvent weights (ternary solvents) and solute message passing occurring before solvent message passing
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

    atom_state =   [atom_state_solv1, atom_state_solv2, atom_state_solv3]
    bond_state =   [bond_state_solv1, bond_state_solv2, bond_state_solv3]
    global_state = [global_state_solv1, global_state_solv2, global_state_solv3]

    return atom_state, bond_state, global_state



# Intramolecular GNN message block, with shared solute and solvent weights (ternary solvents) and solute message passing following solvent message passing
def message_block_solu_solv_shared_ternary_SoluteLast(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solute, atom_state_solv1, atom_state_solv2, atom_state_solv3 = original_atom_state
    bond_state_solute, bond_state_solv1, bond_state_solv2, bond_state_solv3 = original_bond_state
    global_state_solute, global_state_solv1, global_state_solv2, global_state_solv3 = original_global_state
    connectivity_solute, connectivity_solv1, connectivity_solv2, connectivity_solv3 = connectivity

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

    #solute
    global_state_update = atom_av(atom_state_solute)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solute = global_residcon([global_state_solute, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    bond_state_solute = bond_residcon([bond_state_solute, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    atom_state_solute = atom_residcon([atom_state_solute, new_atom_state])
   
    
    #Return 
    atom_state =   [atom_state_solute, atom_state_solv1, atom_state_solv2, atom_state_solv3]
    bond_state =   [bond_state_solute, bond_state_solv1, bond_state_solv2, bond_state_solv3]
    global_state = [global_state_solute, global_state_solv1, global_state_solv2, global_state_solv3]

    return atom_state, bond_state, global_state

