import tensorflow as tf

import os
import json 
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import random
from pathlib import Path
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from gnn_multisol import *
import nfp
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit,GroupKFold,GroupShuffleSplit,StratifiedGroupKFold,LeaveOneGroupOut
import rdkit

rand_seed = 0
random.seed(rand_seed)
np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/gpu:0"
else:
    device = "/cpu:0"
print("Using device:",device)

print("\n\n\n\nTraining Ternary Subgraph GNN - start at MACHINE TIME",datetime.now())






def data_process_MixSolDBv4_CT(target,
                            data_subset_str = 'b',
                            ):
    data_frac = 1.0

    data_dir = Path.cwd()/"data/csvs"

    print(f"\n ! Using DB MixSol_v4 Combined Target (CT) with target {target}")
    data_in_path = data_dir/"MixSolDB_v4_CombinedTarget.csv"
    print("\nDATA IN",data_in_path,"\n")

    data = pd.read_csv(data_in_path, low_memory=False)
    data = data.reset_index() #! Necessary for later code, shouldn't impact index assignments at all.
    print("\nData after loading:\n\t",data.shape, data.columns.shape,list(data.columns))

  
    # Below added because DGSolv_constant already exists in MixSolv4CT
    target_str_constant = f"{target}_constant"

    print("Renamed columns:\n\t",list(data.columns))
    data[target_str_constant] = data[target_str_constant].astype(float)



    single_solv_mask = (data.mol_frac_solvent1 != 0) & (
                            data.mol_frac_solvent2 == 0) & (
                            data.mol_frac_solvent3 == 0)

    binary_solv_mask = (data.mol_frac_solvent1 != 0) & (
                            data.mol_frac_solvent2 != 0) & (
                            data.mol_frac_solvent3 == 0)

    ternary_solv_mask = (data.mol_frac_solvent1 != 0) & (
                            data.mol_frac_solvent2 != 0) & (
                            data.mol_frac_solvent3 != 0)


    data_lookup_dict = {
        "s": single_solv_mask,
        "b": binary_solv_mask,
        "t": ternary_solv_mask,
        "s+b": single_solv_mask | binary_solv_mask,
        "s+t": single_solv_mask | ternary_solv_mask,
        "b+t": binary_solv_mask | ternary_solv_mask,
        "s+b+t": single_solv_mask | binary_solv_mask | ternary_solv_mask,
    }
    print(f"\n\n~~~~~~Chosen data subset is '{data_subset_str}'.")
    data = data.loc[data_lookup_dict[data_subset_str], :]


    if (data_subset_str.lower().count('t') > 0):
        print("\n\n!^|^|^! Keeping ternary datapoints !^|^|^!")
    
    
    cols_dropna_dict = {
        "s": ["can_smiles_solute", "can_smiles_solvent1", "mol_frac_solvent1"],

        "b": ["can_smiles_solute", "can_smiles_solvent1", "can_smiles_solvent2",
                                    "mol_frac_solvent1", "mol_frac_solvent2"],

        "t": ["can_smiles_solute", "can_smiles_solvent1", "can_smiles_solvent2",
                                    "can_smiles_solvent3",
                                    "mol_frac_solvent1", "mol_frac_solvent2",
                                    "mol_frac_solvent3"],


        "s+b": ["can_smiles_solute", "can_smiles_solvent1", "mol_frac_solvent1"],

        "s+t": ["can_smiles_solute", "can_smiles_solvent1", "mol_frac_solvent1"],

        "b+t": ["can_smiles_solute", "can_smiles_solvent1", "can_smiles_solvent2",
                                    "mol_frac_solvent1", "mol_frac_solvent2"],

        "s+b+t": ["can_smiles_solute", "can_smiles_solvent1", "mol_frac_solvent1"],
    }
    cols_to_dropna = cols_dropna_dict[data_subset_str]

    print(f"Dropping datapoints w/ NaN SMILES or mole fractions...")
    print(f"Columns to dropna: {cols_to_dropna}")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset = cols_to_dropna)
    print("\tAfter:\t",data.shape)


    print(f"! SKIPPING dummy SMILES assignment...")

    print(f"Dropping datapoints w/ NaN {target}...")
    print("\tBefore:\t", data.shape)
    data = data.dropna(subset=target_str_constant)
    print("\tAfter:\t", data.shape)


    def get_solvent_system(row):
        if type(row['can_smiles_solvent2']) == float:
            smi_2_placeholder = "None"
        else:
            smi_2_placeholder = row['can_smiles_solvent2']

        if type(row['can_smiles_solvent3']) == float:
            smi_3_placeholder = "None"
        else:
            smi_3_placeholder = row['can_smiles_solvent3']

        sorted_solvents = sorted([row['can_smiles_solvent1'], smi_2_placeholder,
                                  smi_3_placeholder
                                  ])
        return f"{sorted_solvents[0]}/{sorted_solvents[1]}/{sorted_solvents[2]}"


    data["solvent_system"] = data.apply(get_solvent_system, axis=1)
    return data


def data_process_all(database, target, data_subset_str):
    db_options = {
                    "MixSol_v4_CT": ["logS", "DGsolv"],
                  }

    if database not in list(db_options.keys()):
        print(f"\nWARNING!!!: Could not find database '{database}' in database options. Options:\n{db_options}")
        sys.exit()

    avail_targets = db_options[database]
    if target not in avail_targets:
        print(f"\nWARNING!!!: Could not find target '{target}' for database '{database}'. Options:\n{db_options}")
        sys.exit()

    db_functions = {
                    "MixSol_v4_CT": data_process_MixSolDBv4_CT,
                  }

    data = db_functions[database](target, data_subset_str = data_subset_str,)
    return data





def get_train_test_NFPx2_AltSplits_Ternary(data, sample_weight, batch_size, fold_number, split_type='shuffle', output_val_col = "DGsolv_constant"):
    print(f"Using output val col of '{output_val_col}'.")
    X_data = data[['can_smiles_solute', 'can_smiles_solvent1',
                       'can_smiles_solvent2', 'can_smiles_solvent3',
                       'mol_frac_solvent1', 'mol_frac_solvent2', 'mol_frac_solvent3',
                       'T_K', ]]
    y_data = data[output_val_col]
    
    num_splits = 5
    if split_type == 'shuffle':
        # Separates training+validation from test
        index_train_valid, index_test, dummy_train_valid, dummy_test = train_test_split(data['index'],
                                    data['index'], test_size = 0.1, random_state = rand_seed) 
        test_exp = data[data['index'].isin(index_test)]
        train_valid = data[data['index'].isin(index_train_valid)] # BOTH training and validation set
        kfold = KFold(n_splits = num_splits, shuffle = True, random_state = rand_seed) # Split training and validation into 10
        train_valid_split = list(kfold.split(train_valid))[fold_number] # Lets you choose chunk which is validation set (0-9)
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
        # Iterate over all possible solutes left out
        for i, (train_valid_test_index, logo_index) in enumerate(logo_split.split(X_data, y_data, groups_solute)):
                X_data_logo = X_data.iloc[logo_index, :]
                y_data_logo = y_data.iloc[logo_index]
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


    #### == Construct Preprocessor ==
    preprocessor = CustomPreprocessor_NFPx2_ternary(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features,
    )
    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")
    
    train_all_smiles = list( set(list(train['can_smiles_solvent1']) + 
                                list(train['can_smiles_solvent2']) + 
                                list(train['can_smiles_solvent3']) +
                                 list(train['can_smiles_solute']) ) )
    
    #* Initially preprocessor has no info about atom and bond types, so we iterate over all SMILES to get atom and bond classes
    #* Also have bond_tokenizer - shortening/classifying atom and bond feature info. Class #1/2/x will be converted to 64dim vector
    for smiles in train_all_smiles:
        preprocessor.construct_feature_matrices(smiles, train=True)
    
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    # Below must match output signature specified in gnn.py
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32), 
                        tf.TensorSpec(shape=(), dtype=tf.float32), 
                        ) 
    

    #* Generates input data (incl. all atom,bond,global features defined in preprocessor)
    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(train, preprocessor, sample_weight, True, output_val_col = output_val_col), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    print("\nTRAIN DATA\n",train_data) 

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




def create_concat_GNN_ShareWeights_ternary(model_name_in, train_data, valid_data, test_data, 
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
                      ):
    ##################
    #* Beginning of GNN Construction and Operations
    features_dim = num_hidden
    print(f"\nFold number is: {fold_number} - previously defaulted to 0, check to match data split\n")
    
    #! Define input for solute/solvent
    #* layers.Input is a placeholder to receive dict w/ atom_feature_matrix, bond_feature_matrix, connectivity, and global features
    # layers.Input is NOT a model layer, but a function to construct Tensors!

    # Solute
    atom_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='atom_solute')
    bond_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='bond_solute')
    connectivity_Input_solute = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solute')
    global_Input_solute = layers.Input(shape=[5], dtype=tf.float32, name='mol_features_solute') #! Change shape as needed to fit global features


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
    weight_Input_edges = layers.Input(shape=[None, 4], dtype=tf.float32, name='weight_edges') # None here allows for variable # of edge weights
    print("WEIGHT INPUT EDGES",weight_Input_edges)
    ######
    
    temp_Input = layers.Input(shape=[1], dtype=tf.float32, name='temp_val')
    num_solvents_Input = layers.Input(shape=[1], dtype=tf.float32, name='num_solvents')

    
    #! Define embedding and dense layers for solute/solvent

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
    weight_embedding_edges = layers.Dense(features_dim, name='weight_Embedding_edges')(weight_Input_edges)
    print("WEIGHT EMBEDDING EDGES",weight_embedding_edges)
    
    global_state_GNN2_initial = layers.Dense(features_dim, activation='relu')(temp_Input)
    
    
    
    #* Create message passing blocks
    #* If curious about layers, go to GNN.py and look at message_block function
    if share_weights.count("all") > 0:
        print("\n !! GNN1: Sharing Solute and Solvents Weights.\n")
        Layers_In = [

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
                                                                                Layers_In)

            atom_state_solute, atom_state_solvent1, atom_state_solvent2, atom_state_solvent3 = atom_states_out
            bond_state_solute, bond_state_solvent1, bond_state_solvent2, bond_state_solvent3 = bond_states_out
            global_state_solute, global_state_solvent1, global_state_solvent2, global_state_solvent3 = global_states_out
    elif share_weights.count("all_solt_last") > 0:
        print("\n !! GNN1: Sharing Solute and Solvents Weights. Solute goes last! \n")
        Layers_In = [
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
        print("\n !!GNN 1: Solvents share weights, Solute has independent weights.\n")
        Layers_In = [
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
                                                                                        features_dim, i, dropout, surv_prob_i)
            atom_state_solvent1, bond_state_solvent1, global_state_solvent1 = message_block(atom_state_solvent1, 
                                                                                        bond_state_solvent1, 
                                                                                        global_state_solvent1, 
                                                                                        connectivity_Input_solvent1, 
                                                                                        features_dim, i, dropout, surv_prob_i)


            atom_state_solvent2, bond_state_solvent2, global_state_solvent2 = message_block(atom_state_solvent2,  
                                                                                    bond_state_solvent2,  
                                                                                    global_state_solvent2,  
                                                                                    connectivity_Input_solvent2, 
                                                                                    features_dim, i, dropout, surv_prob_i)
            atom_state_solvent3, bond_state_solvent3, global_state_solvent3 = message_block(atom_state_solvent3,  
                                                                                bond_state_solvent3, 
                                                                                global_state_solvent3, 
                                                                                connectivity_Input_solvent3, 
                                                                                features_dim, i, dropout, surv_prob_i)
    else:
        print(f"Could not find a valid share weights option (Given: {share_weights})")
        print("Valid options are ['all', 'all_solt_last', 'solvs', 'noshare']")


    
    

    X1 = tf.tile(ratio_Input_solvent1, [1,features_dim]) 
    X2 = tf.tile(ratio_Input_solvent2, [1,features_dim])
    X3 = tf.tile(ratio_Input_solvent3, [1,features_dim])

    solvent1_vector = tf.math.multiply(X1, global_state_solvent1)
    solvent2_vector = tf.math.multiply(X2, global_state_solvent2)
    solvent3_vector = tf.math.multiply(X3, global_state_solvent3)
    solvent12_vector = tf.concat([solvent1_vector, solvent2_vector], -1) 
    solvent12_vector = layers.Dense(features_dim, activation='relu')(solvent12_vector)
    solvent12_vector = layers.Dense(features_dim, activation='relu')(solvent12_vector)
    
    solvent123_vector = tf.concat([solvent12_vector, solvent3_vector], -1) 
    solvent123_vector = layers.Dense(features_dim, activation='relu')(solvent123_vector)
    solvent123_vector = layers.Dense(features_dim, activation='relu')(solvent123_vector)

    # Combine solvent vector with solute vector
    # prediction is final output
    readout_vector = tf.concat([solvent123_vector, global_state_solute], -1)
    readout_vector = layers.Dense(features_dim, activation='relu')(readout_vector)
    readout_vector = layers.Dense(features_dim)(readout_vector)
    prediction = layers.Dense(1)(readout_vector) 



    # NOTE THAT ORDER MATTERS HERE. MUST MATCH GNN (create_tensor_dataset func yield)
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

    
    model_metrics = [
           tf.keras.metrics.mean_absolute_error,
           tf.keras.metrics.mean_squared_error,
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
    tensorboard = keras.callbacks.TensorBoard(log_dir=("model_files/"+model_name_in))
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

    model.load_weights(model_path)
    
    print("Using argument of td_final...",)
    train_results = model.predict(td_final).squeeze()
    print("Shape train results",model.predict(td_final).shape,)
    valid_results = model.predict(valid_data).squeeze()
    test_results = model.predict(test_data).squeeze()

    
    train_labels = tf.convert_to_tensor(list(train_df[output_val_col]), dtype=tf.float32)
    print("Train labels",train_labels)
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
    rmse_train = sklearn.metrics.mean_squared_error(y_true=train_labels, y_pred=train_results, squared=False) # Will fail in scitkit-learn >1.3
    rmse_valid = sklearn.metrics.mean_squared_error(y_true=valid_labels, y_pred=valid_results, squared=False) # Will fail in scikit-learn >1.3
    rmse_test = sklearn.metrics.mean_squared_error(y_true=test_labels, y_pred=test_results, squared=False) # Will fail in scikit-learn > 1.3
    r2_train = sklearn.metrics.r2_score(y_true=train_labels,
                                        y_pred=train_results)
    r2_valid = sklearn.metrics.r2_score(y_true=valid_labels,
                                        y_pred=valid_results)
    r2_test = sklearn.metrics.r2_score(y_true=test_labels,
                                        y_pred=test_results)

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



#~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&~^&






if __name__ == '__main__':
    with tf.device(device):
        parser = ArgumentParser()
        parser.add_argument('-n', '--modelname', type=str, required=True, default='unnamed_model', help='Model name (REQ) - impacts save directory! (default=unnamed_model)')
        parser.add_argument('--db', type=str, default='MixSol_v4_CT', help='Database of binary solubilities to be used (default=MixSol_v1). Options: ["MixSol_v1" (logS, DGsolv), "MixSol_v3" (logS, DGsolv), "ComboDB_v1" (logS), "MixSol_v4" (logS, DGsolv)]')
        parser.add_argument('-t', '--target', type=str, default='DGsolv', help='Prediction target to be used ("DGsolv" or "logS", default=DGsolv). DGsolv not available for all databases and may throw an error!')
        parser.add_argument('-lr', '--lr', type=float, default=1.0e-4, help='Learning rate for training - note this is only an initial LR and further LR is specified in code. (default=1.0e-4)')
        parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for training. (default=1000)')
        parser.add_argument('-e', '--epochs', type=int, default=1000, help='# Epochs for training. (default=1000)')
        parser.add_argument('-m', '--num_messages', type=int, default=5, help='number of message-passing blocks (default=5)')
        parser.add_argument('-hid', '--num_hidden', type=int, default=128, help='number of nodes in hidden layers (default=128)')
        parser.add_argument('-f','--fold_number', type=int, default=0, help='Fold number for Kfold (default=0). Relevant for all splitting types).')
        parser.add_argument('-s', '--split_option', type=str, default='shuffle', help='Split Options (default=shuffle) \
                    0: Shuffle split ("shuffle"). ~80:10:10 split,\
                    1: Solute split ("solute"). ~80:10:10 split by unique solutes (10% of unique solutes in test). Actual data split %s vary.\
                    2: Solvent system split ("solvent_system"). ~80:10:10 split by unique solvent systems (10% of unique solvent systems in test). Actual data split %s vary')

        parser.add_argument('-w', '--sample_weight', type=float, default=1.0, help='Whether to use sample weights (default=1.0) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-d', '--dropout', type=float, default=1.0e-10, help='Dropout. (default=1.0e-10)')
        parser.add_argument('--surv_prob', type=float, default=1.0, help='Survival probability. default=1.0)')
        parser.add_argument('-o','--overwrite', type=bool, default=False, help='Whether to overwrite model files if model directory already exists (default=False).')
        parser.add_argument('--share_weights', type=str, default='noshare', help='How to share weights (default="noshare") \
                    Valid options are ["all", "all_solt_last", "solvs", "noshare"] \
                    0: Share solvent and solute weights ("all"). \
                    1: Share solvent and solute weights, solute last ("all_solt_last"). \
                    2: Share solvent weights, solute is indepedent. ("solvs"). \
                    3: Share no weights ("noshare"). \
                        '
                    )
        parser.add_argument('--data_subset', type=str, default='b+t', help='Data subset to use; cAsE SeNsiTIvE! (default="b+t") \
                    Valid options are ["s","b","t","s+b","s+t","b+t","s+b+t"] \
                    0: Only single solvent datapoints (dummy SMILES used!) ("s"). \
                    1: Only binary solvent datapoints (dummy SMILES used!) ("b") . \
                    2: Only ternary solvent datapoints ("t") . \
                    3: Single and binary solvent datapoints (dummy SMILES used!) ("s+b") . \
                    4: Single and ternary solvent datapoints (dummy SMILES used!) ("s+t") . \
                    5: Binary and ternary solvent datapoints (dummy SMILES used!) ("b+t") . \
                    5: Single and binary and ternary datapoints used (dummy SMILES used!) ("s+b+t") . \
                        '
                    )

        args_argparse = parser.parse_args()
        

        if args_argparse.target not in ["DGsolv", "logS"]:
            print(f"\n ! WARNING: Prediction target {args_argparse.target} not found in ['DGsolv', 'logS']. Exiting.")
            sys.exit()
        output_val_col = f"{args_argparse.target}_constant"
        
        data_in = data_process_all(database = args_argparse.db, 
                                    target = args_argparse.target,
                                    data_subset_str = args_argparse.data_subset)

        
        ##### Data Splitting!
        fold_num_used_NFPx2 = args_argparse.fold_number #!!! Fold Num set here
        split_type = args_argparse.split_option #!!! Change as needed


        batch_size_NFPx2 = args_argparse.batch_size # 1000
        sample_weight_NFPx2 = 1.0

        
        if split_type not in ['shuffle', 'solute', 'solvent_system', 'leave_one_solute']:
            print("WARNING: Could not find split type'",split_type,"' in ['shuffle', 'solute', 'solvent_system', 'leave_one_solute'].")
            print("Please double check the split type logic and your value!")
            sys.exit()
        
        print()
        if split_type == 'shuffle':

            print("USING 'shuffle' SPLITTING - fold",fold_num_used_NFPx2,"\nSplit type:",split_type)
            preprocessor_NFPx2,out_sig_NFPx2,datasets_NFPx2,dataframes_NFPx2 = get_train_test_NFPx2_AltSplits_Ternary(data=data_in, sample_weight=sample_weight_NFPx2, 
                                                                                      batch_size=batch_size_NFPx2, fold_number=fold_num_used_NFPx2,
                                                                                                   split_type='shuffle',
                                                                                                   output_val_col = output_val_col)
            train_data_final_NFPx2, train_data_NFPx2, valid_data_NFPx2, test_data_NFPx2 = datasets_NFPx2
            train_NFPx2, valid_NFPx2, test_NFPx2 = dataframes_NFPx2
        elif split_type == 'solute':

            print("USING 'solute' SPLITTING - fold",fold_num_used_NFPx2,"\nSplit type:",split_type)

            preprocessor_NFPx2,out_sig_NFPx2,datasets_NFPx2,dataframes_NFPx2 = get_train_test_NFPx2_AltSplits_Ternary(data=data_in, sample_weight=sample_weight_NFPx2, 
                                                                                      batch_size=batch_size_NFPx2, fold_number=fold_num_used_NFPx2,
                                                                                                   split_type='group_kfold_solute',
                                                                                                   output_val_col = output_val_col)
            train_data_final_NFPx2, train_data_NFPx2, valid_data_NFPx2, test_data_NFPx2 = datasets_NFPx2
            train_NFPx2, valid_NFPx2, test_NFPx2 = dataframes_NFPx2
        elif split_type == 'solvent_system':

            print("USING 'solvent_system' SPLITTING - fold",fold_num_used_NFPx2,"\nSplit type:",split_type)
            preprocessor_NFPx2,out_sig_NFPx2,datasets_NFPx2,dataframes_NFPx2 = get_train_test_NFPx2_AltSplits_Ternary(data=data_in, sample_weight=sample_weight_NFPx2, 
                                                                                  batch_size=batch_size_NFPx2, fold_number=fold_num_used_NFPx2,
                                                                                               split_type='group_kfold_solvent_system',
                                                                                                   output_val_col = output_val_col)
            train_data_final_NFPx2, train_data_NFPx2, valid_data_NFPx2, test_data_NFPx2 = datasets_NFPx2
            train_NFPx2, valid_NFPx2, test_NFPx2 = dataframes_NFPx2
        elif split_type == 'leave_one_solute':

            print("USING 'leave_one_solute' SPLITTING - fold",fold_num_used_NFPx2,"\nSplit type:",split_type)
            preprocessor_NFPx2,out_sig_NFPx2,datasets_NFPx2,dataframes_NFPx2 = get_train_test_NFPx2_AltSplits_Ternary(data=data_in, sample_weight=sample_weight_NFPx2, 
                                                                                  batch_size=batch_size_NFPx2, fold_number=fold_num_used_NFPx2,
                                                                                               split_type='group_kfold_leave_one_solute',
                                                                                               output_val_col = output_val_col)
            train_data_final_NFPx2, train_data_NFPx2, valid_data_NFPx2, test_data_NFPx2 = datasets_NFPx2
            train_NFPx2, valid_NFPx2, test_NFPx2 = dataframes_NFPx2
    
    
        kwargs_concat_ShareWeights = {
                            "model_name_in": f"Concat_Ternary_{args_argparse.modelname}_{str(args_argparse.split_option).title()}Split_Fold{fold_num_used_NFPx2}",
                             "train_data": train_data_NFPx2, 
                             "valid_data": valid_data_NFPx2, 
                             "test_data": test_data_NFPx2, 
                             "train_df": train_NFPx2,
                             "valid_df": valid_NFPx2,
                             "test_df": test_NFPx2,
                             "preprocessor": preprocessor_NFPx2,
                             "output_signature": out_sig_NFPx2,
                             "batch_size": args_argparse.batch_size, # 1000
                             "sample_weight": sample_weight_NFPx2,
                             "td_final": train_data_final_NFPx2,
                             "num_hidden": args_argparse.num_hidden, # 128
                             "num_messages": args_argparse.num_messages, # 5, Argable
                             "learn_rate": args_argparse.lr, # 1.0e-4
                             "num_epochs": args_argparse.epochs, # Should be ~500-1000
                             "node_aggreg_op": 'mean',
                             "do_stoich_multiply": 'before_dense',
                             "dropout": args_argparse.dropout,
                             "fold_number": fold_num_used_NFPx2,
                             "output_val_col": output_val_col,
                             "share_weights": args_argparse.share_weights,
                            }
        
        
        model_path = Path(Path.cwd()/"model_files"/kwargs_concat_ShareWeights["model_name_in"])
        print("Model path:",model_path)
        if model_path.exists():
            print("Model path found")
            if args_argparse.overwrite == False:
                print("\n!!!!!!WARNING!!!!!!")
                print("ERROR - model path exists and overwrite is False, EXITING!!!\nUse -o or --overwrite True to change this behaviour.")
                print("!!!!!!WARNING!!!!!!")
                sys.exit()
        else:
            print("Model path either not found or is being overwritten, creating...",model_path)
            model_path.mkdir()
                          

        print("\n\nArgparse args\n\t",args_argparse)
                
                
        print("\n\nKeyword arguments (train function input)\n\t",kwargs_concat_ShareWeights)
        for key, value in kwargs_concat_ShareWeights.items():
            excluded_dict_keys = ["train_data", "valid_data", "test_data", "train_df", "valid_df", "test_df", "td_final"]
                
            if key not in excluded_dict_keys:
                print(f"'{key}': \t{value}")
                
                
        start_time = datetime.now()
        print("\n\nModel train start time: ",start_time)
        with open("model_files/" + kwargs_concat_ShareWeights["model_name_in"] + "/model_params.txt",'w') as f:
            f.write(f"Model train start time: {start_time}")
            f.write(f"\nArgparse args:\n{args_argparse}")
            f.write("\n")
            f.write(f"\nKeyword args:\n{kwargs_concat_ShareWeights}")
            f.write("\n")
            
            
        run_model = True
        if run_model is True:
            model_NFPx2, results_NFPx2 = create_concat_GNN_ShareWeights_ternary(**kwargs_concat_ShareWeights)
        else:
            print("Run model set to False, skipping model run.")
        end_time = datetime.now()
        print("RUN TIME:",end_time - start_time)

    
