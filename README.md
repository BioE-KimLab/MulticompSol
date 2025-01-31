# MulticompSol Data Repository
This is a data repository associated with the manuscript titled "Enhancing Predictive Models for Solubility in Multi-Solvent Systems using Semi-Supervised Graph Neural Networks" by Hojin Jung‡, Christopher D. Stubbs‡, Sabari Kumar, Raúl Pérez-Soto, Su-min Song, Yeonjoon Kim, and Seonah Kim (‡Equal contribution).

This repository consists of: 
- A novel small molecule solubility database (for solutes in 1-3 solvents)
- Code to train multicomponent solubility models (for solutes in 1-3 solvents)
- Code to perform semi-supervised distillation (for up to student 5 with thresholds of 0.3 and 1)

## Using this Repository

To use the database and models in this repository, you will need a working installation of Python (v3.8-3.10) on your computer alongside the required packages (see "Packages Required"). All code was tested in Windows 10 64-bit and CentOS Stream 8, and so it should work on most modern operating systems. Please report any issues with using this code on GitHub.


### Training Models

- All model training requires a working Python environment, with GPU access and a CUDA setup ideal but not necessary (see "Packages Required" and "Using this Repository"). Getting CUDA and TensorFlow to work together on a GPU can be challenging, so the GNN model code falls back to a CPU if a GPU cannot be found.
- For all GNN models, descriptor generation is included as part of model training. Descriptors used can be changed in gnn_multisol.py (atom_features, bond_features, global_features functions). *Note that changing the number of features will generally require changing the shapes specified in any preprocessor used.*


#### Training GNN Models

- To train GNN models, first check whether your machine has CUDA and TensorFlow GPU support setup. This is often a machine-specific process, and depends on your graphics card, its supported CUDA versions, the CUDA versions installed, and the TensorFlow version installed (among other factors)
- GPU use is *not* required for GNN model training, but significant slowdowns may occur if a GPU is not used
- To train GNN models, use the following code snippets as an example (other options available by using the --help flag or checking source code). 
  - Subgraph Binary: `nohup python train_subgraph_binary.py -n "Example_BinarySubgraph" > Log_ExampleBinarySubgraph.txt &`
  - Subgraph Ternary: `nohup python train_subgraph_ternary.py -n "Example_TernarySubgraph" > Log_Example_TernarySubgraph.txt &`
  - Concat Binary: `nohup python train_concat_binary.py -n "Example_BinaryConcat" > Log_ExampleBinaryConcat.txt &`
  - Concat Ternary: `nohup python train_concat_ternary.py -n "Example_TernaryConcat" > Log_Example_TernaryConcat.txt &`
- Trained GNN models will be saved in models/.../model_files. Each folder has the preprocessor used, the best model (best_model.h5), and the prediction results (kfold_#.csv)

### Loading Models

- GNN Models
  - Trained GNN models can be loaded from the .h5 file found in /model_files/.../best_model.h5. To load, you will need to import the nfp package and pass nfp.custom_objects from nfp as custom_objects to the model load call. Rough example code can be found below.
  - Model results can be found in the same directory as the h5 file, in the csv file named `kfold_?.csv`, where ? is the fold number for that run (0-4, e.g. kfold_0.csv).

```python
def predict_df(df, model_name, csv_file_dir):
    model_dir = Path.cwd()/(f'model_files/{model_name}')
    csv_name = Path(csv_file_dir).stem
    
    model = tf.keras.models.load_model(model_dir/'best_model.h5', custom_objects = nfp.custom_objects)
	#! Will need to change the preprocessor depending on model - consult the respective training script. 
	# (e.g. train_subgraph_binary.py for binary subgraph models)
    preprocessor = CustomPreprocessor_NFPx2(  
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json(model_dir/'preprocessor.json')
    
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

    df_data = tf.data.Dataset.from_generator(
	#! Will need to change dataset generation function depending on model - consult the respective training script. 
	# (e.g. train_subgraph_binary.py for binary subgraph models)
        lambda: create_tf_dataset_NFPx2(df, preprocessor, 1.0, False), output_signature=output_signature)\ 
        .cache()\
        .padded_batch(batch_size=len(df))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    pred_results = model.predict(df_data).squeeze()
    df['predicted'] = pred_results
	return df

```

## Packages Required

All of the following were retrieved from PyPI, but should also be available on conda-forge.  Most model development was done in Python 3.8.13, but should work fine for Python 3.8 - 3.10 (3.7 may also work, but hasn't been tested). Note that a few packages require specific version numbers (nfp, TensorFlow, pandas, RDKit). Other packages have their version specified for reproducibility, and it is recommended to use the versions specified when possible.

#### Utility

- matplotlib (v3.5.3)
- seaborn (v0.12.0)
- JupyterLab (v3.4.5)

#### Descriptor Generation

- mordred (v1.2.0)
- RDKit (v2022.3.5)

#### ML/Vector Math

- numpy (v1.23.2)
- scipy (v1.9.0)
- pandas (v1.4.3)
- scikit-learn (v1.1.2) (<1.3)
- tensorflow (v2.9.1)
- tensorflow-addons (v0.18.0)
- Keras (v2.9.0)
- nfp (v0.3.0 exactly)

## Filing Issues
Please report all issues or errors with code on GitHub wherever possible.
