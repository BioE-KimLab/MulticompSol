import re
import sys
from pathlib import Path

# Given a name of a model folder will print results to terminal.
# e.g. 'python print_results.py "model_001"' (change model_001 to an appropriate name)
args = sys.argv

model_folder = sys.argv[1]
model_folder_pth = Path.cwd()/"model_files"/model_folder
def parse_results_txt(model_folder):
    if not Path(model_folder/'results.txt').exists():
        print("Warning: results.txt not found.\nExiting...")
        sys.exit()

    print("--> ",model_folder.stem," <--")




    with open(model_folder/'results.txt', 'r') as f:
        txt = f.read()

    lines = re.split('\n', txt)
    maes, rmses, r2s = [x.strip() for x in lines]

    maes = re.sub("MAEs:\s+","",maes)
    rmses = re.sub("RMSEs:\s+","",rmses)
    r2s = re.sub("R2s:\s+","",r2s)



    maes = ';'.join(maes.split(','))
    rmses = ';'.join(rmses.split(','))
    r2s = ';'.join(r2s.split(','))


    joined_str = ';'.join([maes,rmses,r2s])
    print(f"\nResults (line):\n{joined_str}\n")

    grid_str = '\n'.join([maes,rmses,r2s])
    print(f"\nnResults (grid):\n{grid_str}\n")

    kfold_files = [x for x in model_folder.glob("kfold*.csv")]
    if len(kfold_files) == 0:
        print("Could not find kfold files.")
    elif len(kfold_files) > 1:
        print(f"More than one kfold file found: {kfold_files}. Skipping!")
    else:
        with open(kfold_files[0], 'r') as f:
            kfold_lines = f.readlines()
        kfold_header = kfold_lines[0]
        if kfold_header.count("DGsolv_constant") > 0:
            print("Model uses DGsolv.")
        elif kfold_header.count("logS_constant") > 0:
            print("Model uses logS.")
        else:
            print("Could not determine model prediction target.")

    if not Path(model_folder/'model_params.txt').exists():
        print("Could not find model params.")
    else:
        with open(model_folder/'model_params.txt', 'r') as f:
            model_params_lines = f.readlines()
        print("\nModel params (argparse):",model_params_lines[2])

        
        
parse_results_txt(model_folder_pth)
