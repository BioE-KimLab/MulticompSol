from ssd_functions import *
print(gpus[0])
print(f"Random seed: {rand_seed}")

##### Arguments!
fold_num = 0 #!!! Fold Num set here
batch_size = 1000 # 1000
sample_weight = 1 # 1.0
split_type='shuffle'
output_val_col = 'target_constant' #'DGsolv_constant'

# Cycle1 (after trained Teacher model)
df = pd.read_csv('data/MixSolDB.csv')
df = process_all_data(df)
cycle_num=1
model_path = Path(Path.cwd()/"model_files"/f"Teacher_ShuffleSplit_Fold{fold_num}")
model = keras.models.load_model(model_path/'best_model.h5', custom_objects = nfp.custom_objects)

preprocessor = CustomPreprocessor_NFPx2_ternary( # UPDATE as needed for binary/ternary/other architectures
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
preprocessor.from_json(model_path/'preprocessor.json')

output_signature = (preprocessor.output_signature, # UPDATE as needed
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

cosmors_df = pd.read_csv('data/DGsolv_cosmors_241007_sinbinter_random_1m.csv')
cosmors_df = process_cosmors_data(cosmors_df)

cosmors_df_subsets = []
for i in range(len(cosmors_df)//10000):
    cosmors_df_subsets.append(cosmors_df.iloc[10000*i:10000*(i+1)])

if len(cosmors_df)//10000 != len(cosmors_df)/10000:
    cosmors_df_subsets.append(cosmors_df.iloc[10000*(i+1):])

print("Number of subsets in 'cosmors_df_subsets':",len(cosmors_df_subsets))

pred_results=[]
for i in tqdm(range(len(cosmors_df_subsets))):
    cosmors_df_subset = cosmors_df_subsets[i]
    cosmors_df_subset_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(cosmors_df_subset, preprocessor, 1.0, False, output_val_col=output_val_col), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=len(cosmors_df_subset))\
        .prefetch(tf.data.experimental.AUTOTUNE)
    pred_result = model.predict(cosmors_df_subset_data).squeeze()
    pred_results += list(pred_result)
cosmors_df['DGsolv_predicted'] = pred_results
cosmors_df['SSD_absolute_error'] = abs(cosmors_df['DGsolv_cosmors'] - cosmors_df['DGsolv_predicted'])
cosmors_df.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}.csv', index=False)

##### Thresholding Augmented Data
thres = [0.3, 1]

for thre in thres:
    cosmors_df_aug = cosmors_df[cosmors_df['SSD_absolute_error']<thre]
    cosmors_df_aug.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_aug_threshold{thre}.csv', index=False)
    print(f"Threshold: {thre}, Number of augmented data points: {len(cosmors_df_aug)}")
    
    cosmors_df_leftover = cosmors_df[cosmors_df['SSD_absolute_error']>=thre]
    cosmors_df_leftover.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_leftover_threshold{thre}.csv', index=False)
    print(f"Threshold: {thre}, Number of leftover data points: {len(cosmors_df_leftover)}")

    combined_df = pd.concat([df, cosmors_df_aug], axis=0)
    combined_df.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num+1}_combined_threshold{thre}.csv', index=False)
    print(len(df), len(cosmors_df_aug), len(combined_df))

# Cycle 2-6 (Student 1-5)
cycle_nums=[2,3,4,5,6]
for cycle_num in cycle_nums:
    print(f"Cycle number: {cycle_num}")
    thres = [0.3, 1]
    for i in tqdm(range(len(thres))):
        thre = thres[i]
        ### load combined_df
        combined_df = pd.read_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_combined_threshold{thre}.csv')
        
        DGsolv = []
        for _, row in combined_df.iterrows():
            if row.tag == 'cosmors':
                DGsolv.append(row.DGsolv_predicted)
            else:
                DGsolv.append(row.target)
        combined_df['target'] = DGsolv
        combined_df['target_constant'] = tf.constant(list(DGsolv))

        print("USING 'shuffle' SPLITTING - fold 0")
        preprocessor,output_signature,datasets,dataframes = get_train_test_NFPx2_AltSplits_Ternary(data=combined_df, sample_weight=sample_weight, rand_seed=rand_seed,
                                                                                    batch_size=batch_size, fold_number=fold_num,
                                                                                                split_type=split_type,
                                                                                                output_val_col = output_val_col)
        train_data_final, train_data, valid_data, test_data = datasets
        train, valid, test = dataframes

        ##### Model!
        kwargs_GNN = {"model_name_in": f"Student{cycle_num-1}_ShuffleSplit_Fold{fold_num}_Threshold{thre}",
                            "train_data": train_data,
                            "valid_data": valid_data, 
                            "test_data": test_data, 
                            "train_df": train,
                            "valid_df": valid,
                            "test_df": test,
                            "preprocessor": preprocessor,
                            "output_signature": output_signature,
                            "batch_size": 1000,
                            "sample_weight": 1, 
                            "td_final": train_data_final,
                            "num_hidden": 128,
                            "num_messages": 5,
                            "learn_rate": 1.0e-4,
                            "num_epochs": 1000, # Should be ~500-1000
                            "node_aggreg_op": 'mean',
                            "do_stoich_multiply": 'before_dense', #TODO: None
                            "dropout": 1.0e-10,
                            "fold_number": fold_num,
                            "output_val_col": output_val_col,
                            "share_weights": 'solvs',
                            "split_type": split_type
                            }

        model_path = Path(Path.cwd()/"model_files"/kwargs_GNN["model_name_in"])
        print("Model path:",model_path)
        model_path.mkdir()

        print("\n\nKeyword arguments (train function input)\n\t")
        for key, value in kwargs_GNN.items():
            excluded_dict_keys = ["train_data", "valid_data", "test_data", "train_df", "valid_df", "test_df", "td_final"]
            if key not in excluded_dict_keys:
                print(f"'{key}': \t{value}")
                
        print("\n\nModel train start time: ",datetime.now())
        # Write args to file before running
        with open("model_files/" + kwargs_GNN["model_name_in"] + "/model_params.txt",'w') as f:
            f.write(f"Model train start time: {datetime.now()}")
            f.write(f"\nKeyword args:\n{kwargs_GNN}")
            f.write("\n")

        model, results = create_GNN_NFPx2_ShareWeights_ternary(**kwargs_GNN)
        print("\n\nModel train end time: ",datetime.now())

        df_in = pd.read_csv(f'data/MixSolDB_cosmors_cycle{cycle_num-1}_leftover_threshold{thre}.csv')
        df_in = process_cosmors_data(df_in)

        df_in_subsets = []
        for i in range(len(df_in)//10000):
            df_in_subsets.append(df_in.iloc[10000*i:10000*(i+1)])
        if len(df_in)//10000 != len(df_in)/10000:
            df_in_subsets.append(df_in.iloc[10000*(i+1):])
        print("Number of subsets in 'df_in_subsets':",len(df_in_subsets))

        pred_results=[]
        for i in tqdm(range(len(df_in_subsets))):
            df_in_subset = df_in_subsets[i]
            df_in_subset_data = tf.data.Dataset.from_generator(
                lambda: create_tf_dataset_NFPx2_ternary_ShareWeights(df_in_subset, preprocessor, 1.0, False, output_val_col=output_val_col), output_signature=output_signature)\
                .cache()\
                .padded_batch(batch_size=len(df_in_subset))\
                .prefetch(tf.data.experimental.AUTOTUNE)
            pred_result = model.predict(df_in_subset_data).squeeze()
            pred_results += list(pred_result)
        df_in['DGsolv_predicted'] = pred_results
        df_in['SSD_absolute_error'] = abs(df_in['DGsolv_cosmors'] - df_in['DGsolv_predicted'])
        df_in.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_threshold{thre}.csv')

        df_in_aug = df_in[df_in['SSD_absolute_error']<thre]
        df_in_aug.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_aug_threshold{thre}.csv', index=False)
        print(f"Threshold: {thre}, Number of data points: {len(df_in_aug)}")

        df_in_leftover = df_in[df_in['SSD_absolute_error']>=thre]
        df_in_leftover.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_leftover_threshold{thre}.csv', index=False)
        print(f"Threshold: {thre}, Number of leftover data points: {len(df_in_leftover)}")

        comb_df = pd.read_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_combined_threshold{thre}.csv')
        aug_df = pd.read_csv(f'data/MixSolDB_cosmors_cycle{cycle_num}_aug_threshold{thre}.csv')
        comb_df = pd.concat([comb_df, aug_df], axis=0)
        comb_df.to_csv(f'data/MixSolDB_cosmors_cycle{cycle_num+1}_combined_threshold{thre}.csv', index=False)