#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --output=example_2.out

#source activate mlfold


#folder_with_pdbs="../training/data/HLA/combined_test/"


#output_dir="../training/test_outputs/HLA_designB_e300"
#output_dir="../training/do/All_designB_e400_5tiao"


#new_strings=("MHC_pdbs_A0101" "MHC_pdbs_A0201" )
new_strings=("MHC_pdbs_A0301" "MHC_pdbs_A1101" "MHC_pdbs_A2402" "MHC_pdbs_B0702"
 "MHC_pdbs_B1501" "MHC_pdbs_B1502" "MHC_pdbs_B2705" "MHC_pdbs_B2709"
 "MHC_pdbs_B3501" "MHC_pdbs_B5701" "MHC_pdbs_B5801" "MHC_pdbs_E0101" "MHC_pdbs_E0103")
for new_string in "${new_strings[@]}"; do
    # input file path
    folder_with_pdbs="../training/data/HLA/test_fenjiyinxin/${new_string}/test/"

    ### output file path
    output_dir="../training/do/${new_string}_designB_e400_5tiao"





    if [ ! -d $output_dir ]
    then
        mkdir -p $output_dir
    fi

    path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
    path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
    #chains_to_design="A B"
    chains_to_design="B"
    python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

    python ../helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"



    python ../protein_mpnn_run.py \
            --jsonl_path $path_for_parsed_chains \
            --chain_id_jsonl $path_for_assigned_chains \
            --out_folder $output_dir \
            --num_seq_per_target 5 \
            --sampling_temp "0.1" \
            --seed 37 \
            --batch_size 1 \
            --path_to_model_weights "../training/hla_model_weights/n200_tb256_vb32/model_weights" \
            --model_name "epoch400_step43900"
done