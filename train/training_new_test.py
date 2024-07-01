import argparse
import os.path
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    import torch.utils.data as Data
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import itertools
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    # BATCH_COPIES = args.batch_size

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
##数据
    # data_path = args.path_for_training_data
    # params = {
    #     "LIST"    : f"{data_path}/list.csv",
    #     "VAL"     : f"{data_path}/valid_clusters.txt",
    #     "TEST"    : f"{data_path}/test_clusters.txt",
    #     "DIR"     : f"{data_path}",
    #     "DATCUT"  : "2030-Jan-01",
    #     "RESCUT"  : args.rescut, #resolution cutoff for PDBs
    #     "HOMO"    : 0.70 #min seq.id. to detect homo chains
    # }
    #
    #
    # LOAD_PARAM = {'batch_size': 1,
    #               'shuffle': True,
    #               'pin_memory':False,
    #               'num_workers': 4} #4
    #
    #
    # if args.debug:
    #     args.num_examples_per_epoch = 50
    #     args.max_protein_length = 1000
    #     args.batch_size = 1000
    #
    # train, valid, test = build_training_clusters(params, args.debug)
    #
    # train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    # train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    # valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)



    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)


    if PATH:
        checkpoint = torch.load(PATH)
        # total_step = 0  # write total_step from the checkpoint
        # epoch = 0  # write epoch from the checkpoint
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # with ProcessPoolExecutor(max_workers=12) as executor:
    #     q = queue.Queue(maxsize=3)
    #     p = queue.Queue(maxsize=3)
    #     for i in range(3):
    #         q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
    #         p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
    #     pdb_dict_train = q.get().result()
    #     pdb_dict_valid = p.get().result()

    ## parse_PDB_biounits
    def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''

        alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
        states = len(alpha_1)
        alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

        aa_1_N = {a: n for n, a in enumerate(alpha_1)}
        aa_3_N = {a: n for n, a in enumerate(alpha_3)}
        aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
        aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
        aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

        def AA_to_N(x):
            # ["ARND"] -> [[0,1,2,3]]
            x = np.array(x);
            if x.ndim == 0: x = x[None]
            return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

        def N_to_AA(x):
            # [[0,1,2,3]] -> ["ARND"]
            x = np.array(x);
            if x.ndim == 1: x = x[None]
            return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

        xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
        for line in open(x, "rb"):
            line = line.decode("utf-8", "ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:12 + 4].strip()
                    resi = line[17:17 + 3]
                    resn = line[22:22 + 5].strip()
                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1
                    #         resn = int(resn)
                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

        # convert to numpy arrays, fill in missing values
        seq_, xyz_ = [], []
        try:
            for resn in range(min_resn, max_resn + 1):
                if resn in seq:
                    for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k], 20))
                else:
                    seq_.append(20)
                if resn in xyz:
                    for k in sorted(xyz[resn]):
                        for atom in atoms:
                            if atom in xyz[resn][k]:
                                xyz_.append(xyz[resn][k][atom])
                            else:
                                xyz_.append(np.full(3, np.nan))
                else:
                    for atom in atoms: xyz_.append(np.full(3, np.nan))
            return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
        except TypeError:
            return 'no_chain', 'no_chain'

    ##parse_pdb
    def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
        c = 0
        pdb_dict_list = []
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                         'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet + extra_alphabet

        if input_chain_list:
            chain_alphabet = input_chain_list

        biounit_names = [path_to_pdb]
        for biounit in biounit_names:
            my_dict = {}
            s = 0
            concat_seq = ''
            concat_N = []
            concat_CA = []
            concat_C = []
            concat_O = []
            concat_mask = []
            coords_dict = {}
            for letter in chain_alphabet:
                if ca_only:
                    sidechain_atoms = ['CA']
                else:
                    sidechain_atoms = ['N', 'CA', 'C', 'O']
                xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
                if type(xyz) != str:
                    concat_seq += seq[0]
                    my_dict['seq_chain_' + letter] = seq[0]
                    coords_dict_chain = {}
                    if ca_only:
                        coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                    else:
                        coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                        coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                        coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                        coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                    my_dict['coords_chain_' + letter] = coords_dict_chain
                    s += 1
            fi = biounit.rfind("/")
            my_dict['name'] = biounit[(fi + 1):-4]
            my_dict['num_of_chains'] = s
            my_dict['seq'] = concat_seq
            if s <= len(chain_alphabet):
                pdb_dict_list.append(my_dict)
                c += 1
        return pdb_dict_list

    ##StructureDatasetPDB
    class StructureDatasetPDB():
        def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
                     alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
            alphabet_set = set([a for a in alphabet])
            discard_count = {
                'bad_chars': 0,
                'too_long': 0,
                'bad_seq_length': 0
            }

            self.data = []

            start = time.time()
            for i, entry in enumerate(pdb_dict_list):
                seq = entry['seq']
                name = entry['name']

                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start

                # print('Discarded', discard_count)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ##StructureDataset
    class StructureDataset():
        def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
                     alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
            alphabet_set = set([a for a in alphabet])
            discard_count = {
                'bad_chars': 0,
                'too_long': 0,
                'bad_seq_length': 0
            }

            with open(jsonl_file) as f:
                self.data = []

                lines = f.readlines()
                start = time.time()
                for i, line in enumerate(lines):
                    entry = json.loads(line)
                    seq = entry['seq']
                    name = entry['name']

                    # Convert raw coords to np arrays
                    # for key, val in entry['coords'].items():
                    #    entry['coords'][key] = np.asarray(val)

                    # Check if in alphabet
                    bad_chars = set([s for s in seq]).difference(alphabet_set)
                    if len(bad_chars) == 0:
                        if len(entry['seq']) <= max_length:
                            if True:
                                self.data.append(entry)
                            else:
                                discard_count['bad_seq_length'] += 1
                        else:
                            discard_count['too_long'] += 1
                    else:
                        if verbose:
                            print(name, bad_chars, entry['seq'])
                        discard_count['bad_chars'] += 1

                    # Truncate early
                    if truncate is not None and len(self.data) == truncate:
                        return

                    if verbose and (i + 1) % 1000 == 0:
                        elapsed = time.time() - start
                        print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))
                if verbose:
                    print('discarded', discard_count)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ##tied_featurize
    def tied_featurize(batch, device, chain_dict=None, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None,
                       pssm_dict=None, bias_by_res_dict=None, ca_only=False):
        """ Pack and pad batch into torch tensors """
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        B = len(batch)
        lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
        L_max = max([len(b['seq']) for b in batch])
        if ca_only:
            X = np.zeros([B, L_max, 1, 3])
        else:
            X = np.zeros([B, L_max, 4, 3])
        residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
        chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
        pssm_coef_all = np.zeros([B, L_max], dtype=np.float32)  # 1.0 for the bits that need to be predicted
        pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32)  # 1.0 for the bits that need to be predicted
        pssm_log_odds_all = 10000.0 * np.ones([B, L_max, 21],
                                              dtype=np.float32)  # 1.0 for the bits that need to be predicted
        chain_M_pos = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
        bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
        chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
        S = np.zeros([B, L_max], dtype=np.int32)
        omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
        # Build the batch
        letter_list_list = []
        visible_list_list = []
        masked_list_list = []
        masked_chain_length_list_list = []
        tied_pos_list_of_lists_list = []
        for i, b in enumerate(batch):
            if chain_dict != None:
                masked_chains, visible_chains = chain_dict[
                    b['name']]  # masked_chains a list of chain letters to predict [A, D, F]
            else:
                masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
                visible_chains = []
            masked_chains.sort()  # sort masked_chains
            visible_chains.sort()  # sort visible_chains
            all_chains = masked_chains + visible_chains
        for i, b in enumerate(batch):
            mask_dict = {}
            a = 0
            x_chain_list = []
            chain_mask_list = []
            chain_seq_list = []
            chain_encoding_list = []
            c = 1
            letter_list = []
            global_idx_start_list = [0]
            visible_list = []
            masked_list = []
            masked_chain_length_list = []
            fixed_position_mask_list = []
            omit_AA_mask_list = []
            pssm_coef_list = []
            pssm_bias_list = []
            pssm_log_odds_list = []
            bias_by_res_list = []
            l0 = 0
            l1 = 0
            for step, letter in enumerate(all_chains):
                if letter in visible_chains:
                    letter_list.append(letter)
                    visible_list.append(letter)
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                    chain_length = len(chain_seq)
                    global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                    chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                    chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                    if ca_only:
                        x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                        if len(x_chain.shape) == 2:
                            x_chain = x_chain[:, None, :]
                    else:
                        x_chain = np.stack([chain_coords[c] for c in
                                            [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                             f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                    x_chain_list.append(x_chain)
                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                    l0 += chain_length
                    c += 1
                    fixed_position_mask = np.ones(chain_length)
                    fixed_position_mask_list.append(fixed_position_mask)
                    omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                    omit_AA_mask_list.append(omit_AA_mask_temp)
                    pssm_coef = np.zeros(chain_length)
                    pssm_bias = np.zeros([chain_length, 21])
                    pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                    pssm_coef_list.append(pssm_coef)
                    pssm_bias_list.append(pssm_bias)
                    pssm_log_odds_list.append(pssm_log_odds)
                    bias_by_res_list.append(np.zeros([chain_length, 21]))
                if letter in masked_chains:
                    masked_list.append(letter)
                    letter_list.append(letter)
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                    chain_length = len(chain_seq)
                    global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                    masked_chain_length_list.append(chain_length)
                    chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                    chain_mask = np.ones(chain_length)  # 1.0 for masked
                    if ca_only:
                        x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                        if len(x_chain.shape) == 2:
                            x_chain = x_chain[:, None, :]
                    else:
                        x_chain = np.stack([chain_coords[c] for c in
                                            [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                             f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                    x_chain_list.append(x_chain)
                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                    l0 += chain_length
                    c += 1
                    fixed_position_mask = np.ones(chain_length)
                    if fixed_position_dict != None:
                        fixed_pos_list = fixed_position_dict[b['name']][letter]
                        if fixed_pos_list:
                            fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                    fixed_position_mask_list.append(fixed_position_mask)
                    omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                    if omit_AA_dict != None:
                        for item in omit_AA_dict[b['name']][letter]:
                            idx_AA = np.array(item[0]) - 1
                            AA_idx = np.array(
                                [np.argwhere(np.array(list(alphabet)) == AA)[0][0] for AA in item[1]]).repeat(
                                idx_AA.shape[0])
                            idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                            omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                    omit_AA_mask_list.append(omit_AA_mask_temp)
                    pssm_coef = np.zeros(chain_length)
                    pssm_bias = np.zeros([chain_length, 21])
                    pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                    if pssm_dict:
                        if pssm_dict[b['name']][letter]:
                            pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                            pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                            pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                    pssm_coef_list.append(pssm_coef)
                    pssm_bias_list.append(pssm_bias)
                    pssm_log_odds_list.append(pssm_log_odds)
                    if bias_by_res_dict:
                        bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                    else:
                        bias_by_res_list.append(np.zeros([chain_length, 21]))

            letter_list_np = np.array(letter_list)
            tied_pos_list_of_lists = []
            tied_beta = np.ones(L_max)
            if tied_positions_dict != None:
                tied_pos_list = tied_positions_dict[b['name']]
                if tied_pos_list:
                    set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                    for tied_item in tied_pos_list:
                        one_list = []
                        for k, v in tied_item.items():
                            start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                            if isinstance(v[0], list):
                                for v_count in range(len(v[0])):
                                    one_list.append(start_idx + v[0][v_count] - 1)  # make 0 to be the first
                                    tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                            else:
                                for v_ in v:
                                    one_list.append(start_idx + v_ - 1)  # make 0 to be the first
                        tied_pos_list_of_lists.append(one_list)
            tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

            x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
            all_sequence = "".join(chain_seq_list)
            m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
            chain_encoding = np.concatenate(chain_encoding_list, 0)
            m_pos = np.concatenate(fixed_position_mask_list, 0)  # [L,], 1.0 for places that need to be predicted

            pssm_coef_ = np.concatenate(pssm_coef_list, 0)  # [L,], 1.0 for places that need to be predicted
            pssm_bias_ = np.concatenate(pssm_bias_list, 0)  # [L,], 1.0 for places that need to be predicted
            pssm_log_odds_ = np.concatenate(pssm_log_odds_list, 0)  # [L,], 1.0 for places that need to be predicted

            bias_by_res_ = np.concatenate(bias_by_res_list,
                                          0)  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

            l = len(all_sequence)
            x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
            X[i, :, :, :] = x_pad

            m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0,))
            m_pos_pad = np.pad(m_pos, [[0, L_max - l]], 'constant', constant_values=(0.0,))
            omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list, 0), [[0, L_max - l]], 'constant',
                                      constant_values=(0.0,))
            chain_M[i, :] = m_pad
            chain_M_pos[i, :] = m_pos_pad
            omit_AA_mask[i,] = omit_AA_mask_pad

            chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0,))
            chain_encoding_all[i, :] = chain_encoding_pad

            pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max - l]], 'constant', constant_values=(0.0,))
            pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
            pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))

            pssm_coef_all[i, :] = pssm_coef_pad
            pssm_bias_all[i, :] = pssm_bias_pad
            pssm_log_odds_all[i, :] = pssm_log_odds_pad

            bias_by_res_pad = np.pad(bias_by_res_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
            bias_by_res_all[i, :] = bias_by_res_pad

            # Convert to labels
            indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
            S[i, :l] = indices
            letter_list_list.append(letter_list)
            visible_list_list.append(visible_list)
            masked_list_list.append(masked_list)
            masked_chain_length_list_list.append(masked_chain_length_list)

        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
        X[isnan] = 0.

        # Conversion
        pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
        pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
        pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

        tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

        jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
        bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
        phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
        psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
        omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
        dihedral_mask = np.concatenate([phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]],
                                       -1)  # [B,L,3]
        dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
        residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
        S = torch.from_numpy(S).to(dtype=torch.long, device=device)
        X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
        chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
        chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
        omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
        chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
        if ca_only:
            X_out = X[:, :, 0]
        else:
            X_out = X
        return X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta

    #StructureLoader
    class StructureLoader():
        def __init__(self, dataset, batch_size=100, shuffle=True,
                     collate_fn=lambda x: x, drop_last=False):
            self.dataset = dataset
            self.size = len(dataset)
            self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
            self.batch_size = batch_size
            sorted_ix = np.argsort(self.lengths)

            # Cluster into batches of similar sizes
            clusters, batch = [], []
            batch_max = 0
            for ix in sorted_ix:
                size = self.lengths[ix]
                if size * (len(batch) + 1) <= self.batch_size:
                    batch.append(ix)
                    batch_max = size
                else:
                    clusters.append(batch)
                    batch, batch_max = [], 0
            if len(batch) > 0:
                clusters.append(batch)
            self.clusters = clusters

        def __len__(self):
            return len(self.clusters)

        def __iter__(self):
            np.random.shuffle(self.clusters)
            for b_idx in self.clusters:
                batch = [self.dataset[i] for i in b_idx]
                yield batch

    #zhuang list
    def zhuanlist(dataset_train):
        dataset_train_1 = []
        for i in dataset_train:
            dataset_train_1.append([i])
        return dataset_train_1

    #jin
    # pdb_dict_train = parse_PDB(args.pdb_path_train)
    # pdb_dict_train = StructureDatasetPDB(pdb_dict_train, truncate=None, max_length=100)
    # pdb_dict_valid = parse_PDB(args.pdb_path_test)
    # pdb_dict_valid = StructureDatasetPDB(pdb_dict_valid, truncate=None, max_length=100)
    #
    # dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
    # dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
    #
    # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size) #
    # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

    # pdb_path_chains = args.pdb_path_chains
    # pdb_path_chains = args.pdb_path_chains

    '''new '''
    ####Change to def
    def make_chain_id_dict(pdb_dict_list):
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']  # ['A','B', 'C',...]

        designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)
        return chain_id_dict

    # for p in os.listdir(my_path):
    
    #train_data

    # def lujin(my_path): #wen jian jia ming
    #     pdb_dict_list_trains = []
    #
    #     for file in os.listdir(my_path):
    #         if not os.path.exists(my_path):
    #             continue
    #         pdb_dict_list_train=parse_PDB(os.path.join(my_path, file))
    #         pdb_dict_list_trains += pdb_dict_list_train
    #     return pdb_dict_list_trains
    #
    # pdb_dict_list_train = lujin(args.pdb_path_train)
    #



    dataset_train_all = StructureDataset(args.jsonl_path_train, truncate=None, max_length=args.max_length) # n ge [5ge zi dian]
    # for i in range()
    # dataset_train = zhuanlist(dataset_train) #[n ge [5ge zi dian]]
    # dataset_train = torch.from_numpy(dataset_train)
    # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)  #

    # pdb_dict_list_train = parse_PDB(args.pdb_path_train)
    # dataset_train = StructureDatasetPDB(pdb_dict_list_train, truncate=None, max_length=args.max_length)
    #
    # chain_id_dict_train = make_chain_id_dict(pdb_dict_list_train)

    #valid_data
    dataset_valid_all = StructureDataset(args.jsonl_path_valid, truncate=None, max_length=args.max_length)
    # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    # pdb_dict_list_valid = parse_PDB(args.pdb_path_test)
    # dataset_valid = StructureDatasetPDB(pdb_dict_list_valid, truncate=None, max_length=args.max_length)
    #
    # chain_id_dict_valid = make_chain_id_dict(pdb_dict_list_valid)


    #native
    # pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only)
    # dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    #
    # chain_id_dict = make_chain_id_dict(pdb_dict_list,args.pdb_path_chains)
    #
    # chain_id_dict = make_chain_id_dict(pdb_dict_list,args.pdb_path_chains)
    ####
    # all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']  # ['A','B', 'C',...]
    # if args.pdb_path_chains:
    #     designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
    # else:
    #     designed_chain_list = all_chain_list
    # fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    # chain_id_dict = {}
    # chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)

##

    # bz = 10

    #bz
    bz_train = args.batch_size_train
    bz_valid = args.batch_size_valid

    start_index_train = 0
    start_index_valid = 0
    #

    ##
    reload_c = 1
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.


        #bz
        if start_index_train>len(dataset_train_all):
            start_index_train =0
            dataset_train = dataset_train_all[start_index_train:start_index_train + bz_train]
        else:
            dataset_train = dataset_train_all[start_index_train:start_index_train + bz_train]

        if start_index_valid>len(dataset_valid_all):
            start_index_valid =0
            dataset_valid = dataset_valid_all[start_index_valid:start_index_valid + bz_valid]
        else:
            dataset_valid = dataset_valid_all[start_index_valid:start_index_valid + bz_valid]
        # if start_index>int(len(dataset_train)-bz):
        # dataset_train = dataset_train[start_index:start_index + bz]

        # if e % args.reload_data_every_n_epochs == 0:
        #     bz = 10
        #     start_index = 0
        #     if reload_c != 0:
        #
        #         dataset_train = dataset_train[start_index:start_index+bz]
        #         start_index = start_index + bz
        # #         # pdb_dict_train = q.get().result()
        # #         # dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
        # #
        # #         loader_train = Data.TensorDataset(dataset_train)
        # #         loader_train = Data.DataLoader(dataset=loader_train,batch_size=args.batch_size,shuffle=True)
        # #         loader_train = loader_train.numpy()
        # #         # loader_train=DataLoader(dataset=dataset_train,batch_size=args.batch_size,shuffle=True)
        # #         # loader_train = loader_train.numpy()
        # #
        # #         # loader_valid = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=True)
        # #
        # #         # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        # #         # pdb_dict_valid = p.get().result()
        # #         # dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        # #         # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        # #         # q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        # #         # p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        #     reload_c += 1
        # for i in range(0, len(dataset_train) , 5):
        #     dataset_train[i:i+5]
        for ix, protein in enumerate(dataset_train):
            # print(ix,protein,'\n')


            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            # batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            # batch_clones = [copy.deepcopy(protein) for i in range(1)]
            (
                X, S, mask, lengths,
                chain_M, chain_encoding_all,
                chain_list_list, visible_list_list,
                masked_list_list, masked_chain_length_list_list,
                chain_M_pos, omit_AA_mask, residue_idx,
                dihedral_mask, tied_pos_list_of_lists_list,
                pssm_coef, pssm_bias, pssm_log_odds_all,
                bias_by_res_all, tied_beta
             ) = tied_featurize([protein], device)

        # for _, batch in enumerate(loader_train):
            start_batch = time.time()
            # X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                scaler.scale(loss_av_smoothed).backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()

            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            total_step += 1 #once

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            #
            for ix, protein in enumerate(dataset_valid):
                score_list = []
                global_score_list = []
                all_probs_list = []
                all_log_probs_list = []
                S_sample_list = []
                # batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
                # batch_clones = [copy.deepcopy(protein) for i in range(1)]
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                    [protein], device)
                # X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                #     batch_clones, device)
            # for _, batch in enumerate(loader_valid):
            #     X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)

        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1)
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')

        #bz
        start_index_train = start_index_train + bz_train
        start_index_valid = start_index_valid + bz_valid

        checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)

        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 



    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")  ##
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")

    # change
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--pdb_path_train", type=str, default="./data/1r1f.pdb", help="path for loading training data")
    argparser.add_argument("--pdb_path_test", type=str, default="3HTN.pdb", help="path for loading training data")



    #input Processing into a jsonl file
    argparser.add_argument("--jsonl_path_train", type=str, default="./data/HLA/combined_train/parsed_pdbs.jsonl",help="path for loading training data") #json
    argparser.add_argument("--jsonl_path_valid", type=str, default="./data/HLA/combined_val/parsed_pdbs.jsonl",help="path for loading valid data") #json
    argparser.add_argument("--path_for_outputs", type=str, default="./hla_model_weights/n200_tb256_vb32", help="path for logs and model weights") #output weight path
    # argparser.add_argument("--previous_checkpoint", type=str, default="./vanilla_model_weights/v_48_002.pt",help="path for previous model weights, e.g. file.pt")  ##pretrain ckpt
    argparser.add_argument("--previous_checkpoint", type=str, default="./exp_020/model_weights/epoch_last.pt",help="path for previous model weights, e.g. file.pt")  ##pretrain ckpt

    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--batch_size_train", type=int, default=256, help="number of tokens for one batch") #
    argparser.add_argument("--batch_size_valid", type=int, default=32, help="number of tokens for one batch") #

    args = argparser.parse_args()

    main(args)
