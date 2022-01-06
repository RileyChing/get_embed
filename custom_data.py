from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import copy
import pickle


class CustomDataset(Dataset):
    def __init__(self, prefix, args):
        assert prefix == args.train_prefix or prefix == args.test_prefix
        
        print(f"Loading {prefix}_id.pickle...")
        with open(f"../{args.data_dir}/{prefix}_ids.pickle", 'rb') as f:
            dialogues_ids = pickle.load(f)[:10]

        print(f"Loading {prefix}_utters.pickle...")
        with open(f"../{args.data_dir}/{prefix}_utters.pickle", 'rb') as f:
            dialogues_utters = pickle.load(f)[:10]

        self.input_ids = []  # (N, L)
        self.token_type_ids = []  # (N, L)
        self.labels = []  # (N, L)
        self.dialogues = []
        
        print(f"Processing {prefix} data...")

        total_seq_ids = []
        for d, dialogue_ids in enumerate(tqdm(dialogues_ids)):
            cur_sp = 1
            hists = []
            for t, token_ids in enumerate(dialogue_ids):
                if cur_sp == 1:
                    sp_id = args.sp1_id
                else:
                    sp_id = args.sp2_id
                    
                if len(hists) < args.max_turns:
                    hists.append([sp_id] + token_ids)
                else:
                    hists = hists[1:] + [[sp_id] + token_ids]
                    
                cur_sp = (cur_sp % 2) + 1
                total_seq_ids.append(copy.deepcopy(hists))
        
        for s, seq_ids in enumerate(tqdm(total_seq_ids)):
            if len(seq_ids) > 1 and seq_ids[-1][0] == args.sp2_id:
                seq_ids[0] = [args.bos_id] + seq_ids[0]
                seq_ids[-1] = seq_ids[-1] + [args.eos_id]
                
                total_len = 0
                for token_ids in seq_ids:
                    total_len += len(token_ids)
                    
                if total_len > args.max_len:
                    seq_ids = [token_ids[:args.utter_len] for token_ids in seq_ids]
                    seq_ids[-1][-1] = args.eos_id
                    
                token_type_id = [[token_ids[0]] * len(token_ids) if t != 0 else [token_ids[1]] * len(token_ids) for t, token_ids in enumerate(seq_ids)]
                lm_label = [[-100] * len(token_ids) if t != len(seq_ids)-1 else token_ids for t, token_ids in enumerate(seq_ids)]
                input_id = list(chain.from_iterable(seq_ids))
                token_type_id = list(chain.from_iterable(token_type_id))
                lm_label = list(chain.from_iterable(lm_label))
                
                assert len(input_id) == len(lm_label) and len(input_id) == len(token_type_id)
                
                self.input_ids.append(input_id)
                self.token_type_ids.append(token_type_id)
                self.labels.append(lm_label)

        total_seq_utters = []
        for dd, dialogue_utters in enumerate(tqdm(dialogues_utters)):
            texts = []
            for tt, token_utters in enumerate(dialogue_utters):
                if len(texts) < args.max_turns:
                    texts.append(token_utters)
                else:
                    texts = texts[1:] + [token_utters]

                total_seq_utters.append(copy.deepcopy(texts))

        for ss, seq_utters in enumerate(tqdm(total_seq_utters)):
            lenth = len(seq_utters)
            # print(total_seq_ids[ss][0][1])
            spid = total_seq_ids[ss][0][1]
            if len(seq_utters) > 1 :
                if (lenth % 2 == 0 and lenth != 5) or (lenth == 5 and spid == 50261):
                    # seq_ids[0] = [args.bos_id] + seq_ids[0]
                    # seq_ids[-1] = seq_ids[-1] + [args.eos_id]

                    total_len = 0
                    for token_utters in seq_utters:
                        total_len += len(token_utters)

                    if total_len > args.max_len:
                        seq_utters = [token_utters[:args.utter_len] for token_utters in seq_utters]
                        # seq_utters[-1][-1] = args.eos_id

                    # token_type_id = [[token_ids[0]] * len(token_ids) if t != 0 else [token_ids[1]] * len(token_ids) for
                    #                 t, token_ids in enumerate(seq_ids)]
                    # lm_label = [[-100] * len(token_ids) if t != len(seq_ids) - 1 else token_ids for t, token_ids in
                    #             enumerate(seq_ids)]
                    dialogue = list(seq_utters)  # chain.from_iterable(seq_utters))

                    # assert len(self.input_ids[s]) == len(dialogue)

                    self.dialogues.append(dialogue)


    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx], idx, self.dialogues[idx]
    
    
class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id
        
    def pad_collate(self, batch):

        input_ids, token_type_ids, labels = [], [], []
        for i, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[2]))
            #idx.append(torch.LongTensor(seqs[3]))

            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        #idx = torch.nn.utils.rnn.pad_sequence(idx, batch_first=True, padding_value=self.pad_id)
    
        return input_ids, token_type_ids, labels#, idx
