import os
from utilities import load_object, save_object
from seq2seq_attention.build_dataloaders import (
    build_fields, 
    get_datasets,
    build_vocab,
) 
MIN_FREQ = 2
MAX_VOCAB_SIZE = 8000  

num_props = 1

ws_range = [2, 3, 4, 5, 10, 20, 30]

vocab_sizes = dict()
if os.path.isfile("vocab_sizes"):
    vocab_sizes = load_object("vocab_sizes")

print(vocab_sizes)

sf1, sf2 = 5, 5
for vocs in range(sf1):
    for nf1 in range(sf1):
        for nf2 in range(sf2):
            for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                for ws_use in ws_range:
                    varname = filename.replace("actual_train_", "")

                    if vocs in vocab_sizes:
                        if nf1 + 1 in vocab_sizes[vocs]:
                            if nf2 + 1 in vocab_sizes[vocs][nf1 + 1]:
                                if ws_use in vocab_sizes[vocs][nf1 + 1][nf2 + 1]:
                                    if varname in vocab_sizes[vocs][nf1 + 1][nf2 + 1][ws_use]:
                                        continue

                    src_field, trg_field = build_fields()
                    okall = False
                    while not okall:
                        try:
                            train_set, val_set, test_set = get_datasets(train_path="tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv", 
                                                                        val_path="tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                                        test_path="tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv",
                                                                        src_field=src_field, 
                                                                        trg_field=trg_field)
                        
                            build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
                            # Check vocabulary 
                            print(varname, str(nf1 + 1), str(nf2 + 1), ws_use, len(src_field.vocab), len(trg_field.vocab))

                            okall = True
                        except:
                            okall = False

                    if vocs not in vocab_sizes:
                        vocab_sizes[vocs] = dict()

                    if nf1 + 1 not in vocab_sizes[vocs]:
                        vocab_sizes[vocs][nf1 + 1] = dict()
                        
                    if nf2 + 1 not in vocab_sizes[vocs][nf1 + 1]:
                        vocab_sizes[vocs][nf1 + 1][nf2 + 1] = dict()

                    if ws_use not in vocab_sizes[vocs][nf1 + 1][nf2 + 1]:
                        vocab_sizes[vocs][nf1 + 1][nf2 + 1][ws_use] = dict()
                        
                    vocab_sizes[vocs][nf1 + 1][nf2 + 1][ws_use][varname] = {"src": len(src_field.vocab), "trg": len(trg_field.vocab)}
                    save_object("vocab_sizes", vocab_sizes)