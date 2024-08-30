import pandas as pd
import os
from utilities import save_object, load_object
from pytorch_utilities import print_predictions
import torch
from tqdm import tqdm
from time import time
from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.model import Seq2Seq_With_Attention
from seq2seq_attention.build_dataloaders import (
    build_fields,
    build_bucket_iterator,
    get_datasets,
    build_vocab,
)
from seq2seq_attention.translate import translate_sentence

BATCH_SIZE = 256
DEVICE = "cuda"
LR = 5e-4 
EPOCHS = 20
MAX_VOCAB_SIZE = 8000
MIN_FREQ = 2
ENC_EMB_DIM = 256
HIDDEN_DIM_ENC = 512
HIDDEN_DIM_DEC = 512
NUM_LAYERS_ENC = 1
NUM_LAYERS_DEC = 1
EMB_DIM_TRG = 256 
TEACHER_FORCING = 0.8
PROGRESS_BAR = False
USE_WANDB = True
DROPOUT = 0
TRAIN_ATTENTION = False
progress_bar=False
disable_pro_bar = not progress_bar

num_props = 1

ws_range = [2, 3, 4, 5, 10, 20, 30]

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

marking_for_range = dict()

for ix in range(len(num_to_params)):
    if num_to_params[ix] == 2:
        marking_for_range[num_to_ws[ix]] = ix

print(marking_for_range)

model_name = "GRU_Att"

vocab_sizes = dict()
if os.path.isfile("vocab_sizes"):
    vocab_sizes = load_object("vocab_sizes")

sf1, sf2 = 5, 5
for nf1 in range(sf1):
    for nf2 in range(sf2):
        use_eval = True
        train_a_model = True
        if train_a_model:
            for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                varname = filename.replace("actual_train_", "")
                if "time" in varname:
                    continue
                for ws_use in ws_range:
                    if os.path.isfile("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv"):
                        continue

                    if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name):
                        os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name)

                    if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/train/" + model_name):
                        os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/train/" + model_name)
                    if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/val/" + model_name):
                        os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/val/" + model_name)
                    if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/test/" + model_name):
                        os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/test/" + model_name)

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
                            print(varname, "train_attention" + str(marking_for_range[ws_use]), str(nf1 + 1), str(nf2 + 1), ws_use, len(src_field.vocab), len(trg_field.vocab))

                            okall = True
                        except:
                            okall = False

                    if 2 not in vocab_sizes:
                        vocab_sizes[2] = dict()

                    if nf1 + 1 not in vocab_sizes[2]:
                        vocab_sizes[2][nf1 + 1] = dict()
                        
                    if nf2 + 1 not in vocab_sizes[2][nf1 + 1]:
                        vocab_sizes[2][nf1 + 1][nf2 + 1] = dict()

                    if ws_use not in vocab_sizes[2][nf1 + 1][nf2 + 1]:
                        vocab_sizes[2][nf1 + 1][nf2 + 1][ws_use] = dict()
                        
                    vocab_sizes[2][nf1 + 1][nf2 + 1][ws_use][varname] = {"src": len(src_field.vocab), "trg": len(trg_field.vocab)}
                    save_object("vocab_sizes", vocab_sizes)

                    train_loader = build_bucket_iterator(dataset=train_set, batch_size=BATCH_SIZE, device=DEVICE)
                    val_loader = build_bucket_iterator(dataset=val_set, batch_size=BATCH_SIZE, device=DEVICE)
                    test_loader = build_bucket_iterator(dataset=test_set, batch_size=BATCH_SIZE, device=DEVICE)

                    # Safe number of batches in train loader and eval points
                    perc = 0.25
                    n_batches_train = len(train_loader)
                    eval_points = [
                        round(i * perc * n_batches_train) - 1 for i in range(1, round(1 / perc))
                    ]
                    eval_points.append(n_batches_train - 1)

                    # Get padding/<sos> idxs
                    src_pad_idx = src_field.vocab.stoi["<pad>"]
                    trg_pad_idx = trg_field.vocab.stoi["<pad>"]
                    seq_beginning_token_idx = src_field.vocab.stoi["<sos>"]
                    assert src_field.vocab.stoi["<sos>"] == trg_field.vocab.stoi["<sos>"]

                    # Init model wrapper class
                    model = Seq2Seq_With_Attention(
                        lr=LR,
                        enc_vocab_size=len(src_field.vocab),
                        vocab_size_trg=len(trg_field.vocab),
                        enc_emb_dim=ENC_EMB_DIM,
                        hidden_dim_enc=HIDDEN_DIM_ENC,
                        hidden_dim_dec=HIDDEN_DIM_DEC,
                        dropout=DROPOUT,
                        padding_idx=src_pad_idx,
                        num_layers_enc=NUM_LAYERS_ENC,
                        num_layers_dec=NUM_LAYERS_DEC,
                        emb_dim_trg=EMB_DIM_TRG,
                        trg_pad_idx=trg_pad_idx,
                        device=DEVICE,
                        seq_beginning_token_idx=seq_beginning_token_idx,
                        train_attention=TRAIN_ATTENTION,
                    )

                    # Send model to device
                    model.send_to_device()

                    train_losses = []
                    val_losses = []
                    best_val_loss = float('inf')

                    for epoch in range(EPOCHS):

                        now = time()

                        # Init loss stats for epoch
                        train_loss = 0

                        n_batches_since_eval = 0

                        for n_batch, train_batch in enumerate(
                            tqdm(
                                train_loader,
                                desc=f"Epoch {epoch}",
                                unit="batch",
                                disable=disable_pro_bar,
                            )
                        ):

                            model.seq2seq.train()

                            # Take one gradient step
                            train_loss += model.train_step(
                                src_batch=train_batch.src[0],
                                trg_batch=train_batch.trg,
                                src_lens=train_batch.src[1],
                                teacher_forcing=TEACHER_FORCING,
                            )

                            n_batches_since_eval += 1

                            # Calculate and safe train/eval losses at 25% of epoch
                            if n_batch in eval_points and use_eval:
                
                                now_eval = time()

                                # Evaluate
                                eval_loss = evaluate(model=model, eval_loader=val_loader)

                                print(f"Evaluation time: {(time()-now_eval)/60:.2f} minutes.")

                                # Save mean train/val loss
                                train_losses.append(train_loss / n_batches_since_eval)
                                val_losses.append(eval_loss)

                                # Set counter to 0 again
                                n_batches_since_eval = 0
                                train_loss = 0

                                print(
                                    f"Epoch {epoch} [{round(n_batch*100/n_batches_train)}%]: Train loss [{train_losses[-1]}]   |  Val loss [{eval_loss}]\n"
                                )
                                print("##########################################\n")
                                # Save the model if the validation loss is the best we've seen so far
                                if eval_loss < best_val_loss:
                                    best_val_loss = eval_loss                
                                    torch.save(model.seq2seq.state_dict(), "train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth")
                                    print(f"Best model saved with validation loss: {eval_loss:.4f}")

                        print(f"Epoch Training time: {(time()-now)/60:.2f} minutes.")
                
                    save_object("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train_losses", train_losses)

                    save_object("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val_losses", val_losses)
                    
                    if not use_eval:
                        torch.save(model.seq2seq.state_dict(), "train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth")
                    else:
                        model.seq2seq.load_state_dict(torch.load("train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth", map_location=DEVICE))
                    
                    model.seq2seq.eval()
        
                    with torch.no_grad():

                        y_train_all = []
                        predict_train_all = []
                        pd_train = pd.read_csv("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_train_short_" + str(ws_use) + ".csv", sep = ">")
                        list_x = list(pd_train["x"])
                        list_y = list(pd_train["y"])
                        for i in range(len(list_x)):
                            translation, _, _, _ = translate_sentence(
                                sentence=str(list_x[i]),
                                seq2seq_model=model.seq2seq,
                                src_field=src_field,
                                bos=src_field.init_token,
                                eos=src_field.eos_token,
                                eos_idx=src_field.vocab.stoi[src_field.eos_token],
                                trg_field=trg_field,
                                max_len=33,
                            )
                            y_train_all.append([str(list_y[i])]) 
                            predict_train_all.append([translation]) 
                        print_predictions(y_train_all, predict_train_all, "train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train.csv")

                        y_val_all = []
                        predict_val_all = []
                        pd_val = pd.read_csv("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_val_short_" + str(ws_use) + ".csv", sep = ">")
                        list_x = list(pd_val["x"])
                        list_y = list(pd_val["y"])
                        for i in range(len(list_x)):
                            translation, _, _, _ = translate_sentence(
                                sentence=str(list_x[i]),
                                seq2seq_model=model.seq2seq,
                                src_field=src_field,
                                bos=src_field.init_token,
                                eos=src_field.eos_token,
                                eos_idx=src_field.vocab.stoi[src_field.eos_token],
                                trg_field=trg_field,
                                max_len=33,
                            )
                            y_val_all.append([str(list_y[i])]) 
                            predict_val_all.append([translation]) 
                        print_predictions(y_val_all, predict_val_all, "train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv")

                        y_test_all = []
                        predict_test_all = []
                        pd_test = pd.read_csv("tokenized_data/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + varname + "_test_short_" + str(ws_use) + ".csv", sep = ">")
                        list_x = list(pd_test["x"])
                        list_y = list(pd_test["y"])
                        for i in range(len(list_x)):
                            translation, _, _, _ = translate_sentence(
                                sentence=str(list_x[i]),
                                seq2seq_model=model.seq2seq,
                                src_field=src_field,
                                bos=src_field.init_token,
                                eos=src_field.eos_token,
                                eos_idx=src_field.vocab.stoi[src_field.eos_token],
                                trg_field=trg_field,
                                max_len=33,
                            )
                            y_test_all.append([str(list_y[i])]) 
                            predict_test_all.append([translation]) 
                        print_predictions(y_test_all, predict_test_all, "train_attention" + str(marking_for_range[ws_use]) + "/"+ str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv")