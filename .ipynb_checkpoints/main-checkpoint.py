# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores


logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(nn.Module):
    def __init__(self, hparams, model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        lm_labels = batch["target_ids"].to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device)
        )

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader)

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            lm_labels = batch["target_ids"].to(device)
            lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].to(device)
            )

            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_loader)


def evaluate(data_loader, model, sents, device):
    model.eval()
    outputs, targets = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                        attention_mask=batch['source_mask'].to(device), 
                                        max_length=128)

            dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

            outputs.extend(dec)
            targets.extend(target)

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents)
    scores["all_labels"] = all_labels
    scores["all_preds"] = all_preds

    return scores



# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP: ASQP on {args.dataset}", "="*30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Here is an example (from the dev set):")
train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length)
data_sample = train_dataset[7]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))


# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    model = T5FineTuner(args, T5ForConditionalGeneration.from_pretrained(args.model_name_or_path), tokenizer)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  drop_last=True, shuffle=True, num_workers=4)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
    # )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    print(1, args.n_gpu)
    t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, int(args.n_gpu)))) // args.gradient_accumulation_steps * float(args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    for epoch in range(int(args.num_train_epochs)):
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Training Loss: {train_loss:.4f}")

        val_dataset = get_dataset(tokenizer, "test", args)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4)
        val_loss = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Validation Loss: {val_loss:.4f}")


    # save the final model
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")


# evaluation
if args.do_direct_eval:
    print("\n****** Conduct Evaluating with the last state ******")

    # model = T5FineTuner(args)

    # print("Reload the model")
    # model.model.from_pretrained(args.output_dir)

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')


    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores = evaluate(test_loader, model, sents, device)

    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"F1 = {scores['f1']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)


if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

    model = T5FineTuner(args, tfm_model, tokenizer).to(device)

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores = evaluate(test_loader, model, sents, device)

    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"F1 = {scores['f1']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)