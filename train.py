import numpy as np
from functools import partial
import argparse
import random
import yaml
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model import MemoryPrompt
from dataset import load_dataset, collate_fn


def partition_conversation(length, k):
    """Returns a list of tuples that partition a list of length `length` into blocks of size `k`
    starting from the end of the list
    """
    partitions = []
    for i in range(0, length, k):
        partitions.append((max(0, length - i - k), length - i))
    return partitions[::-1]

def report_gradient_norm(model):
    """Report the gradient norm of the model"""
    parameters = [
        p for p in model.parameters() if p.grad is not None  
    ]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        norms = [torch.norm(p.grad.detach(), 2).to(device) for p in parameters]
        total_norm = torch.norm(torch.stack(norms), 2.0).item()
        max_norm = max(norms)
        mean_norm = total_norm / len(parameters) if len(parameters) > 0 else 0.0
    return f"[Mean norm: {mean_norm:.2f} Max norm: {max_norm:.2f}]"


def train(
    train_dataloader,
    model,
    criterion,
    optimizer,
    epoch,
    accelerator,
    args,
):
    correct = 0
    sum = 0
    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
        conversation, label, prediction_mask = batch
        logits_list = []
        soft_prompt_norm = 0

        model.train()
        optimizer.zero_grad()
        soft_prompt, hidden_state, cell_state = None, None, None
        for i, turn in enumerate(conversation):
            soft_prompt, hidden_state, cell_state, logit = model(
                **turn,
                cell_state=cell_state,
                hidden_state=hidden_state,
                soft_prompt=soft_prompt,
            )
            logits_list.append(logit)
            soft_prompt_norm += torch.norm(soft_prompt, dim=-1).mean()

        soft_prompt_norm /= len(conversation)
        # batch_size, vocab_size, num_turns
        logits = torch.stack(logits_list).permute(1, 2, 0)
        loss = criterion(logits, label)
        loss = torch.sum(loss * prediction_mask) / torch.sum(prediction_mask)
        loss += args.soft_prompt_norm_weight * soft_prompt_norm

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        accelerator.backward(loss)

        optimizer.step()
        grad_info = report_gradient_norm(model)
        optimizer.zero_grad()

        loss = accelerator.gather(loss).mean()
        correct += torch.sum(torch.argmax(logits, dim=1) == label * prediction_mask)
        sum += torch.sum(prediction_mask)
        if batch_idx % args.output_freq == 0:
            accelerator.print(
                f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.2f}, Gradient: {grad_info}"
            )

    correct = accelerator.gather_for_metrics(correct).sum().item()
    sum = accelerator.gather_for_metrics(sum).sum().item()
    accuracy = correct / sum
    accelerator.print(f"Epoch: {epoch} Train Accuracy: {accuracy*100:.2f}")


@torch.no_grad()
def evaluate(dev_dataloader, model, criterion, accelerator):
    correct = 0
    sum = 0
    model.eval()
    for batch in tqdm(dev_dataloader):
        logits_list = []
        conversation, label, prediction_mask = batch
        soft_prompt, hidden_state, cell_state = None, None, None
        for turn in conversation:
            soft_prompt, hidden_state, cell_state, logit = model(
                **turn,
                cell_state=cell_state,
                hidden_state=hidden_state,
                soft_prompt=soft_prompt,
            )
            logits_list.append(logit)

        logits = torch.stack(logits_list).permute(1, 2, 0)
        loss = criterion(logits, label)
        loss = torch.sum(loss * prediction_mask) / torch.sum(prediction_mask)
        loss = accelerator.gather_for_metrics(loss).mean()
        correct += torch.sum(torch.argmax(logits, dim=1) == label * prediction_mask)
        sum += torch.sum(prediction_mask)

    correct = accelerator.gather_for_metrics(correct).sum().item()
    sum = accelerator.gather_for_metrics(sum).sum().item()
    accuracy = correct / sum
    accelerator.print(
        f"Validation Loss: {loss.item():.2f}, Validation Accuracy: {accuracy*100:.2f}"
    )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config", type=str, help="YAML config file that overrides arguments"
    )
    argparser.add_argument(
        "--train_dataset", type=str, help="train dataset in jsonl format"
    )
    argparser.add_argument(
        "--validation_dataset", type=str, help="validation dataset in jsonl format"
    )
    argparser.add_argument(
        "--demonstration_dataset",
        type=str,
        default="data/LAMA_TREx_test",
        help="directory containing demonstrations for each relation",
    )
    argparser.add_argument(
        "--template",
        type=str,
        default="data/relations.jsonl",
        help="template file with a line for each relation",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="model name on huggingface hub",
    )
    argparser.add_argument(
        "--eval_only",
        action="store_true",
        help="evaluate the model without training"
    )
    argparser.add_argument(
        "--checkpoint_name", type=str, help="path to save the memory module"
    )
    argparser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="path to a trained memory module",
    )
    argparser.add_argument("--batch_size", type=int, default=8, help="train bacth size")
    argparser.add_argument(
        "--test_batch_size", type=int, default=4, help="test batch size"
    )
    argparser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=7e-5, help="learning rate")
    argparser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="weight decay on the model parameters",
    )
    argparser.add_argument(
        "--soft_prompt_norm_weight",
        type=float,
        default=0.1,
        help="weight_decay on the memory vectors",
    )
    argparser.add_argument(
        "--grad_clip", type=float, default=1.0, help="gradient clipping"
    )
    argparser.add_argument(
        "--num_prompt_tokens", type=int, default=5, help="number of memory vectors"
    )
    argparser.add_argument(
        "--mlp_hidden_size", type=int, default=1024, help="hidden size of the MLP"
    )
    argparser.add_argument(
        "--mlp_output_size", type=int, default=1024, help="output size of the MLP"
    )
    argparser.add_argument(
        "--num_demonstrations",
        type=int,
        default=4,
        help="number of deonstration facts before the question",
    )
    argparser.add_argument(
        "--output_freq",
        type=int,
        default=100,
        help="frequency of printing training loss",
    )
    argparser.add_argument(
        "--block_size", type=int, default=5, help="number of facts in a block"
    )
    argparser.add_argument(
        "--save_freq", type=int, default=3, help="frequency of saving checkpoints"
    )
    argparser.add_argument(
        "--eval_freq",
        type=int,
        default=3,
        help="frequency of evaluating on validation set",
    )
    argparser.add_argument("--seed", type=int, default=2, help="random seed")
    args = argparser.parse_args()
    # Override arguments with config file
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # update args with config file
        args.__dict__.update(config)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # accelerator
    accelerator = Accelerator()
    accelerator.print(args)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, output_hidden_states=True
    )

    model = MemoryPrompt(
        base_model,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_output_size=args.mlp_output_size,
        num_prompt_tokens=args.num_prompt_tokens,
    )

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    if args.resume_checkpoint is not None:
        model.load_memory_module_state_dict(args.resume_checkpoint)

    # Load dataset
    train_dataset = load_dataset(
        args.train_dataset,
        args.demonstration_dataset,
        args.template,
        tokenizer,
        args.num_demonstrations,
        args.block_size,
    )
    dev_dataset = load_dataset(
        args.validation_dataset,
        args.demonstration_dataset,
        args.template,
        tokenizer,
        args.num_demonstrations,
        args.block_size,
    )

    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer,
        pin_memory=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn_with_tokenizer,
        shuffle=False,
        pin_memory=True,
    )

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Loss
    criterion = nn.CrossEntropyLoss(reduction="none")

    (
        model,
        optimizer,
        train_dataloader,
        dev_dataloader,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, 
    )

    if args.eval_only:
        evaluate(dev_dataloader, model, criterion, accelerator)
        return

    for epoch in range(1, args.epochs + 1):
        train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            accelerator,
            args,
        )
        if epoch % args.eval_freq == 0:
            evaluate(dev_dataloader, model, criterion, accelerator)
        if epoch % args.save_freq == 0:
            model = accelerator.unwrap_model(model)
            if args.augmented_model_type == "memoryprompt":
                memory_module_state_dict = model.get_memory_module_state_dict()
                accelerator.save(
                    memory_module_state_dict,
                    f"checkpoints/{args.checkpoint_name}_checkpoint_{epoch}.pt",
                )
            else:
                accelerator.save(
                    model.state_dict(),
                    f"checkpoints/{args.checkpoint_name}_checkpoint_{epoch}.pt",
                )


if __name__ == "__main__":
    main()
