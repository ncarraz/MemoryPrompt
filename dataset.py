import os
import pandas as pd
import numpy as np
import random
import torch
from datasets import Dataset


def templatize(template, sub_label, obj_label=None):
    """Templatize a fact using the template.

    Args:
        template: Template string containing [X] and [Y] placeholders.
        sub_label: Subject label that will replace [X].
        obj_label: Object label that will replace [Y]. If None, only the left part of the template is used.

    Returns:
        Templatized fact.
    """
    if obj_label is None:
        # Only left part of the template is used
        return template.replace("[X]", sub_label).split("[Y]")[0].strip()
    else:
        return (
            template.replace("[X]", sub_label)
            .replace("[Y]", obj_label)
            .replace(" .", ".")
        )


def get_demonstrations(num_demo, fact_data_path, template_dict):
    """Get demonstration examples for each relation to be used for few-shot prompting

    Args:
        num_demo: Nummber of demonstration examples to use for each relation.
        fact_data_path: Path to the directory containing the fact data.
        template_dict: Dictionary containing the templates for each relation.

    Returns:
        Dictionary containing the demonstration examples for each relation.
    """
    demonstrations = {}
    fact_data_dict = {
        relation: pd.read_json(
            os.path.join(fact_data_path, relation + ".jsonl"),
            lines=True,
            orient="records",
        )
        for relation in template_dict.keys()
    }
    for relation in template_dict.keys():
        template = template_dict[relation]
        data = fact_data_dict[relation]
        ids = random.sample(range(len(data)), num_demo)
        facts = data.iloc[ids]
        demo = facts.apply(
            lambda x: templatize(template, x["sub_label"], x["obj_label"]), axis=1
        ).tolist()
        demonstrations[relation] = " ".join(demo)
    return demonstrations


def prepare_dataset(
    batch, template_dict, demonstrations=None, block_size=5
):
    """Segment the dataset into block_size blocks and add labels

    Args:
        batch: batch of examples.
        template_dict: Dictionary containing the templates for each relation.
        demonstrations: Dictionary containing the demonstration examples for each relation.
        block_size: Number of facts(turns) in each block.
    Returns:
        Prepared bactch.
    """
    last_label_mask = [] # Identifies the last label position
    labels = []
    result_prompts = []
    prompts = [
        templatize(template_dict[batch["relation"][i]],
            batch["sub_label"][i],
            batch["obj_label"][i],
        )
        for i in range(len(batch["relation"]))
    ]
    # Split the lists into blocks containing block_size prompts
    prompts = [
        "".join(prompts[i : i + block_size]) for i in range(0, len(prompts), block_size)
    ]
    i = 0
    for b, block in enumerate(prompts):
        result_prompts.append(block)
        labels.append(" ")
        last_label_mask.append(0)
        if b == len(prompts) - 1:
            past_sub = batch["sub_label"][: i + block_size]
            past_mask = batch["pivot_mask"][: i + block_size]
            all_subs = set([v for j, v in enumerate(past_sub) if past_mask[j] == 1])
            if len(all_subs) > 0:
                for sub in all_subs:
                    # get the last updated subject
                    idx_to_ask = np.argwhere(np.array(past_sub) == sub)[-1].item()
                    asking_prompt = templatize(
                        template_dict[batch["relation"][idx_to_ask]],
                        batch["sub_label"][idx_to_ask],
                    )
                    if demonstrations is not None:
                        asking_prompt = (
                            demonstrations[batch["relation"][idx_to_ask]]
                            + " "
                            + asking_prompt
                        )
                    result_prompts.append(asking_prompt)
                    labels.append(batch["obj_label"][idx_to_ask])
                    last_label_mask.append(1)

        i = i + block_size  # next block
    batch["label"] = labels
    batch["prompt"] = result_prompts
    batch["last_label_mask"] = last_label_mask 
    return batch


def tokenize(batch, tokenizer):
    batch["input"] = [tokenizer(i).input_ids for i in batch["prompt"]]
    batch["labels"] = [
        tokenizer(" " + i, add_special_tokens=False).input_ids[0]
        for i in batch["label"]
    ]
    batch["num_tokens"] = [len(tokenizer(i).input_ids) for i in batch["prompt"]]
    return batch


def load_dataset(
    dataset_path,
    demonstration_dataset_path,
    template_path,
    tokenizer,
    num_demonstrations,
    block_size=5,
):
    """Load the dataset and prepare it for training.

    Args:
        dataset_path: Path to the jsonl file containing the dataset.
        demonstration_dataset_path: Path to the jsonl file containing the dataset to collect demonstrations from.
        template_path: Path to the jsonl file containing the templates for each relation.
        tokenizer: Tokenizer of the base model.
        num_demonstrations: Number of demonstrations to collect for each relation.
        block_size: Number of facts(turns) in each block.

    Returns:
        Dataset object.
    """
    lama_templates = pd.read_json(template_path, lines=True, orient="records")
    template_dict = {
        row["relation"]: row["template"] for i, row in lama_templates.iterrows()
    }

    dataset_df = pd.read_json(dataset_path, lines=True, orient="records")
    dataset = Dataset.from_pandas(dataset_df)
    demonstrations = (
        None
        if num_demonstrations == 0
        else get_demonstrations(
            num_demonstrations, demonstration_dataset_path, template_dict
        )
    )
    dataset = dataset.map(
        lambda example: {"num_updates": example["pivot_mask"].count(1)}
    )
    # Prepare and tokenize the dataset
    dataset = dataset.map(
            prepare_dataset,
            fn_kwargs={
                "template_dict": template_dict,
                "demonstrations": demonstrations,
                "block_size": block_size,
            },
        )
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["relation", "sub_label", "obj_label", "pivot_mask"],
    )
    return dataset


def collate_fn(batch, tokenizer, add_metadata=False):
    """Pad the conversations and turns to have the same length."""

    def pad_list_of_list(list_of_list, padding_value):
        max_len = max([len(s) for s in list_of_list])
        return [s + [padding_value] * (max_len - len(s)) for s in list_of_list]

    inputs = [sample["input"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    mask = [sample["last_label_mask"] for sample in batch]

    # Pad the conversations
    inputs_padded = np.array(
        pad_list_of_list(inputs, [tokenizer.pad_token_id]), dtype=object
    )
    labels_padded = torch.tensor(pad_list_of_list(labels, -100))
    mask_padded = torch.tensor(pad_list_of_list(mask, 0))

    # Pad the turns
    results = []
    for turn_id in range(inputs_padded.shape[1]):
        turn = inputs_padded[:, turn_id].tolist()
        result = tokenizer.pad(
            {"input_ids": turn},
            padding=True,
            return_tensors="pt",
        )
        results.append(result)
    if add_metadata:
        num_updates = [sample["num_updates"] for sample in batch]
        num_turns = [sample["turns"] for sample in batch]
        num_distractors = [sample["num_distractors"] for sample in batch]
        metadata = {
            "num_updates": torch.tensor(num_updates),
            "num_turns": torch.tensor(num_turns),
            "num_distractors": torch.tensor(num_distractors),
        }
        return results, labels_padded, mask_padded, metadata
    else:
        return results, labels_padded, mask_padded