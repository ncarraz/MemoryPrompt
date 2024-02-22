import torch
import torch.nn as nn


class MemoryPrompt(nn.Module):
    def __init__(
        self,
        base_model,
        mlp_hidden_size=1024,
        mlp_output_size=1024,
        num_prompt_tokens=5,
    ):
        """Left-to-right language model augmented with a memory module that produces soft prompts after each turn.

        Args:
            base_model: Left-to-right language model to wrap
            mlp_hidden_size: Hidden size of the MLP
            mlp_output_size: Output size of the MLP
            num_tokens_soft_prompt: Number of soft prompt tokens taken as input and generated
        """
        super().__init__()
        self.base_model = base_model
        self.num_prompt_tokens = num_prompt_tokens
        self.word_embeddings = self.base_model.get_input_embeddings()
        self.embedding_dim = self.word_embeddings.weight.shape[1]
        self.vocab_size = self.word_embeddings.weight.shape[0]

        # Memory module
        self.mlp = _build_one_layer_mlp(
            self.embedding_dim, mlp_output_size, mlp_hidden_size
        )
        self.lstm = nn.LSTM(mlp_output_size, num_prompt_tokens * self.embedding_dim)

        # Initialize the MLP and LSTM weights
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1e-4)
                module.bias.data.fill_(0)

        self.mlp.apply(_init_weights)
        self.lstm.apply(_init_weights)

        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )
        self.curr_soft_prompt = None  # useful for generation

    def forward(
        self,
        cell_state=None,
        hidden_state=None,
        soft_prompt=None,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        """
        Produces the next soft_prompts, hidden state, cell state and logits (or loss when labels are provided)
        """
        assert (
            self.base_model.config.output_hidden_states == True
        ), "We need hidden states to compute the next soft prompt. Make sure that the base model has output_hidden_states=True"
        # Initialize hidden state and cell state
        if hidden_state is None:
            hidden_state = torch.zeros(
                1,
                input_ids.shape[0],
                self.embedding_dim * self.num_prompt_tokens,
                device=input_ids.device,
            )  # num_LSTM_layer, batch_size, num_tokens*embed_dim
            cell_state = torch.zeros(
                1,
                input_ids.shape[0],
                self.embedding_dim * self.num_prompt_tokens,
                device=input_ids.device,
            )
        if soft_prompt is None:
            # No soft prompt provided, use the base model to generate one
            # This happens during the first turn
            base_model_output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )
            if labels is None:
                input_lengths = attention_mask.sum(dim=1).to(torch.int64)
                logits = base_model_output.logits[
                    torch.arange(input_ids.shape[0]), input_lengths - 1
                ]  # logits of the last token
                output = logits
            else:
                output = base_model_output.loss
        else:
            # Adapt the intput to account for the soft prompts
            batch_size = input_ids.shape[0]
            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, soft_prompt.shape[1]).to(
                    input_ids.device
                )
                attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1
                )
            if labels is not None:
                labels = torch.cat(
                    (
                        torch.zeros(batch_size, soft_prompt.shape[1])
                        .fill_(-100)
                        .to(input_ids.device),
                        labels,
                    ),
                    dim=1,
                ).long()
            kwargs.update(
                {
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            # Base model takes as input the concatenation of the soft prompt and the input
            input_lengths = attention_mask.sum(dim=1).to(torch.int64)
            soft_prompt = soft_prompt.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1)
            base_model_output = self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            if labels is None:
                logits = base_model_output.logits[
                    torch.arange(input_ids.shape[0]), input_lengths - 1
                ]  # logits of the last token
                output = logits
            else:
                output = base_model_output.loss

        # MLP takes as input the last hidden state of the base model
        input_lengths = attention_mask.sum(dim=1).to(torch.int64)
        last_hidden_state = base_model_output.hidden_states[-1][
            torch.arange(input_ids.shape[0]), input_lengths - 1
        ]  # hidden state of the last token
        mlp_output = self.mlp(last_hidden_state)

        # LSTM cell computes the next soft prompt and cell state
        next_soft_prompt, (next_hidden_state, next_cell_state) = self.lstm(
            mlp_output.unsqueeze(0),
            (hidden_state, cell_state),
        )
        next_soft_prompt = next_soft_prompt.squeeze(0).reshape(
            input_ids.shape[0], self.num_prompt_tokens, self.embedding_dim
        )
        return (
            next_soft_prompt,
            next_hidden_state,
            next_cell_state,
            torch.nan_to_num(output),
        )

    def get_memory_module_state_dict(self):
        return {
            "mlp": self.mlp.state_dict(),
            "lstm": self.lstm.state_dict(),
        }

    def load_memory_module_state_dict(self, path):
        """load the state dict of the memory module"""
        state_dict = torch.load(path)
        self.mlp.load_state_dict(state_dict["mlp"])
        self.lstm.load_state_dict(state_dict["lstm"])

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None:
            inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
            prompts = self.curr_soft_prompt
            prompts = prompts.to(inputs_embeds.dtype)
            model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
            model_kwargs["input_ids"] = None
        return model_kwargs

    def generate(self, **kwargs):
        self.curr_soft_prompt = kwargs.pop("soft_prompt")
        self.base_model.prepare_inputs_for_generation = (
            self.prepare_inputs_for_generation
        )
        if kwargs.get("attention_mask", None) is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(
                kwargs["input_ids"].shape[0], self.num_prompt_tokens
            ).to(kwargs["input_ids"].device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, kwargs["attention_mask"]), dim=1
            )
        outputs = self.base_model.generate(**kwargs)
        self.base_model.prepare_inputs_for_generation = (
            self.base_model_prepare_inputs_for_generation
        )
        return outputs


def _build_one_layer_mlp(in_dim, out_dim, hidden_size):
    linear1 = nn.Linear(in_dim, hidden_size)
    relu = nn.ReLU()
    linear2 = nn.Linear(hidden_size, out_dim)
    return nn.Sequential(linear1, relu, linear2)
