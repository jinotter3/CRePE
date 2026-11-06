# Text prompt encoding for Wan2.1.
#
# Adapted from DiffSynth-Studio's WanPrompter (https://github.com/modelscope/DiffSynth-Studio,
# Apache-2.0). Tokenization, text cleaning and padding are unchanged; the
# ModelManager / prompt-refiner plumbing was removed.

import html
import string

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer

from .text_encoder import WanTextEncoder


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({
                "padding": "max_length",
                "truncation": True,
                "max_length": self.seq_len,
            })
        _kwargs.update(**kwargs)
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)
        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter:
    """umT5 tokenizer + encoder that returns Wan2.1 text context embeddings."""

    def __init__(self, tokenizer_path, text_encoder: WanTextEncoder = None, text_len: int = 512):
        self.text_len = text_len
        self.tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=text_len, clean="whitespace")
        self.text_encoder = text_encoder

    @torch.no_grad()
    def encode_prompt(self, prompt, device="cuda"):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb
