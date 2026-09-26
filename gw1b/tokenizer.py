"""Tokenizer training and loading.

The class tokenizer is a SentencePiece model (BPE or unigram) with byte fallback, trained by the
students on a sample of the pretraining corpus (Team 1). A thin wrapper gives a uniform interface,
and `export_hf()` writes a Hugging Face-compatible tokenizer for the public release.

    python -m gw1b.tokenizer train --input corpus.txt --out $GW1B_GROUP/tokenizer/gw1b-32k --vocab 32000 --type bpe
    python -m gw1b.tokenizer inspect --model $GW1B_GROUP/tokenizer/gw1b-32k.model --text "Hello Pegasus"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Iterable, Iterator

import sentencepiece as spm

SPECIAL = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}


def train_sentencepiece(inputs: str | list[str] | Iterable[str], model_prefix: str, vocab_size: int = 32000,
                        model_type: str = "bpe", input_sentence_size: int = 5_000_000, num_threads: int | None = None,
                        character_coverage: float = 0.9995, max_sentence_length: int = 16384,
                        extra_args: dict | None = None) -> str:
    """Train a SentencePiece model. `inputs` = a text file, a list of files, or an iterator of strings.

    Settings follow the Llama recipe: byte fallback (no UNK for rare characters), digits split
    individually, whitespace preserved, no unicode normalization. Returns the path to the .model file.
    """
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
    args = dict(
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,               # "bpe" | "unigram"
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        unk_id=SPECIAL["unk"], bos_id=SPECIAL["bos"], eos_id=SPECIAL["eos"], pad_id=SPECIAL["pad"],
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads or max(1, min(64, os.cpu_count() or 1)),
        train_extremely_large_corpus=True,
    )
    args.update(extra_args or {})
    if isinstance(inputs, (str, list)):
        spm.SentencePieceTrainer.train(input=inputs, **args)
    else:
        spm.SentencePieceTrainer.train(sentence_iterator=iter(inputs), **args)
    return model_prefix + ".model"


class Tokenizer:
    """Uniform wrapper around a SentencePiece model."""

    def __init__(self, model_path: str):
        self.path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = int(self.sp.get_piece_size())
        self.unk_id, self.bos_id, self.eos_id, self.pad_id = (self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id(),
                                                              self.sp.pad_id())
        with open(model_path, "rb") as f:
            self.sha256 = hashlib.sha256(f.read()).hexdigest()[:16]

    @classmethod
    def load(cls, model_path: str) -> "Tokenizer":
        return cls(model_path)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = True, num_threads: int | None = None) -> list[list[int]]:
        out = self.sp.encode(texts, out_type=int, num_threads=num_threads or -1)
        if add_eos and self.eos_id >= 0:
            out = [ids + [self.eos_id] for ids in out]
        return out

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(i) for i in ids])

    def pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def stats(self, text: str) -> dict:
        ids = self.encode(text)
        n_bytes = len(text.encode("utf-8"))
        return {"tokens": len(ids), "bytes": n_bytes, "bytes_per_token": n_bytes / max(1, len(ids)),
                "words_per_token": len(text.split()) / max(1, len(ids))}

    def export_hf(self, out_dir: str) -> str:
        """Write a Hugging Face `LlamaTokenizerFast` directory (tokenizer.json etc.) for release."""
        from transformers import LlamaTokenizer  # requires transformers + sentencepiece + protobuf
        tok = LlamaTokenizer(vocab_file=self.path, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                             pad_token="<pad>" if self.pad_id >= 0 else None, legacy=False, add_bos_token=False,
                             add_eos_token=False)
        os.makedirs(out_dir, exist_ok=True)
        tok.save_pretrained(out_dir)
        try:  # also write the fast tokenizer.json when the conversion is available
            from transformers import LlamaTokenizerFast
            LlamaTokenizerFast.from_pretrained(out_dir).save_pretrained(out_dir)
        except Exception as e:  # pragma: no cover - conversion is best effort
            print(f"[tokenizer] fast tokenizer conversion skipped: {e}")
        return out_dir


# ---------------------------------------------------------------------------
# alternative: Hugging Face `tokenizers` byte-level BPE (for Team 1 comparisons)
# ---------------------------------------------------------------------------
def train_hf_bpe(files: list[str], out_json: str, vocab_size: int = 32000) -> str:
    from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, trainers, decoders
    tok = HFTokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(files, trainer)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    tok.save(out_json)
    return out_json


def text_sample_iterator(paths: list[str], max_bytes: int) -> Iterator[str]:
    """Yield lines from text files until `max_bytes` have been produced."""
    n = 0
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line
                n += len(line)
                if n >= max_bytes:
                    return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GW1B tokenizer tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a SentencePiece tokenizer")
    t.add_argument("--input", required=True, nargs="+", help="text file(s), one document/sentence per line")
    t.add_argument("--out", required=True, help="output prefix, e.g. $GW1B_GROUP/tokenizer/gw1b-32k")
    t.add_argument("--vocab", type=int, default=32000)
    t.add_argument("--type", default="bpe", choices=["bpe", "unigram"])
    t.add_argument("--sentences", type=int, default=5_000_000, help="max sentences sampled for training")
    t.add_argument("--threads", type=int, default=None)
    i = sub.add_parser("inspect", help="tokenize a string and print pieces + stats")
    i.add_argument("--model", required=True)
    i.add_argument("--text", default="The George Washington University trains a language model on Pegasus.")
    e = sub.add_parser("export-hf", help="export a Hugging Face tokenizer directory")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        path = train_sentencepiece(a.input, a.out, vocab_size=a.vocab, model_type=a.type,
                                   input_sentence_size=a.sentences, num_threads=a.threads)
        tok = Tokenizer(path)
        meta = {"model": path, "vocab_size": tok.vocab_size, "type": a.type, "sha256": tok.sha256, "inputs": a.input}
        with open(a.out + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta, indent=2))
    elif a.cmd == "inspect":
        tok = Tokenizer(a.model)
        print(tok.pieces(a.text))
        print(tok.stats(a.text))
    elif a.cmd == "export-hf":
        print(Tokenizer(a.model).export_hf(a.out))


if __name__ == "__main__":
    main()
