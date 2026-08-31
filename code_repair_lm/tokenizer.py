import re
from collections import Counter
from typing import Iterable, List, Dict


class CodeTokenizer:
    def __init__(self, vocab_size: int = 2048, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<MASK>": 4,
        }
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.unk_token_id = self.special_tokens["<UNK>"]
        self.bos_token_id = self.special_tokens["<BOS>"]
        self.eos_token_id = self.special_tokens["<EOS>"]
        self.mask_token_id = self.special_tokens["<MASK>"]
        self.token_to_id = {k: v for k, v in self.special_tokens.items()}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.min_freq = 1
        self._frozen = False
        self._seed_tokens = [
            "def", "return", "class", "if", "else", "elif", "for", "while", "in",
            "try", "except", "finally", "with", "as", "import", "from", "pass",
            "break", "continue", "raise", "assert", "lambda", "yield", "True", "False",
            "None", "and", "or", "not", "print", "len", "sum", "list", "dict", "set",
            "tuple", "str", "int", "float", "bool", "range", "enumerate", "zip",
            "input", "open", "type", "self", "cls", "super", "@", "=", ":",
            ";", ",", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/", "%",
            "==", "!=", "<=", ">=", "<", ">", "!", "&", "|", "^", "~", "\n",
            "\t", "\r", ".", "#", "'", '"', "`", "->", "\"\"\""
        ]
        for token in self._seed_tokens:
            self._ensure_token(token)

    def _ensure_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self._frozen or len(self.token_to_id) >= self.vocab_size:
            return self.unk_token_id
        next_id = len(self.token_to_id)
        self.token_to_id[token] = next_id
        self.id_to_token[next_id] = token
        return next_id

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        token_pattern = re.compile(
            r"\s+|\d+(?:\.\d+)?|0[xX][0-9A-Fa-f]+|==|!=|<=|>=|->|[A-Za-z_][A-Za-z0-9_]*|\.|\(|\)|\{|\}|\[|\]|,|:|;|\+|-|\*|/|%|=|<|>|!|&|\||\^|~|@|\\|#|\'|\"|`",
            re.MULTILINE,
        )
        out: List[str] = []
        last_index = 0
        for match in token_pattern.finditer(text):
            start, end = match.span()
            if start > last_index:
                out.extend(list(text[last_index:start]))
            out.append(match.group(0))
            last_index = end
        if last_index < len(text):
            out.extend(list(text[last_index:]))
        if not out:
            return list(text)
        return out

    def build_vocab(self, corpus: Iterable[str]) -> None:
        counts = Counter()
        for text in corpus:
            for token in self._tokenize(text):
                counts[token] += 1
        ordered = [token for token, _ in counts.most_common() if token not in self.token_to_id]
        for token in ordered:
            if len(self.token_to_id) >= self.vocab_size:
                break
            self._ensure_token(token)
        self._frozen = True

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            elif not self._frozen and len(self.token_to_id) < self.vocab_size:
                ids.append(self._ensure_token(token))
            else:
                ids.append(self.unk_token_id)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        pieces = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if token in {"<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"}:
                    continue
                pieces.append(token)
        text = ''.join(pieces)
        return text

    def __len__(self):
        return len(self.token_to_id)
