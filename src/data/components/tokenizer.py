import math
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer


class P10Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="</s>",
        bos_token="<s>",
        max_length=128,
        **kwargs,
    ) -> None:
        self.max_length = max_length
        self.sign_tokens = ["+", "-"]
        self.digit_tokens = [str(i) for i in range(10)]
        self.exp_tokens = [f"E{i}" for i in range(-100, 101)]

        # Add T5 special tokens
        self.special_tokens = [
            pad_token,
            eos_token,
            unk_token,
            bos_token,
        ]
        self.vocab = (
            self.special_tokens + self.sign_tokens + self.digit_tokens + self.exp_tokens
        )

        self._token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self._id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            eos_token=eos_token,
            bos_token=bos_token,
            **kwargs,
        )

    @property
    def bos_token_id(self) -> Optional[int]:
        """Get the ID of the beginning of sequence token."""
        return self._token_to_id.get(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the ID of the end of sequence token."""
        return self._token_to_id.get(self.eos_token)

    @property
    def decoder_start_token_id(self) -> int:
        """Get the ID of decoder start token (should be bos_token_id for T5)."""
        return self._token_to_id.get(self.bos_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs adding special tokens."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_vocab(self) -> Dict[str, int]:
        return self._token_to_id.copy()

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        try:
            number = float(text)
            if number == 0:
                tokens = ["+", "0", "E0"]
                return tokens

            find_e = "e" in text.lower()
            exp_scientific = (
                int(text[text.index("e") + 1 :])
                if find_e
                else int(math.floor(math.log10(abs(number))))
            )

            sign = "+" if number >= 0 else "-"
            number = abs(number)

            number_str = str(number).replace(".", "").lower().split("e")[0]
            num_digits = min(self.max_length - 2, len(number_str))
            number_str = number_str[:num_digits]

            exp = exp_scientific - num_digits + 1

            tokens = [sign] + list(number_str) + [f"E{exp}"]
            return tokens

        except ValueError:
            return [self.unk_token]

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        tokens = [
            t
            for t in tokens
            if t not in [self.pad_token, self.bos_token, self.eos_token]
        ]
        if not tokens or tokens[0] not in ["+", "-"]:
            return self.unk_token

        try:
            sign = 1 if tokens[0] == "+" else -1

            exp_idx = -1
            for i, token in enumerate(tokens):
                if token.startswith("E"):
                    exp_idx = i
                    break

            if exp_idx == -1:
                return self.unk_token

            mantissa_tokens = tokens[1:exp_idx]
            mantissa = int("".join(mantissa_tokens))

            exp = int(tokens[exp_idx][1:])

            result = sign * mantissa * (10**exp)

            if abs(result) < 1e-100:
                return "0.0"
            return f"{result:.16g}"

        except:
            return self.unk_token

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # All special tokens are marked as 1
        special_tokens_ids = {self._token_to_id[token] for token in self.special_tokens}
        mask = [1 if token in special_tokens_ids else 0 for token in token_ids_0]
        if token_ids_1 is not None:
            mask += [1 if token in special_tokens_ids else 0 for token in token_ids_1]
        return mask

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        import json
        import os

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._token_to_id, f, ensure_ascii=False)

        return (vocab_file,)

    def batch_decode(
        self, sequences: List[List[int]], skip_special_tokens: bool = False, **kwargs
    ) -> List[str]:
        decoded = []
        for seq in sequences:
            tokens = [self._convert_id_to_token(idx) for idx in seq]
            if skip_special_tokens:
                tokens = [t for t in tokens if t not in self.special_tokens]
            text = self.convert_tokens_to_string(tokens)
            decoded.append(text)
        return decoded
