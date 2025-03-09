import re


class character_level_tokenizer:
    """
    character-level
    """

    def __init__(
        self, number_bits, pad_token="[PAD]", eos_token="[EOS]", mask_token="[MASK]"
    ):
        self.vocab = (
            [str(x) for x in range(0, 10)]
            + ["+", "="]
            + [pad_token, eos_token]
            + [mask_token]
        )
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.ntokens = len(self.vocab)
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

        self.number_bits = number_bits

    def clean(self, text):
        """
        removes all characters not in the vocabulary
        """
        out = re.sub(self.pattern, "", text)
        return out

    def pre_tokenization(self, text):
        """
        character-level
        """
        return [c for c in text]

    def encode(self, text):
        text_list = self.pre_tokenization(self.clean(text))
        return [self.token_to_id[c] for c in text_list]

    def decode(self, token_list):
        return "".join([self.id_to_token[x] for x in token_list])
