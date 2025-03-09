import re


class character_level_tokenizer:
    """
    character-level
    """

    def __init__(
        self, number_bits, pad_token="[PAD]", eos_token="[EOS]", mask_token="[MASK]"
    ):
        self.vocab = (
            [str(x) for x in range(0, 100)]
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

    # def encode(self, text):
    #     text_list = self.pre_tokenization(self.clean(text))
    #     return [self.token_to_id[c] for c in text_list]

    # def decode(self, token_list):
    #     return "".join([self.id_to_token[x] for x in token_list])

    def encode(self, text):
        text_list = self.pre_tokenization(self.clean(text))
        try:
            plus_idx = text_list.index("+")
            equal_idx = text_list.index("=")
        except ValueError:
            while len(text_list) < self.number_bits:
                text_list.insert(0, "0")
            return [self.token_to_id[c] for c in text_list]  # [::-1]
        # We pad the first number
        while plus_idx < self.number_bits:
            text_list.insert(0, "0")
            plus_idx += 1
            equal_idx += 1
        # We pad the second number
        while equal_idx < 2 * self.number_bits + 1:
            text_list.insert(plus_idx + 1, "0")
            equal_idx += 1
        # If "=" is not the last character, we pad the result of the operation.
        if text_list[-1] != "=":
            while len(text_list) < 3 * self.number_bits + 2:
                text_list.insert(2 * self.number_bits + 2, "0")
        final_text_list = text_list
        first_operand = text_list[: self.number_bits]
        plus_sign = text_list[self.number_bits]
        second_operand = text_list[self.number_bits + 1 : 2 * self.number_bits + 1]
        rest = text_list[2 * self.number_bits + 1 :]
        paired_digits = []
        # We pair the bits of the two numbers
        for i in range(self.number_bits):
            str_elt = first_operand[i] + second_operand[i]
            # If we pair 00,01,...,09 we just use the second element (0,1,...9)
            if len(str_elt) == 2 and str_elt[0] == "0":
                str_elt = str_elt[-1]
            paired_digits.append(str_elt)
            if i < self.number_bits - 1:
                paired_digits.append(self.pad_token)
        final_text_list = (
            [plus_sign] + paired_digits + [rest[0]] + rest[1:]
        )  # Add [::-1] to use the inverse version of the tokenizer
        return [self.token_to_id[c] for c in final_text_list]

    def decode(self, token_list):
        try:
            equal_idx = token_list.index(101)
        except ValueError:
            return "".join(
                [self.id_to_token[x] for x in token_list[-self.number_bits - 2 :]]
            )
        a = ""
        b = ""
        for token in token_list[: 2 * self.number_bits][
            1::2
        ]:  # Access odd-indexed elements directly
            a += str(token // 10)
            b += str(token % 10)
        return (
            str(int(a))
            + "+"
            + str(int(b))
            + "="
            + "".join(
                [self.id_to_token[x] for x in token_list[-self.number_bits - 1 :]]
            )
        )
