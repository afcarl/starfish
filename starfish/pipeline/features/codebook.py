import json


class Codebook:

    def __init__(self, code_array):

        # do some codebook validation
        for code in code_array:

            if not isinstance(code, dict):
                raise ValueError(f'codebook must be an array of dictionary codes. Found: {code}.')

            # verify all necessary fields are present
            required_fields = {'codeword', 'gene_name'}
            missing_fields = required_fields.difference(code)
            if missing_fields:
                raise ValueError(
                    f'Each entry of codebook must contain {required_fields}. Missing fields: {missing_fields}')

        self._codes = code_array

    @property
    def codes(self) -> dict:
        return self._codes

    @classmethod
    def from_json(cls, json_codebook):
        with open(json_codebook, 'r') as f:
            return cls(json.load(f))
