#testovací skrypt
#data/ obsahují testovací soubory a extrahovaná data  #!(to ještě přidám)

from src.methods import hamming_code
import pytest

@pytest.mark.parametrize("input_data, expected_encoded, expected_decoded", [
    ([0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1]),
    ([0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0]),
    ([0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1]),
    ([0, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0]),
    ([0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1]),
    ([0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0]),
    ([0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1], [0, 1, 1, 1]),
    ([1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0]),
    ([1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1]),
    ([1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0]),
    ([1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1]),
    ([1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0]),
    ([1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 1]),
    ([1, 1, 1, 0], [0, 1, 0, 1, 1, 1, 0], [1, 1, 1, 0]),
    ([1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1]),
])

def test_hamming_encode_and_decode(input_data, expected_encoded, expected_decoded):
    encoded = hamming_code.hamming_encode(input_data)
    decoded = hamming_code.hamming_decode(encoded)
    assert encoded == expected_encoded
    assert decoded == expected_decoded



