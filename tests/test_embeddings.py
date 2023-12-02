import pytest
from dataclasses import dataclass
import jax.numpy as jnp
from src.embeddings import encode_text, hash_embedding


@dataclass
class ProximityTestCase:
    reference: str
    similiar: str
    disimiliar: str


test_sentences = [
    "can you explain the mandelbrot set to me",
    "what is the captial city of pakistan?",
    "do you think artificial general intelligence is possible?",
]

metric_space_test_cases = [
    ProximityTestCase(
        reference="can you explain the mandelbrot set to me",
        similiar="can you explain the mandelbrot set to us",
        disimiliar="hola mi amigo que tal"
    ),
    ProximityTestCase(
        reference="what is 7 times 9",
        similiar="what is 4 plus 3",
        disimiliar="a fantastic evening it is tonight wouldn't you agree"
    ),
    ProximityTestCase(
        reference="elephants are great don't you think",
        similiar="don't you think elephants are great",
        disimiliar="describe the backpropogation algorithm to me like im 5"
    ),
]

@pytest.mark.parametrize("sentence", test_sentences)
def test_embedding_bounds(sentence: str):
    """Embeddings array should be bounded between -1 and 1."""
    tokenized_sentence = encode_text(sentence)
    embedding = hash_embedding(tokenized_sentence)
    assert ((embedding > -1) & (embedding < 1)).all()


@pytest.mark.parametrize("test_case", metric_space_test_cases)
def test_embedding_proximity(test_case: ProximityTestCase):
    """Similiar strings should be closer in embedding space."""

    def _cosine_distance(a: jnp.ndarray, b: jnp.ndarray):
        return a @ b / (jnp.sqrt(a @ a) * jnp.sqrt(b @ b))

    # Compute embeddings
    reference_embedding = hash_embedding(encode_text(test_case.reference))
    similiar_embedding = hash_embedding(encode_text(test_case.similiar))
    disimiliar_embedding = hash_embedding(encode_text(test_case.disimiliar))

    # Calc similarities
    similiar_similarity = _cosine_distance(reference_embedding, similiar_embedding)
    disimiliar_similarity = _cosine_distance(reference_embedding, disimiliar_embedding)

    assert similiar_similarity > disimiliar_similarity
