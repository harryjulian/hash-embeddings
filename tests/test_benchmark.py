from src.embeddings import encode_text, hash_embedding


# Kept all the repetitiom here to make the benchmark results legible
def test_benchmark_hash_embedding_short(benchmark):
    sentence = "hello how are you"
    encoded_sentence = encode_text(sentence)
    benchmark(hash_embedding, encoded_sentence)


def test_benchmark_hash_embedding_medium(benchmark):
    sentence = "i want to know about the life of the roman emporer marcus aurelias"
    encoded_sentence = encode_text(sentence)
    benchmark(hash_embedding, encoded_sentence)


def test_benchmark_hash_embedding_long(benchmark):
    sentence = """
        can you help me understand when variables go out of scope in rust?
        i don't understand the concept of lifetimes very well and could use 
        some advice -- i need you to explain this to me intuitively
    """
    encoded_sentence = encode_text(sentence)
    benchmark(hash_embedding, encoded_sentence)