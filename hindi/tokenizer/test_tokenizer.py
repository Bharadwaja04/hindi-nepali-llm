from .tokenizer import Tokenizer

def test_round_trip():
    text = "नेपाल सुन्दर देश हो।"
    tok = Tokenizer("nepali", vocab_size=276)
    tok.train(text, verbose=False)
    assert tok.verify_round_trip(text)

def test_hindi_round_trip():
    text = "भारत एक सुंदर देश है।"
    tok = Tokenizer("hindi", vocab_size=276)
    tok.train(text, verbose=False)
    assert tok.verify_round_trip(text)
