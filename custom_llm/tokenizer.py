import os
import json
import warnings

# Avoid annoying warnings
warnings.filterwarnings("ignore")

class RobustTokenizer:
    def __init__(self, save_dir="c:/Rarey Temp/Ai/With Ai/custom_llm/tokenizer_cache", force_type=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer_type = "char"
        self.vocab = {}
        self.inverse_vocab = {}
        self.encoder = None
        
        # Try loading Hugging Face tokenizer first (standard BPE) if not forced to char
        if force_type != "char":
            try:
                from transformers import GPT2TokenizerFast
                # Try to load from cache directory
                if os.path.exists(os.path.join(save_dir, "vocab.json")):
                    print("Loading saved GPT-2 BPE Tokenizer from local cache...")
                    self.encoder = GPT2TokenizerFast.from_pretrained(save_dir)
                    self.tokenizer_type = "bpe"
                else:
                    # Try downloading once (will cached locally)
                    print("Attempting to download GPT-2 BPE Tokenizer from Hugging Face...")
                    self.encoder = GPT2TokenizerFast.from_pretrained("gpt2")
                    self.encoder.save_pretrained(save_dir)
                    self.tokenizer_type = "bpe"
                    print("GPT-2 BPE Tokenizer downloaded and cached locally.")
            except Exception as e:
                print(f"Could not load/download HuggingFace BPE Tokenizer ({e}). Falling back to offline Character-Level Tokenizer.")

        if self.tokenizer_type == "char" or force_type == "char":
            self.tokenizer_type = "char"
            self.load_or_build_char_tokenizer()

    def load_or_build_char_tokenizer(self):
        vocab_path = os.path.join(self.save_dir, "char_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
            print(f"Loaded offline Character-Level Tokenizer. Vocab size: {len(self.vocab)}")
        else:
            # Build default basic character set (covers English, Hindi, Telugu, numbers, symbols)
            chars = (
                " \n\r\t!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
                "०१२३४५६७८९अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेैोौ्ँंः" # Hindi
                "౦౧౨౩౪౫౬౭౮౯అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహానీుూృెేైొోౌ్౦" # Telugu
            )
            unique_chars = sorted(list(set(chars)))
            self.vocab = {ch: i for i, ch in enumerate(unique_chars)}
            # Special tokens
            self.vocab["<|endoftext|>"] = len(self.vocab)
            self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}
            
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            print(f"Created new offline Character-Level Tokenizer. Vocab size: {len(self.vocab)}")

    @property
    def vocab_size(self):
        if self.tokenizer_type == "bpe":
            return self.encoder.vocab_size
        return len(self.vocab)

    def encode(self, text):
        if self.tokenizer_type == "bpe":
            return self.encoder.encode(text)
        
        # Character-level encoding (handle out of vocabulary characters gracefully)
        ids = []
        for ch in text:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            else:
                # Map unknown characters to space or ignore, we map to space (ID 0 usually)
                ids.append(self.vocab.get(" ", 0))
        return ids

    def decode(self, ids):
        if self.tokenizer_type == "bpe":
            return self.encoder.decode(ids)
            
        # Character-level decoding
        return "".join([self.inverse_vocab.get(idx, "") for idx in ids])

if __name__ == "__main__":
    tokenizer = RobustTokenizer()
    sample = "Hello! नमस्ते! నమస్తే!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {sample}")
    print(f"Encoded : {encoded[:15]}... (length: {len(encoded)})")
    print(f"Decoded : {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")
