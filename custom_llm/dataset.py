import os
import torch
from torch.utils.data import Dataset
import json

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size=256, split="train", cache_dir="c:/Rarey Temp/Ai/With Ai/custom_llm/dataset_cache"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine dataset source
        text_data = ""
        local_txt_path = os.path.join(cache_dir, f"{split}_data.txt")
        
        if os.path.exists(local_txt_path):
            print(f"Loading local cached dataset for {split}...")
            with open(local_txt_path, "r", encoding="utf-8") as f:
                text_data = f.read()
        else:
            # Try to fetch from HuggingFace
            try:
                from datasets import load_dataset
                print(f"Downloading dataset from Hugging Face for {split}...")
                # Using wikitext-2-raw-v1 as it is small and fast to download
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                text_data = "\n".join(dataset["text"])
                
                # Save locally for offline use
                with open(local_txt_path, "w", encoding="utf-8") as f:
                    f.write(text_data)
                print("Dataset downloaded and saved locally.")
            except Exception as e:
                print(f"Could not load dataset from Hugging Face ({e}). Generating fallback local dataset...")
                text_data = self.generate_fallback_data(split)
                with open(local_txt_path, "w", encoding="utf-8") as f:
                    f.write(text_data)
        
        print("Tokenizing dataset text... (this may take a moment for large files)")
        self.tokens = torch.tensor(self.tokenizer.encode(text_data), dtype=torch.long)
        print(f"Dataset loaded. Total tokens: {len(self.tokens)}")

    def generate_fallback_data(self, split):
        # Multi-lingual fallback dataset
        english_stories = [
            "Once upon a time, there was a small bird. It loved to fly high in the blue sky.",
            "The computer is an electronic device that processes data. It makes calculations extremely fast.",
            "Artificial intelligence is training models to learn patterns. We train them using weights and optimization.",
            "Deep learning uses neural networks with many layers. PyTorch is a popular framework for training neural networks.",
            "The Dell G15 laptop has an RTX 3050 GPU. It is great for running local AI models with CUDA acceleration."
        ]
        
        hindi_stories = [
            "एक समय की बात है, एक जंगल में एक शेर रहता था। वह बहुत शक्तिशाली था।",
            "कृत्रिम बुद्धिमत्ता भविष्य की तकनीक है। यह कंप्यूटर को इंसानों की तरह सोचने में मदद करती है।",
            "पायटॉर्च एक लाइब्रेरी है जिसका उपयोग मशीन लर्निंग के लिए किया जाता है।",
            "लैपटॉप पर मॉडल चलाना अब आसान हो गया है। जीपीयू ग्राफिक्स कार्ड गति बढ़ाता है।"
        ]
        
        telugu_stories = [
            "అనగనగా ఒక ఊరిలో ఒక రాజు ఉండేవాడు. ఆయన చాలా దయాగుణం కలవాడు.",
            "కృత్రిమ మేధస్సు (AI) అనేది కంప్యూటర్ల ద్వారా మానవ మేధస్సును అనుకరించే సాంకేతికత.",
            "ఈ ల్యాప్‌టాప్ లో RTX 3050 గ్రాఫిక్స్ కార్డ్ ఉంది. ఇది AI మోడల్స్ శిక్షణకు చాలా సహాయపడుతుంది.",
            "తెలుగు భాష చాలా అందమైనది. ఇది దక్షిణ భారతదేశంలో మాట్లాడే భాష."
        ]
        
        # Mix them up
        all_sentences = english_stories + hindi_stories + telugu_stories
        repeated_data = []
        # Repeat sentences to build a small training corpus (~50KB)
        repeats = 100 if split == "train" else 20
        for _ in range(repeats):
            for sentence in all_sentences:
                repeated_data.append(sentence)
                
        return "\n".join(repeated_data)

    def __len__(self):
        # Number of overlapping chunks of length block_size
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of tokens at index idx
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]  # Target is shifted by 1 token to predict next token
        return x, y

if __name__ == "__main__":
    from tokenizer import RobustTokenizer
    tokenizer = RobustTokenizer()
    dataset = TextDataset(tokenizer, block_size=64, split="train")
    print(f"Dataset length (number of samples): {len(dataset)}")
    x, y = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"Decoded x: {tokenizer.decode(x.tolist())[:50]}...")
