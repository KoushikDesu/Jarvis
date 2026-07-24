import os
import sys
import torch
import shutil

# Make sure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import GPT

def convert_custom_to_hf(checkpoint_path, output_dir):
    """
    Converts custom nanoGPT checkpoint to standard HuggingFace GPT2LMHeadModel.
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        print("Please install transformers: pip install transformers")
        return False
        
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return False
        
    print(f"Loading custom checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    custom_config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    print("Initializing HuggingFace GPT-2 model...")
    # Map custom config parameters to Hugging Face GPT-2 Config
    hf_config = GPT2Config(
        vocab_size=custom_config.vocab_size,
        n_positions=custom_config.block_size,
        n_ctx=custom_config.block_size,
        n_embd=custom_config.n_embd,
        n_layer=custom_config.n_layer,
        n_head=custom_config.n_head,
        activation_function="gelu",
        resid_pdrop=custom_config.dropout,
        embd_pdrop=custom_config.dropout,
        attn_pdrop=custom_config.dropout,
        use_cache=True
    )
    
    hf_model = GPT2LMHeadModel(hf_config)
    
    print("Mapping weights from custom model to Hugging Face...")
    # HF GPT-2 uses Conv1D instead of nn.Linear. This means weights are transposed [in_features, out_features]
    hf_sd = hf_model.state_dict()
    
    # 1. Map embeddings and layer norm
    hf_sd['transformer.wte.weight'].copy_(state_dict['transformer.wte.weight'])
    hf_sd['transformer.wpe.weight'].copy_(state_dict['transformer.wpe.weight'])
    hf_sd['transformer.ln_f.weight'].copy_(state_dict['transformer.ln_f.weight'])
    hf_sd['transformer.ln_f.bias'].copy_(state_dict['transformer.ln_f.bias'])
    
    # Check if weights are tied or separate
    if 'lm_head.weight' in state_dict:
        hf_sd['lm_head.weight'].copy_(state_dict['lm_head.weight'])
    else:
        hf_sd['lm_head.weight'].copy_(state_dict['transformer.wte.weight'])
        
    # 2. Map layers
    for i in range(custom_config.n_layer):
        # Layer Norms
        hf_sd[f'transformer.h.{i}.ln_1.weight'].copy_(state_dict[f'transformer.h.{i}.ln_1.weight'])
        hf_sd[f'transformer.h.{i}.ln_1.bias'].copy_(state_dict[f'transformer.h.{i}.ln_1.bias'])
        hf_sd[f'transformer.h.{i}.ln_2.weight'].copy_(state_dict[f'transformer.h.{i}.ln_2.weight'])
        hf_sd[f'transformer.h.{i}.ln_2.bias'].copy_(state_dict[f'transformer.h.{i}.ln_2.bias'])
        
        # Attention c_attn (Linear -> HF Conv1D needs transpose)
        hf_sd[f'transformer.h.{i}.attn.c_attn.weight'].copy_(state_dict[f'transformer.h.{i}.attn.c_attn.weight'].t())
        if f'transformer.h.{i}.attn.c_attn.bias' in state_dict:
            hf_sd[f'transformer.h.{i}.attn.c_attn.bias'].copy_(state_dict[f'transformer.h.{i}.attn.c_attn.bias'])
            
        # Attention c_proj (Linear -> HF Conv1D needs transpose)
        hf_sd[f'transformer.h.{i}.attn.c_proj.weight'].copy_(state_dict[f'transformer.h.{i}.attn.c_proj.weight'].t())
        if f'transformer.h.{i}.attn.c_proj.bias' in state_dict:
            hf_sd[f'transformer.h.{i}.attn.c_proj.bias'].copy_(state_dict[f'transformer.h.{i}.attn.c_proj.bias'])
            
        # MLP c_fc (Linear -> HF Conv1D needs transpose)
        hf_sd[f'transformer.h.{i}.mlp.c_fc.weight'].copy_(state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].t())
        if f'transformer.h.{i}.mlp.c_fc.bias' in state_dict:
            hf_sd[f'transformer.h.{i}.mlp.c_fc.bias'].copy_(state_dict[f'transformer.h.{i}.mlp.c_fc.bias'])
            
        # MLP c_proj (Linear -> HF Conv1D needs transpose)
        hf_sd[f'transformer.h.{i}.mlp.c_proj.weight'].copy_(state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].t())
        if f'transformer.h.{i}.mlp.c_proj.bias' in state_dict:
            hf_sd[f'transformer.h.{i}.mlp.c_proj.bias'].copy_(state_dict[f'transformer.h.{i}.mlp.c_proj.bias'])
            
    hf_model.load_state_dict(hf_sd)
    print(f"Saving Hugging Face model folder to {output_dir}...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    hf_model.save_pretrained(output_dir)
    
    # Save the standard GPT-2 tokenizer configuration files so llama.cpp knows the vocabulary type
    # We copy basic GPT-2 tokenizer files since our robust tokenizer uses standard GPT-2 BPE vocabulary tokens
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)
    
    print("\n[SUCCESS] Custom model weights converted to Hugging Face format successfully!")
    print(f"To compile this folder to a .gguf file, install llama.cpp and run:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile model.gguf")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to checkpoint.pt")
    parser.add_argument("--output", type=str, default="hf_gpt2_model", help="Output Hugging Face directory")
    args = parser.parse_args()
    
    convert_custom_to_hf(args.checkpoint, args.output)
