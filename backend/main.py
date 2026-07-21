import os
import sys
import base64
import shutil
import urllib.request
import urllib.error
import json
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add workspace directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.web_search import get_search_context, is_connected
from backend.document_parser import DocumentParser
from backend.training_manager import TrainingManager

app = FastAPI(title="Gemini Clone Backend", description="FastAPI server for custom GPT training and chatbot operations")

# Enable CORS so frontend files can call the api local port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp storage directory for uploads
UPLOAD_DIR = "c:/Rarey Temp/Ai/With Ai/backend/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Instantiations
doc_parser = DocumentParser()
train_manager = TrainingManager()

# Custom LLM Lazy load states
custom_model = None
custom_tokenizer = None

import re

def download_file_from_google_drive(share_link: str, destination: str):
    """Download a public file from Google Drive using its share link."""
    file_id = None
    file_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', share_link)
    if file_id_match:
        file_id = file_id_match.group(1)
    else:
        id_param_match = re.search(r'id=([a-zA-Z0-9-_]+)', share_link)
        if id_param_match:
            file_id = id_param_match.group(1)
            
    if not file_id:
        return False
        
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    try:
        print(f"Downloading checkpoint from Google Drive ID: {file_id}...")
        req = urllib.request.Request(
            download_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Failed to download Google Drive checkpoint: {e}")
        return False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    model_type: str  # "custom", "gemini", "ollama"
    use_search: bool
    context_text: Optional[str] = ""
    image_b64: Optional[str] = None
    image_mime: Optional[str] = None
    ollama_model_name: Optional[str] = "phi3"

def run_custom_model_inference(prompt: str, max_tokens: int = 150):
    """Load model checkpoint and generate response locally."""
    global custom_model, custom_tokenizer
    import torch
    
    # Try the cloud-downloaded checkpoint first, then fallback to Drive sync or local path
    checkpoint_path = "c:/Rarey Temp/Ai/With Ai/custom_llm/checkpoint_cloud.pt"
    if not os.path.exists(checkpoint_path):
        g_drive_path = "G:/My Drive/CustomLLM/checkpoint.pt"
        checkpoint_path = g_drive_path if os.path.exists(g_drive_path) else "c:/Rarey Temp/Ai/With Ai/custom_llm/checkpoint.pt"
        
    if not os.path.exists(checkpoint_path):
        return f"System error: Checkpoint file not found. Checked: '{checkpoint_path}'. Please run training first!"
        
    try:
        if custom_model is None:
            sys.path.append("c:/Rarey Temp/Ai/With Ai/custom_llm")
            from model import GPT
            from tokenizer import RobustTokenizer
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading custom model to {device} for inference...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint['config']
            
            custom_model = GPT(config)
            custom_model.load_state_dict(checkpoint['model_state_dict'])
            custom_model.to(device)
            custom_model.eval()
            # Force character tokenizer if model vocab size is small
            force_type = "char" if config.vocab_size < 1000 else None
            custom_tokenizer = RobustTokenizer(force_type=force_type)
            
        device = next(custom_model.parameters()).device
        
        # Tokenize prompt
        input_ids = custom_tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Limit prompt size if it exceeds context window
        block_size = custom_model.config.block_size
        max_tokens = min(max_tokens, block_size - 1)
        avail_context = block_size - max_tokens
        if input_tensor.size(1) > avail_context:
            input_tensor = input_tensor[:, -avail_context:]
            
        # Generate tokens
        with torch.no_grad():
            generated_tensor = custom_model.generate(
                input_tensor, 
                max_new_tokens=max_tokens, 
                temperature=0.7, 
                top_k=20
            )
            
        # Extract only the generated tokens (slice off the input tokens)
        gen_tokens = generated_tensor[0, input_tensor.size(1):].tolist()
        response_text = custom_tokenizer.decode(gen_tokens)
        
        # Clean special tokens
        response_text = response_text.replace("<|endoftext|>", "").strip()
        return response_text
    except Exception as e:
        return f"Failed to execute local inference on custom model: {str(e)}"

def call_gemini_api(request: ChatRequest, api_key: str):
    """Direct HTTP request to Google Gemini API (no external packages needed)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    # 1. Build prompt context (Web search results + File extracts)
    system_instruction = "You are a helpful multimodal AI assistant modeled after Gemini. "
    system_instruction += "You speak Hindi, Telugu, and English. Respond in the language requested by the user. "
    if request.context_text:
        system_instruction += f"\nUploaded Document Context:\n{request.context_text}\n"

    # Assemble request payload
    parts = []
    
    # Add textual prompt (with search results if enabled)
    prompt_text = request.message
    if request.use_search:
        search_ctx = get_search_context(request.message)
        prompt_text = f"{search_ctx}\nUser Question: {prompt_text}"
        
    parts.append({"text": prompt_text})
    
    # Add image inlineData if provided (multimodal)
    if request.image_b64:
        # Clean base64 header if present (e.g., data:image/png;base64,...)
        b64_data = request.image_b64
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
            
        mime_type = request.image_mime or "image/png"
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data
            }
        })
        
    # Translate history for Gemini schema
    contents = []
    for msg in request.history:
        contents.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [{"text": msg.content}]
        })
        
    # Append the current active request parts
    contents.append({
        "role": "user",
        "parts": parts
    })
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
        # Parse candidate response
        output_text = res_data['candidates'][0]['content']['parts'][0]['text']
        return output_text
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"Gemini API Error details: {error_body}")
        try:
            err_json = json.loads(error_body)
            msg = err_json['error']['message']
            return f"Gemini API Error: {msg}"
        except:
            return f"Gemini API Error (HTTP {e.code}): {e.reason}"
    except Exception as e:
        return f"Failed to reach Gemini API: {str(e)}"

def call_ollama_api(request: ChatRequest):
    """Query a local Ollama server running on the user's laptop."""
    url = "http://localhost:11434/api/chat"
    
    system_prompt = "You are a helpful AI assistant. "
    if request.context_text:
        system_prompt += f"\nContext:\n{request.context_text}\n"
    if request.use_search:
        search_ctx = get_search_context(request.message)
        system_prompt += f"\nWeb Search Context:\n{search_ctx}\n"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
        
    # Add active user prompt
    messages.append({"role": "user", "content": request.message})
    
    payload = {
        "model": request.ollama_model_name,
        "messages": messages,
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode('utf-8'))
        return res_data['message']['content']
    except Exception as e:
        return f"Failed to connect to local Ollama. Make sure Ollama app is running locally and model '{request.ollama_model_name}' is pulled (Run: 'ollama run {request.ollama_model_name}' in cmd). Error: {str(e)}"

@app.get("/")
def home():
    return {
        "status": "online",
        "description": "FastAPI AI Engine backend running. Control Panel & Model Router active.",
        "internet_connected": is_connected()
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload documents or pictures, process them, and return extracted text."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ext = os.path.splitext(file.filename)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # Extract text via DocumentParser
        extracted_text = doc_parser.parse_file(file_path)
        
        # Read file as Base64 if it's an image (for direct Gemini vision pass)
        b64_data = None
        if is_image:
            with open(file_path, "rb") as image_file:
                b64_data = base64.b64encode(image_file.read()).decode('utf-8')
                
        # Clean up file after parsing (to save space)
        if not is_image:  # Keep image on disk in case we need it, delete others
            os.remove(file_path)
            
        return {
            "success": True,
            "filename": file.filename,
            "is_image": is_image,
            "extracted_text": extracted_text,
            "image_b64": b64_data,
            "image_mime": file.content_type
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"File upload processing failed: {str(e)}")

@app.post("/api/chat")
def chat(request: ChatRequest, x_gemini_key: Optional[str] = Header(None)):
    """Route user prompts to the correct LLM core."""
    
    # 1. Custom local LLM trained from scratch
    if request.model_type == "custom":
        # Formulate full text prompt
        full_prompt = ""
        if request.context_text:
            full_prompt += f"Context: {request.context_text}\n"
        if request.use_search:
            full_prompt += get_search_context(request.message) + "\n"
            
        # Add conversation history
        for msg in request.history[-3:]:  # Limit history to prevent window overflow
            full_prompt += f"{'User' if msg.role == 'user' else 'AI'}: {msg.content}\n"
            
        full_prompt += f"User: {request.message}\nAI: "
        
        reply = run_custom_model_inference(full_prompt)
        return {"response": reply}
        
    # 2. Local Ollama model
    elif request.model_type == "ollama":
        reply = call_ollama_api(request)
        return {"response": reply}
        
    # 3. Gemini cloud API
    elif request.model_type == "gemini":
        # Check API key priority: Custom request header, then environment variable
        api_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "response": "API Key Missing: Please enter your Google Gemini API Key in the UI Settings panel or set GEMINI_API_KEY environment variable on the server."
            }
        reply = call_gemini_api(request, api_key)
        return {"response": reply}
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type '{request.model_type}'")

# Training Control Routes
@app.get("/api/train/status")
def get_train_status():
    return train_manager.get_status()

@app.post("/api/train/start")
def start_train(
    batch_size: int = Form(16), 
    epochs: int = Form(10), 
    block_size: int = Form(128),
    n_layer: int = Form(4),
    n_head: int = Form(4),
    n_embd: int = Form(256),
    lr: float = Form(6e-4)
):
    result = train_manager.start_training(batch_size, epochs, block_size, n_layer, n_head, n_embd, lr)
    return result

@app.post("/api/train/stop")
def stop_train():
    return train_manager.stop_training()

@app.post("/api/train/kill")
def kill_train():
    return train_manager.kill_training()

@app.get("/api/train/history")
def get_train_history():
    return train_manager.get_log_history()

@app.get("/api/download/colab")
def download_colab():
    from fastapi.responses import FileResponse
    notebook_path = "c:/Rarey Temp/Ai/With Ai/custom_llm/colab_training.ipynb"
    if os.path.exists(notebook_path):
        return FileResponse(notebook_path, media_type="application/octet-stream", filename="colab_training.ipynb")
    raise HTTPException(status_code=404, detail="Notebook file not found.")

class SyncRequest(BaseModel):
    share_link: str

@app.post("/api/model/sync")
def sync_model_weights(request: SyncRequest):
    """Force download the latest checkpoint from Google Drive to the backend server."""
    global custom_model, custom_tokenizer
    dest_path = "c:/Rarey Temp/Ai/With Ai/custom_llm/checkpoint_cloud.pt"
    
    success = download_file_from_google_drive(request.share_link, dest_path)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to download file. Verify your Google Drive share link is valid and set to 'Anyone with link can view'.")
        
    # Reset model caches to force reloading the new checkpoint on the next request
    custom_model = None
    custom_tokenizer = None
    
    try:
        import torch
        checkpoint = torch.load(dest_path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0.0)
        return {
            "success": True, 
            "message": f"Successfully synced weights! Loaded model at Epoch {epoch+1}, Step {step} with Loss {loss:.4f}."
        }
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Downloaded file is corrupted: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bind to environment variable PORT for cloud compatibility, default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
