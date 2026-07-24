// Gemini Clone Frontend Application Logic

// API Configuration
let API_URL = localStorage.getItem("backend_api_url") || "https://jarvis-backend-0cvr.onrender.com";

// State Variables
let currentTab = "chat";
let uploadedFiles = []; // Store data of parsed files: {name, text, b64, mime}
let chatHistory = [];   // Store conversational messages: {role, content}
let isTrainingPollingActive = false;
let trainingPollInterval = null;

// Voice States
let voiceActive = false;
let recognition = null;
let speechUtterance = null;
let voicesList = [];

// Modern Toast Notification Utility
function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.style.position = "fixed";
    toast.style.bottom = "24px";
    toast.style.right = "24px";
    toast.style.background = type === "success" 
        ? "linear-gradient(135deg, #10b981 0%, #059669 100%)" 
        : "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)";
    toast.style.color = "white";
    toast.style.padding = "14px 28px";
    toast.style.borderRadius = "10px";
    toast.style.boxShadow = "0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(255,255,255,0.05)";
    toast.style.zIndex = "9999";
    toast.style.fontSize = "0.95rem";
    toast.style.fontFamily = "system-ui, sans-serif";
    toast.style.fontWeight = "600";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(30px)";
    toast.style.transition = "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)";
    toast.style.backdropFilter = "blur(8px)";
    toast.style.border = "1px solid rgba(255,255,255,0.1)";
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Force reflow
    toast.offsetHeight;
    
    // Animate in
    toast.style.opacity = "1";
    toast.style.transform = "translateY(0)";
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(30px)";
        setTimeout(() => {
            toast.remove();
        }, 400);
    }, 3200);
}

// Initialize Page
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSettings();
    initChat();
    initFileUpload();
    initVoice();
    initTraining();
    checkNetworkStatus();
    
    // Periodically check connection status
    setInterval(checkNetworkStatus, 5000);
});

// Network status indicator
async function checkNetworkStatus() {
    const dot = document.getElementById("net-status-dot");
    const text = document.getElementById("net-status-text");
    try {
        const res = await fetch(`${API_URL}/`);
        const data = await res.json();
        if (res.ok) {
            dot.className = "status-indicator online";
            text.textContent = data.internet_connected ? "Online (Web Search Active)" : "Offline Mode (Local Only)";
        }
    } catch (e) {
        dot.className = "status-indicator offline";
        text.textContent = "Server Disconnected";
    }
}

// 1. Navigation Tabs Control
function initTabs() {
    const navItems = document.querySelectorAll(".nav-item");
    const panels = document.querySelectorAll(".tab-panel");
    
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            
            // Toggle sidebar active item
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");
            
            // Toggle panels visible states
            panels.forEach(panel => panel.classList.remove("active"));
            const targetPanel = document.getElementById(`panel-${targetTab}`);
            if (targetPanel) {
                targetPanel.classList.add("active");
            }
            
            currentTab = targetTab;
            
            // Toggle training logs polling depending on tab
            if (currentTab === "training") {
                startTrainingLogsPolling();
            } else {
                stopTrainingLogsPolling();
            }
            
            // Stop voice mode if navigating away
            if (currentTab !== "voice" && voiceActive) {
                toggleVoiceMode(false);
            }
        });
    });
}

// 2. Settings Management
function initSettings() {
    const modelSelect = document.getElementById("model-routing");
    const geminiGroup = document.getElementById("settings-group-key");
    const geminiKeyInput = document.getElementById("gemini-key-input");
    const gdriveLinkInput = document.getElementById("gdrive-link-input");
    const gdriveLogLinkInput = document.getElementById("gdrive-log-link-input");
    const backendUrlInput = document.getElementById("backend-url-input");
    const saveBtn = document.getElementById("save-settings-btn");
    const syncBtn = document.getElementById("sync-gdrive-btn");
    const syncStatus = document.getElementById("sync-status-msg");
    
    // Load local storage values
    const savedModel = localStorage.getItem("model_routing") || "gemini";
    const savedGeminiKey = localStorage.getItem("gemini_api_key") || "";
    const savedGDriveLink = localStorage.getItem("gdrive_share_link") || "";
    const savedGDriveLogLink = localStorage.getItem("gdrive_log_share_link") || "";
    const savedBackendUrl = localStorage.getItem("backend_api_url") || "https://jarvis-backend-0cvr.onrender.com";
    
    modelSelect.value = savedModel;
    geminiKeyInput.value = savedGeminiKey;
    gdriveLinkInput.value = savedGDriveLink;
    if (gdriveLogLinkInput) {
        gdriveLogLinkInput.value = savedGDriveLogLink;
    }
    if (backendUrlInput) {
        backendUrlInput.value = savedBackendUrl;
    }
    
    updateSettingsVisibility(savedModel);
    updateModelTag(savedModel);

    modelSelect.addEventListener("change", (e) => {
        updateSettingsVisibility(e.target.value);
    });
    
    saveBtn.addEventListener("click", () => {
        const newUrl = backendUrlInput ? backendUrlInput.value.trim() : "https://jarvis-backend-0cvr.onrender.com";
        
        localStorage.setItem("model_routing", modelSelect.value);
        localStorage.setItem("gemini_api_key", geminiKeyInput.value);
        localStorage.setItem("gdrive_share_link", gdriveLinkInput.value);
        if (gdriveLogLinkInput) {
            localStorage.setItem("gdrive_log_share_link", gdriveLogLinkInput.value);
        }
        localStorage.setItem("backend_api_url", newUrl);
        
        // Dynamically update active API URL
        API_URL = newUrl;
        
        updateModelTag(modelSelect.value);
        showToast("Configurations saved successfully!");
    });

    syncBtn.addEventListener("click", async () => {
        const link = gdriveLinkInput.value.trim();
        const logLink = gdriveLogLinkInput ? gdriveLogLinkInput.value.trim() : "";
        if (!link) {
            syncStatus.textContent = "Please enter a valid Google Drive sharing link first!";
            syncStatus.style.color = "#ef4444";
            return;
        }
        
        syncStatus.textContent = "Initiating weight synchronization...";
        syncStatus.style.color = "#3b82f6";
        syncBtn.disabled = true;
        syncBtn.style.opacity = "0.7";
        
        try {
            const resp = await fetch(`${API_URL}/api/model/sync`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ 
                    share_link: link,
                    log_share_link: logLink || null
                })
            });
            
            const data = await resp.json();
            if (resp.ok) {
                syncStatus.textContent = "Sync task accepted. Downloading in background...";
                showToast("Sync task started in the background!");
                
                // Poll status endpoint until finished
                const pollInterval = setInterval(async () => {
                    try {
                        const statusResp = await fetch(`${API_URL}/api/model/sync/status`);
                        if (statusResp.ok) {
                            const statusData = await statusResp.json();
                            syncStatus.textContent = statusData.message || "Syncing...";
                            
                            if (statusData.status === "success") {
                                clearInterval(pollInterval);
                                syncStatus.style.color = "#10b981";
                                syncBtn.disabled = false;
                                syncBtn.style.opacity = "1";
                                showToast("Weights synced successfully!");
                                
                                // Redraw loss chart
                                try {
                                    const resHistory = await fetch(`${API_URL}/api/train/history`);
                                    const history = await resHistory.json();
                                    if (history && history.length > 0) {
                                        plotLossChart(history);
                                        showToast("Loss curves updated successfully!");
                                    }
                                } catch (historyErr) {
                                    console.error("Error loading loss history:", historyErr);
                                }
                            } else if (statusData.status === "failed") {
                                clearInterval(pollInterval);
                                syncStatus.style.color = "#ef4444";
                                syncBtn.disabled = false;
                                syncBtn.style.opacity = "1";
                                showToast("Weight sync failed", "error");
                            }
                        }
                    } catch (pollErr) {
                        console.error("Error polling sync status:", pollErr);
                    }
                }, 3000);
            } else {
                syncStatus.textContent = `Error: ${data.detail || "Failed to start sync"}`;
                syncStatus.style.color = "#ef4444";
                showToast("Failed to start sync", "error");
                syncBtn.disabled = false;
                syncBtn.style.opacity = "1";
            }
        } catch (err) {
            console.error(err);
            syncStatus.textContent = "Network error: Connection to backend failed.";
            syncStatus.style.color = "#ef4444";
            showToast("Connection to server failed", "error");
            syncBtn.disabled = false;
            syncBtn.style.opacity = "1";
        }
    });
    
    // Jarvis Node Runner Toggles
    const copyBtn = document.getElementById("copy-run-btn");
    if (copyBtn) {
        copyBtn.addEventListener("click", () => {
            const cmd = document.getElementById("local-run-command").innerText;
            navigator.clipboard.writeText(cmd);
            showToast("Command copied to clipboard!");
        });
    }

    const toggleNodeBtn = document.getElementById("toggle-backend-node-btn");
    if (toggleNodeBtn && backendUrlInput) {
        const updateNodeBtnText = (url) => {
            if (url.includes("127.0.0.1") || url.includes("localhost")) {
                toggleNodeBtn.textContent = "Switch to Cloud Render (Production)";
                toggleNodeBtn.style.background = "linear-gradient(135deg, #10b981 0%, #059669 100%)";
            } else {
                toggleNodeBtn.textContent = "Switch to Localhost Node (127.0.0.1)";
                toggleNodeBtn.style.background = "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)";
            }
        };
        
        // Initial setup
        updateNodeBtnText(backendUrlInput.value);
        
        toggleNodeBtn.addEventListener("click", () => {
            const currentUrl = backendUrlInput.value.trim();
            let newUrl = "https://jarvis-backend-0cvr.onrender.com";
            if (!currentUrl.includes("127.0.0.1") && !currentUrl.includes("localhost")) {
                newUrl = "http://127.0.0.1:8000";
            }
            
            backendUrlInput.value = newUrl;
            localStorage.setItem("backend_api_url", newUrl);
            API_URL = newUrl;
            updateNodeBtnText(newUrl);
            checkNetworkStatus();
            showToast(`API switched to ${newUrl}`);
        });
    }
    
    function updateSettingsVisibility(val) {
        if (geminiGroup) {
            geminiGroup.style.display = val === "gemini" ? "flex" : "none";
        }
    }
    
    function updateModelTag(val) {
        const tag = document.getElementById("current-model-tag");
        if (!tag) return;
        if (val === "gemini") {
            tag.textContent = "Using Gemini 1.5 Flash (Google Cloud)";
        } else {
            tag.textContent = "Using Custom Model (Trained from Scratch)";
        }
    }
}

// 3. File Upload & Processing
function initFileUpload() {
    const uploadInput = document.getElementById("file-upload");
    const previewStrip = document.getElementById("file-previews");
    
    uploadInput.addEventListener("change", async (e) => {
        const files = e.target.files;
        if (!files.length) return;
        
        for (const file of files) {
            const formData = new FormData();
            formData.append("file", file);
            
            // Create loading preview chip
            const chipId = "chip-" + Date.now();
            addPreviewChip(chipId, file.name, "Uploading...");
            
            try {
                const res = await fetch(`${API_URL}/api/upload`, {
                    method: "POST",
                    body: formData
                });
                
                if (!res.ok) throw new Error("Upload failed");
                const data = await res.json();
                
                if (data.success) {
                    // Update preview chip to success state
                    updatePreviewChip(chipId, file.name, data);
                } else {
                    removePreviewChip(chipId);
                    alert(`Failed to parse file: ${data.message}`);
                }
            } catch (err) {
                console.error("Upload error:", err);
                removePreviewChip(chipId);
                alert(`Error uploading file '${file.name}': Make sure backend server is running.`);
            }
        }
        uploadInput.value = ""; // Clear file selector
    });

    function addPreviewChip(id, filename, text) {
        previewStrip.style.display = "flex";
        const chip = document.createElement("div");
        chip.className = "preview-chip";
        chip.id = id;
        
        // Icon based on type
        let iconSvg = `<svg style="width:14px;height:14px" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`;
        
        chip.innerHTML = `
            ${iconSvg}
            <span class="preview-chip-name">${filename} (${text})</span>
            <span class="preview-chip-remove" onclick="removeUploadedFile('${id}')">&times;</span>
        `;
        previewStrip.appendChild(chip);
    }

    function updatePreviewChip(id, filename, fileData) {
        const chip = document.getElementById(id);
        if (chip) {
            chip.querySelector(".preview-chip-name").textContent = filename;
            
            // Save file data to states
            uploadedFiles.push({
                id: id,
                name: filename,
                extracted_text: fileData.extracted_text,
                image_b64: fileData.image_b64,
                image_mime: fileData.image_mime,
                is_image: fileData.is_image
            });
        }
    }
}

function removeUploadedFile(id) {
    uploadedFiles = uploadedFiles.filter(f => f.id !== id);
    const chip = document.getElementById(id);
    if (chip) chip.remove();
    
    const previewStrip = document.getElementById("file-previews");
    if (uploadedFiles.length === 0) {
        previewStrip.style.display = "none";
    }
}

// 4. Chat Engine Implementation
function initChat() {
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const messagesBox = document.getElementById("chat-messages");
    
    // Textarea auto-resize
    chatInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitChatMessage();
        }
    });

    sendBtn.addEventListener("click", submitChatMessage);

    async function submitChatMessage() {
        const text = chatInput.value.trim();
        const hasFiles = uploadedFiles.length > 0;
        
        if (!text && !hasFiles) return;
        
        // Disable controls
        chatInput.value = "";
        chatInput.style.height = "auto";
        
        // Append user bubble
        appendMessageBubble("user", text, uploadedFiles);
        
        // Collect prompt parameters
        const activeModel = localStorage.getItem("model_routing") || "gemini";
        const geminiKey = localStorage.getItem("gemini_api_key") || "";
        const ollamaName = localStorage.getItem("ollama_model_name") || "phi3";
        const useSearch = document.getElementById("search-toggle").checked;
        
        // Compile upload contents
        let contextText = "";
        let imageB64 = null;
        let imageMime = null;
        
        uploadedFiles.forEach(file => {
            if (file.is_image) {
                imageB64 = file.image_b64;
                imageMime = file.image_mime;
            } else {
                contextText += `\n[Content of File: ${file.name}]\n${file.extracted_text}\n`;
            }
        });
        
        // Add to history
        chatHistory.push({ role: "user", content: text });
        
        // Clear previews panel
        uploadedFiles = [];
        document.getElementById("file-previews").style.display = "none";
        document.getElementById("file-previews").innerHTML = "";
        
        // Create bot thinking bubble
        const botBubbleId = "bot-msg-" + Date.now();
        createThinkingBubble(botBubbleId);
        
        try {
            const res = await fetch(`${API_URL}/api/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Gemini-Key": geminiKey
                },
                body: jsonPayloadBuilder()
            });
            
            if (!res.ok) throw new Error("Server responded with error");
            const data = await res.json();
            
            // Remove thinking state and populate response
            removeThinkingBubble(botBubbleId);
            appendMessageBubble("bot", data.response);
            chatHistory.push({ role: "assistant", content: data.response });
            
        } catch (err) {
            console.error("Chat error:", err);
            removeThinkingBubble(botBubbleId);
            appendMessageBubble("bot", "System Error: Failed to receive response from backend. Verify FastAPI server is running on port 8000.");
        }
        
        function jsonPayloadBuilder() {
            // Limits conversational context to past 10 logs for token safety
            const slicedHistory = chatHistory.slice(-10);
            // SlicedHistory has user prompt at end, backend expects history *before* prompt, so pop last
            slicedHistory.pop(); 
            
            return JSON.stringify({
                message: text,
                history: slicedHistory,
                model_type: activeModel,
                use_search: useSearch,
                context_text: contextText,
                image_b64: imageB64,
                image_mime: imageMime,
                ollama_model_name: ollamaName
            });
        }
    }

    function appendMessageBubble(role, text, fileChips = []) {
        const bubble = document.createElement("div");
        bubble.className = `message ${role === "user" ? "user-msg" : "system-msg"}`;
        
        // Avatar element
        let avatarMarkup = "";
        if (role === "user") {
            avatarMarkup = `
                <div class="message-avatar">
                    <svg style="width:18px;height:18px" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                </div>
            `;
        } else {
            avatarMarkup = `
                <div class="message-avatar">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px;">
                        <circle cx="12" cy="12" r="9.5" stroke="url(#sparkGrad)" stroke-width="1.6" />
                        <circle cx="12" cy="12" r="2.2" fill="url(#sparkGrad)" />
                        <g fill="url(#sparkGrad)">
                            <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" />
                            <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" transform="rotate(120 12 12)" />
                            <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" transform="rotate(240 12 12)" />
                        </g>
                    </svg>
                </div>
            `;
        }

        // Attachments display
        let chipsMarkup = "";
        if (fileChips.length > 0) {
            chipsMarkup += `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">`;
            fileChips.forEach(chip => {
                chipsMarkup += `
                    <span style="font-size:11px;background:rgba(255,255,255,0.08);padding:4px 8px;border-radius:6px;display:inline-flex;align-items:center;gap:6px;">
                        <svg style="width:10px;height:10px" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
                        ${chip.name}
                    </span>`;
            });
            chipsMarkup += `</div>`;
        }

        // Format code snippets inside bot messages
        let formattedText = text;
        if (role === "bot") {
            formattedText = formatBotMarkdown(text);
        } else {
            // Prevent HTML injection for user text
            formattedText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
        }

        bubble.innerHTML = `
            ${avatarMarkup}
            <div class="message-bubble">
                ${chipsMarkup}
                <div>${formattedText}</div>
            </div>
        `;
        
        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function createThinkingBubble(id) {
        const bubble = document.createElement("div");
        bubble.className = "message system-msg thinking";
        bubble.id = id;
        bubble.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px;animation:float-slow 2s infinite ease-in-out;">
                    <circle cx="12" cy="12" r="9.5" stroke="url(#sparkGrad)" stroke-width="1.6" />
                    <circle cx="12" cy="12" r="2.2" fill="url(#sparkGrad)" />
                    <g fill="url(#sparkGrad)">
                        <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" />
                        <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" transform="rotate(120 12 12)" />
                        <path d="M 12 9.8 C 14.5 9.8, 17.5 11, 18.5 12.8 C 15 13.2, 13.2 12.8, 12 12 C 12 11.2, 12 10.5, 12 9.8 Z" transform="rotate(240 12 12)" />
                    </g>
                </svg>
            </div>
            <div class="message-bubble typing-bubble">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function removeThinkingBubble(id) {
        const bubble = document.getElementById(id);
        if (bubble) bubble.remove();
    }

    function formatBotMarkdown(text) {
        let html = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        
        // Code Blocks ```lang code ```
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
        });
        
        // Inline code `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold text **bold**
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Linebreaks
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }
}

// 5. 'Gemini Live' Voice Interface
function initVoice() {
    const voiceCore = document.getElementById("voice-core-btn");
    const voiceContainer = document.querySelector(".voice-container");
    const voiceLangSelect = document.getElementById("voice-lang-select");
    const voiceMuteBtn = document.getElementById("voice-mute-btn");
    const userSubtitle = document.getElementById("sub-user-text");
    const botSubtitle = document.getElementById("sub-bot-text");
    const voiceStatusTitle = document.getElementById("voice-status-title");
    const voiceStatusDesc = document.getElementById("voice-status-desc");

    // Check browser speech recognition availability
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceStatusDesc.textContent = "Web Speech API is not supported in this browser. Please use Chrome for full voice capabilities.";
        voiceCore.style.opacity = "0.5";
        voiceCore.style.pointerEvents = "none";
        return;
    }

    // Initialize Speech Synthesizer voices list
    window.speechSynthesis.onvoiceschanged = () => {
        voicesList = window.speechSynthesis.getVoices();
    };

    voiceCore.addEventListener("click", () => {
        toggleVoiceMode(!voiceActive);
    });

    voiceMuteBtn.addEventListener("click", () => {
        if (!voiceActive) return;
        
        if (voiceMuteBtn.classList.contains("active")) {
            voiceMuteBtn.classList.remove("active");
            voiceMuteBtn.classList.add("muted");
            recognition.stop(); // Stop listening
            voiceStatusTitle.textContent = "Muted";
        } else {
            voiceMuteBtn.classList.remove("muted");
            voiceMuteBtn.classList.add("active");
            recognition.start(); // Start listening
            voiceStatusTitle.textContent = "Listening";
        }
    });

    function toggleVoiceMode(activate) {
        voiceActive = activate;
        
        if (activate) {
            voiceContainer.className = "voice-container active listening";
            voiceStatusTitle.textContent = "Listening";
            voiceStatusDesc.textContent = "Start speaking, I will respond and listen continuously.";
            userSubtitle.textContent = "Listening...";
            botSubtitle.textContent = "";
            
            // Setup Speech recognition
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true; // Essential for low latency speech start detection (interruption)
            recognition.lang = voiceLangSelect.value;
            
            // Audio start detection -> TRIGGER IMMEDIATE INTERRUPTION
            recognition.onaudiostart = () => {
                handleInterruption();
            };

            recognition.onspeechstart = () => {
                handleInterruption();
            };

            let finalTranscript = "";
            
            recognition.onresult = (event) => {
                // If the bot is currently speaking, user speech activity cancels it immediately
                handleInterruption();
                
                let interimTranscript = "";
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                // Show real-time voice feedback
                userSubtitle.textContent = finalTranscript || interimTranscript;
                
                if (finalTranscript.trim() !== "") {
                    // Send final parsed speech text to model
                    const spokenText = finalTranscript.trim();
                    finalTranscript = ""; // Reset
                    
                    // Stop recognition temporarily while query resolves, to prevent self-looping
                    recognition.stop();
                    voiceContainer.className = "voice-container active";
                    voiceStatusTitle.textContent = "Thinking...";
                    
                    sendVoicePrompt(spokenText);
                }
            };

            recognition.onend = () => {
                // Restart recognition if we are still in active live mode and not speaking
                if (voiceActive && !window.speechSynthesis.speaking && voiceMuteBtn.classList.contains("active")) {
                    try {
                        recognition.start();
                        voiceContainer.className = "voice-container active listening";
                        voiceStatusTitle.textContent = "Listening";
                    } catch (e) {}
                }
            };
            
            recognition.start();
            
        } else {
            // Shutdown Live voice Mode
            voiceContainer.className = "voice-container";
            voiceStatusTitle.textContent = "Offline";
            voiceStatusDesc.textContent = "Click the core button to activate Live Conversation";
            userSubtitle.textContent = "Speak to start...";
            botSubtitle.textContent = "";
            
            if (recognition) {
                recognition.stop();
                recognition = null;
            }
            window.speechSynthesis.cancel();
        }
    }

    function handleInterruption() {
        // Core Interruption Algorithm: If model is currently talking and user speaks, cancel speaking immediately
        if (window.speechSynthesis.speaking) {
            console.log("[INTERRUPTION] User input detected during TTS output. Cancelling synthesis...");
            window.speechSynthesis.cancel();
            
            voiceContainer.className = "voice-container active listening";
            voiceStatusTitle.textContent = "Listening";
            botSubtitle.textContent = "[Interrupted]";
        }
    }

    async function sendVoicePrompt(text) {
        const activeModel = localStorage.getItem("model_routing") || "gemini";
        const geminiKey = localStorage.getItem("gemini_api_key") || "";
        const ollamaName = localStorage.getItem("ollama_model_name") || "phi3";
        
        chatHistory.push({ role: "user", content: text });
        
        try {
            const res = await fetch(`${API_URL}/api/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Gemini-Key": geminiKey
                },
                body: JSON.stringify({
                    message: text,
                    history: chatHistory.slice(-6),
                    model_type: activeModel,
                    use_search: false, // Voice is fast, skip search for lower latency
                    ollama_model_name: ollamaName
                })
            });
            
            if (!res.ok) throw new Error();
            const data = await res.json();
            
            chatHistory.push({ role: "assistant", content: data.response });
            botSubtitle.textContent = data.response;
            
            // Speak response text out loud
            speakUtterance(data.response);
            
        } catch (err) {
            botSubtitle.textContent = "Error receiving speech response.";
            // Restart listening
            if (voiceActive && voiceMuteBtn.classList.contains("active")) {
                recognition.start();
            }
        }
    }

    function speakUtterance(text) {
        // Clean speech text of markdown symbols
        const cleanText = text.replace(/[*#`_\-\n]/g, " ").trim();
        
        speechUtterance = new SpeechSynthesisUtterance(cleanText);
        speechUtterance.lang = voiceLangSelect.value;
        
        // Find best voice match for language
        let voices = window.speechSynthesis.getVoices();
        let targetVoice = null;
        
        // Try finding standard voices based on selected lang code
        const langCode = voiceLangSelect.value.split("-")[0]; // "en", "hi", "te"
        for (let v of voices) {
            if (v.lang.startsWith(langCode)) {
                targetVoice = v;
                break;
            }
        }
        if (targetVoice) speechUtterance.voice = targetVoice;
        
        speechUtterance.onstart = () => {
            voiceContainer.className = "voice-container active speaking";
            voiceStatusTitle.textContent = "Speaking";
        };
        
        speechUtterance.onend = () => {
            if (voiceActive) {
                voiceContainer.className = "voice-container active listening";
                voiceStatusTitle.textContent = "Listening";
                userSubtitle.textContent = "Listening...";
                
                // Restart mic recognition loop
                if (voiceMuteBtn.classList.contains("active")) {
                    try {
                        recognition.start();
                    } catch(e) {}
                }
            }
        };
        
        speechUtterance.onerror = () => {
            if (voiceActive && voiceMuteBtn.classList.contains("active")) {
                try {
                    recognition.start();
                } catch(e) {}
            }
        };

        window.speechSynthesis.speak(speechUtterance);
    }
}

// 6. Training Control Center & SVG Plotter
function initTraining() {
    const startBtn = document.getElementById("btn-start-train");
    const stopBtn = document.getElementById("btn-stop-train");
    const badge = document.getElementById("train-status-badge");
    
    startBtn.addEventListener("click", async () => {
        // Grab form fields
        const epochs = document.getElementById("train-epochs").value;
        const batch = document.getElementById("train-batch").value;
        const block = document.getElementById("train-block").value;
        const lr = document.getElementById("train-lr").value;
        const layers = document.getElementById("train-layers").value;
        const heads = document.getElementById("train-heads").value;
        const embd = document.getElementById("train-embd").value;
        
        const formData = new FormData();
        formData.append("epochs", epochs);
        formData.append("batch_size", batch);
        formData.append("block_size", block);
        formData.append("lr", lr);
        formData.append("n_layer", layers);
        formData.append("n_head", heads);
        formData.append("n_embd", embd);
        
        try {
            const res = await fetch(`${API_URL}/api/train/start`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (data.success) {
                alert("Model training initiated in background!");
                startBtn.disabled = true;
                stopBtn.disabled = false;
                badge.className = "badge badge-training";
                badge.textContent = "Training";
            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (e) {
            alert("Failed to connect to backend server. Make sure API is running.");
        }
    });

    stopBtn.addEventListener("click", async () => {
        try {
            const res = await fetch(`${API_URL}/api/train/stop`, { method: "POST" });
            const data = await res.json();
            
            if (data.success) {
                alert(data.message);
                stopBtn.disabled = true;
                // Button state will reset fully via poll
            } else {
                alert(data.message);
            }
        } catch (e) {}
    });
}

function startTrainingLogsPolling() {
    isTrainingPollingActive = true;
    pollTrainingStatus();
    trainingPollInterval = setInterval(pollTrainingStatus, 2000);
}

function stopTrainingLogsPolling() {
    isTrainingPollingActive = false;
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
    }
}

async function pollTrainingStatus() {
    if (!isTrainingPollingActive) return;
    
    try {
        const resStatus = await fetch(`${API_URL}/api/train/status`);
        const status = await resStatus.json();
        
        // Update training status badges and button states
        const startBtn = document.getElementById("btn-start-train");
        const stopBtn = document.getElementById("btn-stop-train");
        const badge = document.getElementById("train-status-badge");
        
        if (status.status === "training") {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            badge.className = "badge badge-training";
            badge.textContent = "Training";
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            badge.className = "badge badge-paused";
            badge.textContent = "Paused";
        }
        
        // Update metric values in monitor board
        if (status.metrics) {
            document.getElementById("metric-step").textContent = status.metrics.step || "-";
            document.getElementById("metric-loss").textContent = (status.metrics.loss !== undefined) ? status.metrics.loss.toFixed(4) : "-";
            document.getElementById("metric-vram").textContent = status.metrics.vram_gb ? `${status.metrics.vram_gb.toFixed(2)} GB` : "CPU";
            document.getElementById("metric-speed").textContent = status.metrics.speed ? `${status.metrics.speed.toFixed(0)} tok/s` : "-";
        }
        
        // Update Checkpoint banner
        const chkStatus = document.getElementById("chk-status");
        const chkInfo = document.getElementById("chk-info");
        if (status.has_checkpoint) {
            chkStatus.textContent = `Active Checkpoint Loaded (${status.checkpoint_size_mb} MB)`;
            chkInfo.textContent = `Weights will resume automatically from Epoch ${status.metrics ? status.metrics.epoch + 1 : 1} upon restart.`;
        } else {
            chkStatus.textContent = "No active checkpoints found";
            chkInfo.textContent = "Run training to generate weights locally.";
        }
        
        // Fetch complete history and plot graph SVG
        const resHistory = await fetch(`${API_URL}/api/train/history`);
        const history = await resHistory.json();
        if (history && history.length > 0) {
            plotLossChart(history);
        }
        
    } catch (e) {
        console.error("Error polling training status:", e);
    }
}

function plotLossChart(history) {
    const svgPath = document.getElementById("loss-chart-path");
    if (!svgPath) return;
    
    // Viewbox configurations of our SVG
    const width = 500;
    const height = 220;
    
    const paddingLeft = 40;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 30;
    
    const chartWidth = width - paddingLeft - paddingRight;
    const chartHeight = height - paddingTop - paddingBottom;
    const zeroY = height - paddingBottom; // Bottom-most line (Loss = 0.0)
    
    // Downsample history if it contains too many steps (limits coordinates inside DOM)
    const maxDataPoints = 80;
    let data = history;
    if (history.length > maxDataPoints) {
        const step = Math.ceil(history.length / maxDataPoints);
        data = history.filter((_, idx) => idx % step === 0);
        // Ensure last item is always in data
        if (data[data.length-1] !== history[history.length-1]) {
            data.push(history[history.length-1]);
        }
    }
    
    // Map bounds
    const maxLossLimit = 8.0; // Fixed y range matching SVG label definitions (0 to 8)
    
    let pathPoints = [];
    const stepCount = data.length;
    
    data.forEach((entry, idx) => {
        // Calculate X coordinate
        const percentX = stepCount > 1 ? idx / (stepCount - 1) : 0;
        const x = paddingLeft + (percentX * chartWidth);
        
        // Calculate Y coordinate
        // Bound loss inside maxLossLimit
        const lossVal = Math.min(Math.max(entry.loss, 0), maxLossLimit);
        const percentY = lossVal / maxLossLimit;
        const y = zeroY - (percentY * chartHeight);
        
        pathPoints.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    });
    
    // Write points to SVG polyline path d="M x,y L x,y ..."
    if (pathPoints.length > 0) {
        const dAttribute = "M " + pathPoints.join(" L ");
        svgPath.setAttribute("d", dAttribute);
    }
}
