import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login

# Set page config for better display
st.set_page_config(page_title="LLaMA Chatbot", page_icon="🦙")
status_placeholder = st.empty()

# Check GPU
if torch.cuda.is_available():
    st.sidebar.success("✅ CUDA is available")
    st.sidebar.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ CUDA is not available. Using CPU.")

# Authentication
try:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Token not found")
    login(token=hf_token)
    status_placeholder.success("🔑 Successfully logged in to Hugging Face!")
except Exception as e:
    status_placeholder.error(f"🚫 Error with HF token: {str(e)}")
    st.stop()

st.title("🦙 LLaMA Chatbot")

# Model loading with detailed status updates
@st.cache_resource
def load_model():
    try:
        model_path = "Alaaeldin/Llama-demo"
        
        with st.spinner("🔄 Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token,
                trust_remote_code=True
            )
            st.success("✅ Tokenizer loaded!")
        
        with st.spinner("🔄 Loading model... This might take a few minutes..."):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )
            st.success("✅ Model loaded!")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model
model, tokenizer = load_model()

# Chat interface
if model and tokenizer:
    st.success("✨ Ready to chat! Enter your message below.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Prepare input
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.error("⚠️ Model loading failed. Please check the error messages above.")
