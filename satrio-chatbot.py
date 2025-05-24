import streamlit as st
import requests
import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configurasi API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("⚠️ OPENROUTER_API_KEY tidak ditemukan! Pastikan file .env sudah dibuat dengan benar.")
    st.stop()

HEADERS = {
  "Authorization": f"Bearer {OPENROUTER_API_KEY}",
  "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost:8501"),
  "X-Title": os.getenv("X_TITLE", "AI Chatbot Streamlit")
}
API_URL = os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")

# Inisialisasi selected_model agar tidak KeyError saat awal load
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "deepseek/deepseek-chat-v3-0324"

# UI
st.title("🧠 AI Chatbot Bubble Style 🧠")
st.markdown(f"Powered by {st.session_state['selected_model']} via OpenRouter 🤖")

# Fungsi untuk menyimpan chat_folders ke file JSON
def save_chats_to_json():
    with open("chat_folders.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state["chat_folders"], f, ensure_ascii=False, indent=2)

def load_chats_from_json():
    if os.path.exists("chat_folders.json"):
        with open("chat_folders.json", "r", encoding="utf-8") as f:
            st.session_state["chat_folders"] = json.load(f)
            if "active_folder" not in st.session_state:
                st.session_state["active_folder"] = list(st.session_state["chat_folders"].keys())[0]

# Inisiasi Riwayat Chat dan load dari JSON jika ada
if "chat_folders" not in st.session_state:
    st.session_state["chat_folders"] = {}
    st.session_state["active_folder"] = None

# Sidebar untuk folder chat
with st.sidebar:
    st.header("Welcome to My Chatbot!")
    # Pilihan model AI
    model_options = {
        "DeepSeek Chat v3": "deepseek/deepseek-chat-v3-0324",
        "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
        "Meta Llama 3 8B Instruct": "meta-llama/llama-3.3-8b-instruct"
    }
    selected_model_label = st.selectbox("Pilih Model AI", list(model_options.keys()), key="select_model")
    st.session_state["selected_model"] = model_options[selected_model_label]

    # Tampilkan daftar chat sebagai list, bisa dipilih untuk dibuka kembali
    if st.session_state["chat_folders"]:
        st.markdown("**Daftar Chat:**")
        # Pilihan chat dengan selectbox
        folder_names = list(st.session_state["chat_folders"].keys())
        selected_folder = st.selectbox("Pilih chat", folder_names, index=folder_names.index(st.session_state["active_folder"]) if st.session_state["active_folder"] in folder_names else 0, key="select_chat_folder")
        if selected_folder != st.session_state["active_folder"]:
            st.session_state["active_folder"] = selected_folder
            st.session_state["chat_history"] = st.session_state["chat_folders"][selected_folder]
            save_chats_to_json()
            st.rerun()
        st.markdown(f"**Chat Aktif:** `{st.session_state['active_folder']}`")
        idx = folder_names.index(st.session_state["active_folder"])
        # Ganti selectbox menjadi text_input untuk chat aktif
        for i, name in enumerate(folder_names):
            if i == idx:
                new_name = st.text_input("Rename Chat", value=name, key="rename_chat_inline")
                if new_name != name:
                    if new_name and new_name not in folder_names:
                        st.session_state["chat_folders"][new_name] = st.session_state["chat_folders"][name]
                        del st.session_state["chat_folders"][name]
                        st.session_state["active_folder"] = new_name
                        save_chats_to_json()
                        st.rerun()
                    elif new_name in folder_names:
                        st.warning("Nama chat sudah ada!")
    else:
        st.info("Belum ada chat. Mulai chat baru di bawah!")
    # Tombol New Chat di bawah subheader Daftar Chat
    if st.button("New Chat"):
        base_name = "Chat"
        i = 1
        auto_name = f"{base_name} {i}"
        while auto_name in st.session_state["chat_folders"]:
            i += 1
            auto_name = f"{base_name} {i}"
        st.session_state["chat_folders"][auto_name] = []
        st.session_state["active_folder"] = auto_name
        st.session_state["chat_history"] = st.session_state["chat_folders"][auto_name]
        save_chats_to_json()
        st.rerun()
    # Tombol Delete Chat dan Hapus Semua Chat di paling bawah sidebar
    st.markdown("---")
    col_del1, col_del2 = st.columns([1, 1])
    with col_del1:
        if st.button("🗑️", key="delete_chat_btn", help="Delete Chat"):
            if len(st.session_state["chat_folders"]) > 1:
                del st.session_state["chat_folders"][st.session_state["active_folder"]]
                st.session_state["active_folder"] = list(st.session_state["chat_folders"].keys())[0]
                save_chats_to_json()
                st.rerun()
            else:
                st.warning("Minimal harus ada satu chat!")
    with col_del2:
        if st.button("❌", key="delete_all_chats_btn", help="Hapus Semua Chat"):
            st.session_state["chat_folders"] = {}
            st.session_state["active_folder"] = None
            st.session_state["chat_history"] = []
            save_chats_to_json()
            st.rerun()
# Inisialisasi chat_history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sinkronisasi chat_history dengan folder aktif
if st.session_state["active_folder"]:
    st.session_state["chat_history"] = st.session_state["chat_folders"][st.session_state["active_folder"]]
else:
    st.session_state["chat_history"] = []

for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message(chat["role"]):
        if chat["role"] == "assistant":
            st.markdown(chat["content"])
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Copy", key=f"copy_btn_{idx}"):
                    st.markdown(f"""
                    <script>
                    navigator.clipboard.writeText({json.dumps(chat['content'])});
                    </script>
                    <span style='color:green'>Teks jawaban berhasil disalin!</span>
                    """, unsafe_allow_html=True)
            with col2:
                if st.button("Regenerate", key=f"regen_btn_{idx}"):
                    # Ambil user prompt sebelum jawaban ini
                    user_prompt = None
                    for i in range(idx-1, -1, -1):
                        if st.session_state.chat_history[i]["role"] == "user":
                            user_prompt = st.session_state.chat_history[i]["content"]
                            break
                    # Gunakan doc_text jika ada
                    doc_text = st.session_state.get('doc_text', None)
                    final_user_input = user_prompt
                    if user_prompt and doc_text:
                        final_user_input = f"Berikut adalah isi dokumen yang saya upload:\n{doc_text}\n\nPertanyaan saya: {user_prompt}"
                    elif not user_prompt and doc_text:
                        final_user_input = f"Jelaskan gambar/dokumen berikut:\n{doc_text}"
                    else:
                        final_user_input = user_prompt
                    # Kirim ulang ke API
                    with st.spinner("Mengetik ulang jawaban..."):
                        payload = {
                            "model": st.session_state["selected_model"],
                            "messages": [
                                {"role": "system", "content": "You are a friendly and helpfull assistant."},
                                {"role": "user", "content": final_user_input}
                            ]
                        }
                        response = requests.post(API_URL, headers=HEADERS, json=payload)
                    if response.status_code == 200:
                        bot_reply = response.json()['choices'][0]['message']['content']
                    else:
                        bot_reply = "⚠️ Maaf, gagal mengambil respons dari OpenRouter."
                    # Update jawaban bot pada index ini
                    st.session_state.chat_history[idx]["content"] = bot_reply
                    st.session_state.chat_history[idx]["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["chat_folders"][st.session_state["active_folder"]] = st.session_state["chat_history"]
                    save_chats_to_json()
                    st.rerun()
        else:
            st.markdown(chat["content"])
        if "timestamp" in chat:
            st.caption(f"⏰ {chat['timestamp']}")


# Input Pengguna
user_input = st.chat_input("Tulis pesan di sini...")

# Gabungkan pertanyaan user dengan isi dokumen jika ada dokumen yang diupload
final_user_input = user_input
if user_input and 'doc_text' in st.session_state and st.session_state['doc_text']:
    final_user_input = f"Berikut adalah isi dokumen yang saya upload:\n{st.session_state['doc_text']}\n\nPertanyaan saya: {user_input}"
elif not user_input and 'doc_text' in st.session_state and st.session_state['doc_text']:
    final_user_input = f"Jelaskan gambar/dokumen berikut:\n{st.session_state['doc_text']}"

# --- Fitur: Otomatis buat folder chat baru saat mulai chat, nama dari keyword ---
if user_input:
    # Jika belum ada folder chat sama sekali, atau tidak ada chat aktif, buat folder baru
    if not st.session_state["active_folder"]:
        words = user_input.split()
        if words:
            auto_name = " ".join(words[:3])
        else:
            auto_name = f"Chat 1"
        # Pastikan nama unik
        base_name = auto_name
        i = 2
        while auto_name in st.session_state["chat_folders"]:
            auto_name = f"{base_name} {i}"
            i += 1
        st.session_state["chat_folders"][auto_name] = []
        st.session_state["active_folder"] = auto_name
        st.session_state["chat_history"] = st.session_state["chat_folders"][auto_name]
        save_chats_to_json()
    # Setelah folder aktif ada, lanjutkan chat di folder yang sama
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.chat_message("user").markdown(user_input)
    st.caption(f"⏰ {timestamp}")
    st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": timestamp})
    # Gabungkan pertanyaan user dengan isi dokumen jika ada dokumen yang diupload
    final_user_input = user_input
    if 'doc_text' in st.session_state and st.session_state['doc_text']:
        final_user_input = f"Berikut adalah isi dokumen yang saya upload:\n{st.session_state['doc_text']}\n\nPertanyaan saya: {user_input}"
    # Kirim API ke OpenRouter
    with st.spinner("Mengetik..."):
        payload = {
            "model": st.session_state["selected_model"],
            "messages": [
                {"role": "system", "content": "You are a friendly and helpfull assistant."},
                {"role": "user", "content": final_user_input}
            ]
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload)
    # Proses Response
    if response.status_code == 200:
        bot_reply = response.json()['choices'][0]['message']['content']
    else:
        bot_reply = "⚠️ Maaf, gagal mengambil respons dari OpenRouter."
    # Tampilkan Respons Bot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.chat_message("assistant").markdown(bot_reply)
    st.caption(f"⏰ {timestamp}")
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "timestamp": timestamp})
    st.session_state["chat_folders"][st.session_state["active_folder"]] = st.session_state["chat_history"]
    save_chats_to_json()
    st.rerun()

# --- File uploader selalu di bawah ---
uploaded_file = st.file_uploader("Upload dokumen (PDF/TXT/DOCX/Gambar)", type=["pdf", "txt", "docx", "jpg", "jpeg", "png", "bmp", "gif", "webp"])
doc_text = None
image_text = None
image_caption = None
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            doc_text = ""
            for page in reader.pages:
                doc_text += page.extract_text()
        except Exception as e:
            st.error(f"Gagal membaca PDF: {e}")
    elif uploaded_file.type == "text/plain":
        doc_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx
            doc = docx.Document(uploaded_file)
            doc_text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Gagal membaca DOCX: {e}")
    elif uploaded_file.type.startswith("image/"):
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(uploaded_file)
            image_text = pytesseract.image_to_string(image)
            st.image(image, caption="Gambar yang diupload", use_column_width=True)
            # Image Captioning
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                import torch
                @st.cache_resource(show_spinner=False)
                def load_blip():
                    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                    return processor, model
                processor, model = load_blip()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                image_caption = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"Penjelasan gambar: {image_caption}")
            except Exception as e:
                st.warning(f"Gagal melakukan image captioning: {e}")
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")
    if doc_text:
        st.session_state['doc_text'] = doc_text
        st.success("Isi dokumen berhasil dibaca!")
    elif image_text:
        st.session_state['doc_text'] = image_text
        st.success("Teks dari gambar berhasil dibaca!")
    elif image_caption:
        st.session_state['doc_text'] = image_caption
        st.success("Penjelasan gambar berhasil diambil!")
else:
    st.session_state['doc_text'] = None
