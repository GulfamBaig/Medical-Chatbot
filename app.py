import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from PIL import Image
import requests
from io import BytesIO
import os

# ------------------ App Configuration ------------------ #
st.set_page_config(
    page_title="MediBot - Your Personal Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Utility Functions ------------------ #
def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

@st.cache_resource
def initialize_llm():
    try:
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl",
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.9,
            repetition_penalty=1.1
        )

        template = """
        You are a highly experienced medical doctor named MediBot. Your task is to:
        1. Analyze the patient's symptoms and medical history
        2. Provide a possible diagnosis (list possible conditions by likelihood)
        3. Recommend appropriate medications (considering the patient's country: {country})
        4. Suggest lifestyle recommendations
        5. Advise when to seek immediate medical attention

        Current conversation:
        {history}

        Patient: {input}
        MediBot:"""

        prompt = PromptTemplate(
            input_variables=["history", "input", "country"], 
            template=template
        )

        memory = ConversationBufferMemory(memory_key="history")

        return LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)

    except Exception as e:
        st.error(f"Failed to initialize language model: {e}")
        return None

# ------------------ UI Elements ------------------ #
def render_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2781/2781395.png", width=100)
    with col2:
        st.title("MediBot - Your Personal Medical Assistant")
        st.caption("AI-powered medical diagnosis and treatment recommendations")

def render_sidebar():
    with st.sidebar:
        st.header("About MediBot")
        st.image("https://cdn-icons-png.flaticon.com/512/3309/3309933.png", width=100)
        st.markdown("""
        MediBot is an AI-powered assistant designed to:
        - Diagnose common symptoms
        - Recommend medications
        - Provide wellness guidance

        ⚠️ **Note:** For informational use only. Consult a real doctor for medical concerns.
        """)
        st.divider()

        st.subheader("Example Questions")
        st.markdown("""
        - I have fever, headache, and muscle pain. What could it be?
        - What's the treatment for seasonal allergies in India?
        - I have a rash on my arms and itching. What should I do?
        """)
        
        st.divider()
        st.write("Developed by **[Your Name]**")
        st.write("Version 1.0")

        st.divider()
        api_key = st.text_input("Enter your HuggingFace API key:", type="password")
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# ------------------ Main Chat Logic ------------------ #
def chat_interface():
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        st.warning("Please enter your HuggingFace API key in the sidebar to continue.")
        return

    if st.session_state.llm_chain is None:
        st.session_state.llm_chain = initialize_llm()
        if st.session_state.llm_chain is None:
            return

    country = st.selectbox(
        "Select your country for region-specific recommendations:",
        ["United States", "India", "United Kingdom", "Canada", "Australia", 
         "Germany", "France", "Japan", "Brazil"],
        index=0
    )

    st.subheader("Consultation Chat")
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Describe your symptoms or ask a medical question...")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.spinner("MediBot is analyzing your symptoms..."):
            try:
                response = st.session_state.llm_chain.invoke(
                    {"input": user_input, "country": country}
                )["text"]
                st.session_state.conversation.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")

# ------------------ Button-Based Questions ------------------ #
def quick_access_buttons():
    st.divider()
    col1, col2, col3 = st.columns(3)

    def inject_question(text):
        if not st.session_state.llm_chain:
            st.warning("Initialize the chatbot by entering your API key first.")
            return
        st.session_state.conversation.append({"role": "user", "content": text})
        st.rerun()

    with col1:
        st.subheader("Common Conditions")
        if st.button("Cold & Flu"): inject_question("What are the symptoms and treatment for cold and flu?")
        if st.button("Allergies"): inject_question("How to treat seasonal allergies?")
        if st.button("Headache"): inject_question("What causes headaches and how to relieve them?")

    with col2:
        st.subheader("First Aid")
        if st.button("Burns"): inject_question("What's the first aid for minor burns?")
        if st.button("Cuts"): inject_question("How to treat minor cuts and wounds?")
        if st.button("Fever"): inject_question("When should I worry about a fever?")

    with col3:
        st.subheader("Wellness Tips")
        if st.button("Sleep Better"): inject_question("How can I improve my sleep quality?")
        if st.button("Healthy Diet"): inject_question("What foods should I eat for better health?")
        if st.button("Stress Relief"): inject_question("What are effective ways to reduce stress?")

# ------------------ Educational Sample Images ------------------ #
def sample_images_section():
    st.divider()
    st.subheader("🧠 Sample Medical Images (for educational use only)")

    image_data = {
        "Brain MRI (Tumor)": "https://upload.wikimedia.org/wikipedia/commons/7/75/MRI_T2_Meningioma.jpg",
        "Skin Rash": "https://upload.wikimedia.org/wikipedia/commons/8/81/Morbilliform_rash.JPG",
        "Allergic Reaction": "https://upload.wikimedia.org/wikipedia/commons/2/2e/Allergic_contact_dermatitis_due_to_catechu_-_2009.jpg",
        "Flu Symptoms Chart": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Cold_vs_flu_diagram.svg/2560px-Cold_vs_flu_diagram.svg.png"
    }

    with st.expander("🔍 Click to view sample images"):
        cols = st.columns(4)
        for idx, (label, url) in enumerate(image_data.items()):
            with cols[idx % 4]:
                img = load_image(url)
                if img:
                    st.image(img, caption=label, use_container_width=True)

# ------------------ Initialize State ------------------ #
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = None

# ------------------ Run the App ------------------ #
if __name__ == "__main__":
    try:
        render_header()
        render_sidebar()
        chat_interface()
        quick_access_buttons()
        sample_images_section()

        st.divider()
        st.warning("""
        **Important Disclaimer:** 
        MediBot is an AI assistant for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a licensed healthcare provider for any medical concerns.
        """)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
