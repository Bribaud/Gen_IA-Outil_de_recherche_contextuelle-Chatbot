import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Charger les variables d'environnement
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("L'API Key Anthropic est manquante.")

# Configuration de la page
st.set_page_config(page_title="IA Recherche Contextuelle", page_icon="🔎", layout="centered")

# Titre principal
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>IA Recherche Contextuelle</h1>
    <p style='text-align: center; color: gray;'>Trouvez des réponses précises en quelques secondes</p>
""", unsafe_allow_html=True)

# Conteneur principal
with st.container():
    st.subheader("📂 Ajouter des documents")
    urls = [st.text_input(f"URL {i + 1}") for i in range(3)]

    if st.button("🚀 Lancer l'analyse"):
        with st.spinner("Analyse en cours..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
            docs = text_splitter.split_documents(data)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            with open("faiss_store_anthropic.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
        st.success("✅ Analyse terminée !")

# Zone de recherche
st.subheader("🔍 Poser une question")
query = st.text_area("Tapez votre question ici")
if st.button("📡 Rechercher"):
    if os.path.exists("faiss_store_anthropic.pkl"):
        with st.spinner("Recherche en cours..."):
            with open("faiss_store_anthropic.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                llm = ChatAnthropic(model_name="claude-3-opus-20240229", anthropic_api_key=anthropic_api_key)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <h4>📝 Réponse :</h4>
            <p>{}</p>
        </div>
        """.format(result["answer"]), unsafe_allow_html=True)

        if "sources" in result and result["sources"]:
            st.subheader("📌 Sources Utilisées")
            for source in result["sources"].split("\n"):
                st.markdown(f"🔗 [{source}]({source})")
    else:
        st.warning("Veuillez d'abord analyser des documents avant de poser une question.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>© 2025 - IA Recherche Contextuelle</p>
""", unsafe_allow_html=True)