from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load CSVs
df_lignes = pd.read_csv("r_lignes.csv")
df_trains = pd.read_csv("trains.csv")

# Create embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Set up vector store
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    # Create documents for lines
    for i, row in df_lignes.iterrows():
        content = f"LigneID: {row['ligneid']}, NumeroLigne: {row['numeroligne']}, SeuilKM: {row['seuilkm']}"
        document = Document(
            page_content=content,
            metadata={"source": "r_lignes", "ligneid": row['ligneid']},
            id=f"ligne-{i}"
        )
        documents.append(document)
        ids.append(f"ligne-{i}")
    
    # Create documents for trains
    for i, row in df_trains.iterrows():
        content = (
            f"TrainID: {row['trainid']}, LigneID: {row['ligneid']}, "
            f"TypeMaterielID: {row['typematerielid']}, CodeGMAO: {row['codegmao']}, "
            f"TypeTrame: {row['type_trame']}, Prefixe: {row['prefixe']}, "
            f"IDTrainComposition: {row['id_train_composition']}, CodeNavette: {row['codenavette']}, "
            f"NumExploitant: {row['num_exploitant']}, NumGMAO: {row['num_gmao']}"
        )
        document = Document(
            page_content=content,
            metadata={"source": "trains", "ligneid": row['ligneid'], "trainid": row['trainid']},
            id=f"train-{i}"
        )
        documents.append(document)
        ids.append(f"train-{i}")
        
vector_store = Chroma(
    collection_name="train_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)