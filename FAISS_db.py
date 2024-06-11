"""
Script to create vectorstore for webscrapped documents and write to pickle file.
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name='emilyalsentzer/Bio_ClinicalBERT')

# 定义文件路径
file_path = '/Users/liangxin/Downloads/vector/notesall.txt'

def read_in_chunks(file_path, chunk_size=1024*1024):
    """逐块读取文件的生成器函数。"""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data

def process_and_store_chunks(file_path, chunk_size=1024*1024, slice_size=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"])
    all_texts = []

    # 读取文件的每个块并处理
    for chunk in read_in_chunks(file_path, chunk_size):
        all_texts.append(chunk)

    # 使用 save_documents_slice 函数保存文档
    db = save_documents_slice(all_texts, slice_size=slice_size)

    return db

def save_documents_slice(texts, index="finance_index", slice_size=100):
    print("documents:", str(len(texts)))
    len_texts = len(texts)
    db_num = 1  # 定义第一个初始值，当为1时使用原始db，否则新建一个db

    for i in range(0, len_texts, slice_size):
        slice_texts = texts[i:i + slice_size]
        print("当前第:", str(i))
        item_slice_texts = []

        for text in slice_texts:
            if len(text) < 2:
                print("当前内容过少跳过:", text)
                continue
            item_slice_texts.append(text)

        if item_slice_texts:
            if db_num == 1:
                db = FAISS.from_texts(item_slice_texts, embeddings)
            else:
                db1 = FAISS.from_texts(item_slice_texts, embeddings)
                db.merge_from(db1)  # 新的db和原始db合并

            db_num += 1

    db.save_local(index)
    return db

# 处理大文件，分块处理并存储嵌入
db = process_and_store_chunks(file_path)

# 将索引序列化为字节数据
pkl = db.serialize_to_bytes()

# 将序列化后的数据保存到一个 pickle 文件中
with open('faiss_store1', 'wb') as f:
    pickle.dump(pkl, f)

print("FAISS 索引已创建并保存到 'faiss_store1'")

