import os
os.environ["OPENAI_API_KEY"] = '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'

from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.embeddings import HuggingFaceEmbeddings
from numpy import dot
from numpy.linalg import norm
import numpy as np
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.chains import LLMChain

#functions

#유사도
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
#요약
sum_prompt_template = PromptTemplate.from_template("{콘텐츠} \n 위 내용을 간단히 요약해줘.")
sum_llm = ChatOpenAI()
sum_chain = LLMChain(llm = sum_llm, prompt = sum_prompt_template)

#토큰길이
tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

#텍스트 split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=100, length_function = tiktoken_len
)

#text embedding (공통)
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#document loader
root="/Users/matthew/Desktop/CookieLearn-backend/app"
file1 = root+"/analysis_result/all_notext.csv"
file2 = root+"/analysis_result/df_sp_dropped.csv"
file3 = root+"/analysis_result/all_pairs.csv"

csv1 = CSVLoader(file_path=file1, encoding="utf-8")
data1 = csv1.load()

csv2 = CSVLoader(file_path=file2, encoding="utf-8")
data2 = csv2.load()

df3 = pd.read_csv(file3,names=["text_headline","text_sentence","bias_label"])

lhl = df3["text_headline"].values.tolist()
lts = df3["text_sentence"].values.tolist()
lbl = df3["bias_label"].values.tolist()

#vector store
db1 = Chroma.from_documents(data1, ko)
db2 = Chroma.from_documents(data2, ko)

#retrievers
llm1 = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.2)
llm2 = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.2)

metadata_field_info = [
    AttributeInfo(
        name="time",
        description="기사의 작성 시간을 의미한다.",
        type="string",
    ),
    AttributeInfo(
        name="category_name",
        description="기사의 카테고리를 의미한다.",
        type="string",
    ),
    AttributeInfo(
        name="text_company",
        description="기사가 작성된 언론사를 의미한다. 각 언론사는 중립을 추구해야 하지만, 일부 언론사의 기사는 특정한 정치적 편향도를 갖는 경우가 있다.",
        type="string",
    ),
    AttributeInfo(
        name="text_headline",
        description="기사의 헤드라인을 의미한다. 전반적인 기사의 내용 중 가장 임팩트있는 내용이거나 주요 내용을 요약한 것과 같다. 기사의 추천을 원하는 경우 기사의 헤드라인을 출력해 줄 수 있다.",
        type="integer",
    ),
    AttributeInfo(
        name="content_url",
        description="원본 기사가 존재하는 url이다. 기사의 원문 또는 추천을 원하는 경우 url값을 추가로 출력해 주는 것이 좋다. ",
        type="string",
    ),
    AttributeInfo(
        name="article_clusters",
        description="유사한 정치적 성향의 기사들을 라벨링 해둔 값이다. 같은 값을 갖는 기사들이 서로 유사한 양상을 띄는 것으로 볼 수 있다. 비슷한 기사를 추천해 달라는 질문에 대답을 하기 위해 사용된다. 어떤 기사와 유사한 기사를 추천해 달라고 할 때, 같은 article cluster 값을 지닌 다른 기사를 추천해 주면 된다.",
        type="int",
    ),
    AttributeInfo(
        name="betweenenss",
        description="이 기사와 유사한 내용을 공유하는 기사의 개수와 관련한 값이다. 값이 클수록 해당 기사가 속한 유형(article cluster)를 대표할 수 있다고 볼 수 있다. 특정 성향의 대표적인 기사를 추천해 달라는 질문에 사용될 수 있다.",
        type="float",
    ),
    AttributeInfo(
        name="closeness",
        description="이 기사와 같은 주제를 다룬 다른 모든 기사들과의 유사성 간의 평균이다. 값이 클수록 해당 기사가 현재 다루는 주제의 기사를 잘 대표하는 기사가 될 수 있다. 값이 클수록 중도성향에 가깝다. 어떤 주제의 대표적인 기사를 추천해 달라는 질문에 사용될 수 있다.",
        type="float",
    ),
    AttributeInfo(
        name="pagerank",
        description="기사의 접근성을 수치화 한 값이다. 값이 클수록 다른 기자들이 해당 기사를 많이 인용한 것이다. 즉, 값이 클수록 해당 기사가 현재 다루는 주제의 기사를 잘 대표하는 기사가 될 수 있다. 어떤 주제의 대표적인 기사를 추천해 달라는 질문에 closeness와 함꼐 사용될 수 있다. ",
        type="string",
    ),
]

document_content_description1 = "최근 뉴스 기사의 정치적 성향을 포함하여 요약한 정보"
document_content_description2 = "최근 뉴스 기사 중 대표값이나 유사성을 구분하는데 도움되는 값을 요약한 정보"

sqretriever1 = SelfQueryRetriever.from_llm(llm1,db1,document_content_description1,metadata_field_info,verbose=True)
sqretriever2 = SelfQueryRetriever.from_llm(llm2,db2,document_content_description2,metadata_field_info,verbose=True)

#qa model
qa1 = RetrievalQA.from_chain_type(llm = llm1, chain_type = "stuff", retriever = db1.as_retriever(search_type="mmr", search_kwargs={'k':3, 'fetch_k': 10}), return_source_documents = True)
qa2 = RetrievalQA.from_chain_type(llm = llm2, chain_type = "stuff", retriever = db2.as_retriever(search_type="mmr", search_kwargs={'k':3, 'fetch_k': 10}), return_source_documents = True)

embeddings = ["중립적인 기사로 요약해줘","보수적인 기사 알려줘","진보적인 기사 알려줘","정치적인 기사 알려줘","대표적인 기사 알려줘","유사한 기사 알려줘","상반되는 기사 알려줘"]
ko_embeddings = ko.embed_documents(embeddings)

def langfunction(query): # query : str
    query_q = ko.embed_query(query)
    nrt = 0
    crt = cos_sim(query_q,ko_embeddings[0])
    for i in range(1,7):
        if crt < round(cos_sim(query_q,ko_embeddings[i]),4):
            nrt = i
            crt = round(cos_sim(query_q,ko_embeddings[i]),4)
    if nrt == 0:
        for i in range(1, len(lbl)):
            if float(lbl[i]) <= 0.5 <= float(lbl[i+1]):
                break
        for j in range(i+1,0,-1):
            if cos_sim(ko.embed_query(lts[j]),ko.embed_query(query)) > 0.4:
                break
        for k in range(i+1,len(lbl)):
            if cos_sim(ko.embed_query(lts[k]),ko.embed_query(query)) > 0.4:
                break
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function = tiktoken_len
        )
        texts = text_splitter.split_documents([Document(page_content = lts[j]+"\n"+lts[k])])
        ntext = ""
        for txt in texts:
            ntext += sum_chain.invoke(txt.page_content)['text']
        return sum_chain.invoke(ntext)['text']
    elif nrt <= 3:
        result1 = qa1.invoke(query)
        return(result1['result'])
    else:
        result2 = qa2.invoke(query)
        return(result2['result'])

if __name__ == "__main__":
    print(langfunction("특검과 관련한 보수적인 기사를 추천해줘"))