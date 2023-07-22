from src.config import config
from langchain.llms import HuggingFaceHub # hugging face can replace the openAI model
from src.api.wz13 import wizardVicuna13
from src.api.repli import Replicate
from langchain.chat_models import ChatOpenAI
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from src.api.octoAICloud import OctoAiCloudLLM
from src.api.huggingface_endpoint import HuggingFaceEndpoint

import os

def switchLLM():
    llm=None
   
    if config['LLM_Name']=='ChatOpenAI':
        llm = ChatOpenAI()
    elif config['LLM_Name']=='flan-t5-xxl':
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs=
                         {"temperature":0.5, "max_length":1024,
                           "top_k": 30,"top_p": 0.9, "repetition_penalty": 1.02})
    elif config['LLM_Name']=='huggingCustomEndpoint':
        llm= HuggingFaceEndpoint(endpoint_url=os.getenv('ENDPOINT_URL'),task="text-generation",
                              model_kwargs={"max_new_tokens": 512, "top_k": 30, "top_p": 0.9, "temperature": 0.2, "repetition_penalty": 1.02,})
    elif config['LLM_Name']=='wizardVicuna13_local':
        llm=wizardVicuna13()
    elif config['LLM_Name']=='vicuna13b_replicate':
        llm = Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                        input= {"max_length":8000,"max_new_tokens": 8000})
    elif config['LLM_Name']=='llama_v2_13b_replicate':
        llm = Replicate(model="a16z-infra/llama13b-v2-chat:5c785d117c5bcdd1928d5a9acb1ffa6272d6cf13fcb722e90886a0196633f9d3",
                        input= {"max_length":8000,"max_new_tokens": 8000})
    elif config['LLM_Name']=='llama_v2_70b_replicate':
        llm = Replicate(model="replicate/llama70b-v2-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
                        input= {"max_length":8000,"max_new_tokens": 8000})
    elif config['LLM_Name']=='falcon7b_octoAI':
        llm = OctoAIEndpoint(
            model_kwargs={
                "max_new_tokens": 200,
                "temperature": 0.75,
                "top_p": 0.95,
                "repetition_penalty": 1,
                "seed": None,
                "stop": [],
            },
        )
    print('llm_model: ',config['LLM_Name'])
    return llm