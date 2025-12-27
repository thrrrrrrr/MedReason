from openai import OpenAI  # openai>=1.0.0
import time
import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import random

import logging

# Global variable to hold API cost across multiple LLM calls
api_total_cost = 0.0

# OpenRouter configuration for DeepSeek-V3.2
OPENROUTER_API_KEY = "xxxxxx"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"  # DeepSeek-V3.2 model

# Pricing for DeepSeek-V3.2 on OpenRouter
DEEPSEEK_INPUT_PRICE = 0.27 / 10 ** 6   # $0.27 per million tokens
DEEPSEEK_OUTPUT_PRICE = 1.10 / 10 ** 6  # $1.10 per million tokens

def init_logger(name=''):
    logger = logging.getLogger(__name__)
    # set the logging level to INFO
    logger.setLevel(logging.INFO)
    # save the log to a file
    handler = logging.FileHandler('logs/{name}-{time}.log'.format(name=name, time=time.strftime("%Y%m%d-%H%M%S")))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

def get_json_from_generated_text(text):
    # extract substring between the first '{' and the last '}'
    start = text.find("{")
    end = text.rfind("}")
    json_str = text[start:end+1]
    json_obj = json.loads(json_str)
    
    return json_obj

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.lower(), t.lower(), relation=r.lower().strip())
    return G

def get_topk_similar_entities(entity, knowledge_graph, emb_model,nodeemb_dict, k=5, filter_threshold = 0.8):
    entity_type = entity["type"]
    # get the entities set from the graph
    node_entities_with_type = knowledge_graph.query('x_type=="{}"'.format(entity_type))['x_name'].unique()
    embeddings_for_node_entities = nodeemb_dict[entity_type]
    # embeddings_for_node_entities = torch.load('{}/{}.pt'.format(type_embeddings_path, entity_type.replace('/','_')), weights_only= False)
    entity_embedding = emb_model.encode(entity["name"])
    # load the embeddings for the type
    
    similarity = emb_model.similarity(entity_embedding, embeddings_for_node_entities)
    val,idx = torch.topk(similarity, k)
    
    topk_similarity = similarity[0][idx]
    top1_similarity = topk_similarity[0][0].item()
    
    idx = idx[similarity[0][idx] > filter_threshold]
    
    
    if len(idx) == 0:
        return [], top1_similarity
    elif len(idx) == 1:
        similar_entity = node_entities_with_type[idx[0]]
        return [similar_entity], top1_similarity
    else:
        return node_entities_with_type[idx].tolist(), top1_similarity
    
    
def find_all_path_KG(question_entities,result_entities,G):
    path_all = []
    for q_entity in question_entities:
        for a_entity in result_entities:
            path_all+= list(nx.all_shortest_paths(G, q_entity.lower(), a_entity.lower()))
    return path_all

def generate_node_embeddings(knowledge_graph_path = 'data/kg_small.csv', emb_model_name = 'abhinand/MedEmbed-large-v0.1'):
    knowledge_graph = pd.read_csv(knowledge_graph_path, low_memory=False)
    
    # Auto-select device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    emb_model = SentenceTransformer(emb_model_name).to(device)
    print(f"Using device for embeddings: {device}")
    types = knowledge_graph['x_type'].unique()
    nodeemb_dict = {}
    for t in types:
        print("generating embeddings for type: ", t)
        entities_in_types = knowledge_graph.query('x_type=="{}"'.format(t))['x_name'].unique()
        type_embeddings = emb_model.encode(list(entities_in_types))
        nodeemb_dict[t] = type_embeddings
    torch.save(nodeemb_dict, 'node_embeddings.pt')
    return

def compute_usage(response):
    """Calculate API usage cost for DeepSeek-V3.2"""
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    cost = {
        "input": input_tokens * DEEPSEEK_INPUT_PRICE,
        "output": output_tokens * DEEPSEEK_OUTPUT_PRICE,
    }

    cost["total"] = sum(cost.values())
    
    return cost

def run_llm(prompt, temperature=0.0, max_tokens=3000, engine="deepseek", max_attempt=10):
    """
    Call DeepSeek-V3.2 via OpenRouter API
    Args:
        prompt: The prompt text
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        engine: Ignored, always uses DeepSeek-V3.2
        max_attempt: Maximum retry attempts
    """
    global api_total_cost
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt}
    ]
    
    flag = 0
    result = None
    
    while flag == 0 and max_attempt > 0:
        max_attempt -= 1
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            result = response.choices[0].message.content
            cost = compute_usage(response)
            
            # Update global API cost
            api_total_cost += cost["total"]
            
            flag = 1
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            result = "openai error, retry"
            time.sleep(2)
    
    return result
    
def coarse_entity_extraction(text,temperature = 0.0, max_tokens = 3000, engine="gpt-4o"):
    Extract_prompt = """ You are a helpful, pattern-following medical assistant. 
Given the text in a medical or biomedical context, precisely extract all entities from the text. 

### Output Format
Strictly follow the JSON structure below. 
The type of each entity MUST STRICTLY BELONG to one type from:
1. gene/protein
2. drug 
3. effect/phenotype 
4. disease 
5. biological_process 
6. molecular_function 
7. cellular_component 
8. exposure 
9. pathway 
10. anatomy

```json
{{
"Entity": [
    {{"id": "1", "type": "some_type", "name": "entity_name"}},
    {{"id": "2", "type": "some_type", "name": "entity_name"}},
]
}}
```

### Example
text: 
Course in Hospital: Mr. Johnson arrived in the ER from nursing home with a three-day history of worsening shortness of breath, yellow-green sputum, and increased sputum production. 
He was subsequently diagnosed with a COPD exacerbation and was satting at 84% on 4L O2 by nasal prongs. 
Medical presciptions : TAB PARACIP 500 MG two TABLETS PER ORAL THRICE DAILY AFTER FOOD FOR 5 DAYS INJ. AUGMENTIN 1.2 GM, INTRAVENOUS, THREE TIMES A DAY X 4 DAYS

output:
```json
{{
"Entity": [
    {{"id": "1", "type": "symptoms", "name": "shortness of breath"}},
    {{"id": "2", "type": "symptoms", "name": "yellow-green sputum"}},
    {{"id": "3", "type": "diseases", "name": "COPD"}},
    {{"id": "4", "type": "symptoms", "name": "nasal prongs"}},
    {{"id": "5", "type": "drugs", "name": "PARACIP"}},
    {{"id": "6", "type": "drugs", "name": "AUGMENTIN"}}
]
}}
```

### Input
text: 
{text}

output:
""" 
    prompt = Extract_prompt.format(text=text)
    result = run_llm(prompt,temperature,max_tokens,engine)
    return result


def most_correlated_enetity_selection(question, query_entity, similar_entities, temperature = 0.0, max_tokens = 3000, engine="gpt-4o"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
    Given a medical question and corresponding answer, an query entity which is extracted from the question, and a list of similar entities.
    Select ONE most correlated entity from the list of similar entities based on the question and query entity.
    SELECTED ENTITY MUST BE IN THE SIMILAR ENTITIES.
    IF there is not suitable entity in the similar entities, directly return the NONE.
    
    ### Output Format
    Strictly follow the JSON structure below:
    ```json
    {{
        "selected_entity": {{
            "name": "selected_entity_name",
            "id": a int number, the index of the selected entity in the similar entities list, from 0 to N-1
            "reason": "reason for choosing this entity"
        }}
    }}
    ```
    
    if there is no suitable entity, return:
    ```json
    {{
        "selected_entity": {{
            "name": "NONE",
            "id": "NONE",
            "reason": "reason for not choosing any entity, you need to explain why the entities in the similar entities list are not suitable"
        }}
    }}
    ```
    
    ### Input:
    question: {question}
    query entity: {query_entity}
    similar entities: {similar_entities}
    
    output:
    """
    # convert the list of similar entities to a string
    similar_entities_str = ', '.join("{}.{}".format(idx, ent) for idx, ent in enumerate(similar_entities))
    prompt = Reformat_prompt.format(question = question, query_entity=query_entity, similar_entities=similar_entities_str)
    result = run_llm(prompt,temperature,max_tokens,engine)
    return result


def QA_reformat_based_on_entity(question, answer, entity_list_text, temperature = 0.0, max_tokens = 5000, engine="gpt-4o"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
Given a medical question and answer, and all a list of entities.
You need to reformat the question and answer into a pair of description and conclusion.

MUST MAKE SURE the conclusion and description paragraphs contain the entities in the entity list.
You can reallocate information from the question to the description and conclusion paragraphs, to make sure the entities in the entity list are included in the description and conclusion paragraphs.
However, you CAN NOT ADD ANY INFORMATION that is not in the question or answer.

### Output Format
Strictly follow the JSON structure below.

```json
{{
"description": {{
    "text" : "The description of the medical question.",
    "entity": [list of entities in the description, should not be empty]
    }},
"conclusion": {{
    "text" : "The conclusion of the medical question.",
    "entity": [list of entities in the description, should not be empty]
    }}
}}
```

### Example
question: 
A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. 
A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. 
The physician orders antibiotic therapy for the patient. The mechanism of action given blocks cell wall synthesis, which of the following was given?

answer: 
Ceftriaxone

entity list:
1.Fever
2.Knee pain
3.Ceftriaxone

output:
```json
{{
"description": {{
    "text" : "A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient.",
    "entities": ["Fever", "Knee pain"]
    }},
"conclusion": {{
    "text" : "Ceftriaxone is the medication given that blocks cell wall synthesis.",
    "entities": ["Ceftriaxone"]
    }}
}}
```

### Input
question:
{question}

answer:
{answer}

entity list:
{entity_list_text}

output:
"""
    Reformat_prompt = Reformat_prompt.format(question=question, answer=answer, entity_list_text=entity_list_text)
    result = run_llm(Reformat_prompt,temperature,max_tokens,engine)
    return result

def QA_reformat_with_entity_extraction(question, answer, knowledge_graph, emb_model,nodeemb_dict, temperature = 0.0, max_tokens = 5000, engine="gpt-4o"):
    QA_text = '''Question: {question}
    Answer: {answer}'''.format(question=question, answer=answer)
    
    all_entities = coarse_entity_extraction(QA_text,temperature,max_tokens,engine)
    all_entities = get_json_from_generated_text(all_entities)
    type_set = set(knowledge_graph['x_type'].unique())
    # using the following code for multiprocessing
    # x_types_dict = nx.get_node_attributes(knowledge_graph, 'x_type')
    # type_set = set(x_types_dict.values())
    result_entities = []
    for entity in all_entities["Entity"]:
        if entity["type"] not in type_set:
            continue
        similar_entities, top1_similarity = get_topk_similar_entities(entity, knowledge_graph, emb_model,nodeemb_dict, k=10, filter_threshold = 0.7)
        
        if similar_entities == []:
            continue

        selected_entity = None
        # if perfect match, directly select the entity
        for ent in similar_entities:
            if entity["name"].lower() == ent.lower():
                selected_entity = {"name": ent, "id": str(similar_entities.index(ent))}
                break
            
        if top1_similarity > 0.85 and selected_entity == None:
            selected_entity = {"name": similar_entities[0], "id": str(0)}
            
        # if no perfect match, use the question to select the most correlated entity
        if selected_entity == None:
            selected_entity = most_correlated_enetity_selection(QA_text, entity["name"], similar_entities)
            selected_entity = get_json_from_generated_text(selected_entity)["selected_entity"]
            
        if selected_entity["name"] != "NONE":
                result_entities.append(similar_entities[int(selected_entity["id"])])
    # remove duplicated entities
    result_entities = list(set(result_entities))
    entities_text = '\n'.join(['{}.{}'.format(idx+1, ent) for idx, ent in enumerate(result_entities)])
    result = QA_reformat_based_on_entity(question, answer, entities_text, temperature, max_tokens, engine)
    
    return result

def path_sampling(path_all, question, answer, topK_reasoning_paths, max_path_number_per_group = 50, engine = "gpt-4o", logger = None):
    # path_all is a list of list, each list is a reasoning path
    # divide the path_all into different groups, paths with same beginning and end are in the same group
    path_groups = {}
    for path in path_all:
        if len(path) < 2:
            continue
        path_key = (path[0], path[-1])
        if path_key not in path_groups:
            path_groups[path_key] = []
        path_groups[path_key].append(path)
    # traverse the path_groups, and sample topK reasoning paths from each group
    sampled_paths = []
    for path_key in path_groups:
        if logger is not None:
            logger.info(f"Sampling for Path group: {path_key}")
        # if the number of paths in the group is more than max_path_number_per_group
        # randomly sample max_path_number_per_group paths from the group before use LLM to futher sample
        if len(path_groups[path_key]) > max_path_number_per_group:
            path_groups[path_key] = random.sample(path_groups[path_key], max_path_number_per_group)
        text_for_group_paths = '\n'.join([str(idx+1) + ':' + '->'.join(inner_list) for idx,inner_list in enumerate(path_groups[path_key])])
        # get the sampled paths in json format
        result_for_group = most_correlated_path_selection(question, text_for_group_paths, answer, topK=topK_reasoning_paths, engine=engine)
        # for each path_i in result_for_group["Paths"], add path_i["path"] to sampled_paths
        for path_i in result_for_group["Paths"]:
            sampled_paths.append(path_i["path"].split('->'))
    return sampled_paths

def llm_generate_answer_with_reasoning(
    question,
    options,
    reasoning,
    engine = 'gpt-4'
):
    prompt = f"""
You are an expert in the medical domain. You need to answer the following question based on the provided reasoning.
YOU MUST USE THE PROVIDED REASONING TO ANSWER THE QUESTION.
If the answer choices are provided, please choose ONE answer from the answer choices.

Question:
{question}
{options}
Reasoning:
{reasoning}
"""
    response = run_llm(prompt, engine=engine)
    return response
        

def most_correlated_path_selection(question, paths_text, answer, topK = 2, temperature = 0.0, max_tokens = 4000, engine="gpt-4o"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
    Given a medical question and possible relation paths that link to the answer.
    Select up to {topK} most correlated entity from the relation paths list based on the question and the answer.
    If total number of paths is less than {topK}, select all of them.
    
    ### Output Format
    Strictly follow the JSON structure below.
    ```json
    {{
    "Paths": [
        {{"ranking": "1", "path": "sample_path_1", reason": "reason for choosing this path"}},
        .....      
    ]
    }}
    ```
    
    ### Input:
    question: {question}
    answer: {answer}
    paths: {paths}
    
    output:
    """
    # convert the list of similar entities to a string
    prompt = Reformat_prompt.format(question = question, answer = answer, paths=paths_text, topK=topK)
    result = run_llm(prompt,temperature,max_tokens,engine)
    result_paths = get_json_from_generated_text(result)
    return result_paths

def llm_judge_answer(llm_output, answer, engine='gpt-4o'):
    prompt = f"""
You are an expert in the medical domain. Given a correct answer, and the answer from medical student.
You need to judge whether the answer from medical student is correct, by comparing the answer from medical student with the correct answer.
Your response must be 'True' or 'False'.
If the answer is correct, please respond with 'True'.
If the answer is wrong, please respond with 'False'.
Correct answer:
{answer}
Answer from medical student:
{llm_output}
"""
    return run_llm(prompt, engine=engine)


def extract_correct_reasoning_path(reasoning_text, temperature=0.0, max_tokens=1000, engine="deepseek"):
    prompt = f"""
You are a medical reasoning expert. Given a reasoning text, extract ONE reasoning path from the "Finding reasoning paths" section.

If there are multiple paths, randomly select one.
If there is no "Finding reasoning paths" section or it's empty, generate a reasonable reasoning path based on the reasoning process.

The output should be a single path in the format: "Entity1 -> Entity2 -> Entity3 -> ... -> EntityN"

Example output:
"Neoadjuvant chemotherapy -> Cisplatin -> Protein-DNA cross-linking -> Therapeutic effect on cancer"

### Input Reasoning Text:
{reasoning_text}

### Output Format:
Return ONLY the reasoning path string, nothing else.

Reasoning path:
"""
    
    result = run_llm(prompt, temperature, max_tokens, engine)
    return result.strip()


def generate_error_reasoning_paths(correct_path, reasoning_text, temperature=0.7, max_tokens=2000, engine="deepseek"):
    prompt = f"""
You are a medical reasoning expert. Given a correct reasoning path and the original reasoning text, generate FOUR incorrect reasoning paths according to the following rules:

Correct reasoning path:
{correct_path}

Original reasoning text:
{reasoning_text}

Generate 4 error paths with these specific modifications:

1. **Error Path 1 (Shuffled)**: Randomly shuffle the order of entities in the reasoning path
   - Keep all entities but change their order
   - Make sure it's logically incorrect

2. **Error Path 2 (Wrong Conclusion)**: Keep the path mostly correct but change the final conclusion to something wrong
   - Keep first N-1 entities the same
   - Change only the last entity to a medically incorrect conclusion

3. **Error Path 3 (Wrong Entity)**: Replace one intermediate entity with an incorrect medical term
   - Keep first and last entities
   - Replace one entity in the middle with a different disease/condition

4. **Error Path 4 (Missing Steps)**: Randomly remove some entities from the path
   - Delete 1-2 intermediate entities
   - Make sure the path becomes logically incomplete

### Output Format:
Return a JSON object with exactly this structure:
```json
{{
    "error_path_1": "shuffled path here",
    "error_path_2": "path with wrong conclusion here",
    "error_path_3": "path with wrong entity here",
    "error_path_4": "path with missing steps here"
}}
```

Generate the 4 error paths:
"""
    
    result = run_llm(prompt, temperature, max_tokens, engine)
    
    try:
        error_paths = get_json_from_generated_text(result)
        return error_paths
    except Exception as e:
        print(f"Error parsing error paths: {e}")
        # 返回默认错误路径
        return {
            "error_path_1": "",
            "error_path_2": "",
            "error_path_3": "",
            "error_path_4": ""
        }


def extract_reasoning_paths_complete(reasoning_text, temperature=0.0, max_tokens=5000, engine="deepseek"):
    # 1. 提取正确的推理路径
    correct_path = extract_correct_reasoning_path(reasoning_text, temperature, max_tokens, engine)
    
    # 2. 生成4条错误的推理路径
    error_paths = generate_error_reasoning_paths(correct_path, reasoning_text, temperature, max_tokens, engine)
    
    # 3. 返回完整结果
    return {
        "correct_reasoning_path": correct_path,
        "error_reasoning_path_1": error_paths.get("error_path_1", ""),
        "error_reasoning_path_2": error_paths.get("error_path_2", ""),
        "error_reasoning_path_3": error_paths.get("error_path_3", ""),
        "error_reasoning_path_4": error_paths.get("error_path_4", "")
    }
    
    
