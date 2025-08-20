from langchain_ollama import ChatOllama
from langchain_core.example_selectors import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 4.示例：
# 原句：受台风影响，上海市浦东新区和崇明岛受灾严重
# 标注：受台风影响，<location>上海市浦东新区</location>和<location>崇明岛</location>受灾严重

def create_final_prompt(model_name):
    # Initialize large language models
    llm = ChatOllama(
        model=model_name,
        base_url='XXX',   #Replace with your Ollama server URL (e.g., '192.168.1.100:11434')
        temperature=0,
        num_predict=4096,
    )
    if model_name == "deepseek-r1:671b":
        print('The model is excessively large; resort to the use of costly APIs!')
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            # Replace "XXX" with your own API base URL for the service
            openai_api_base="XXX",  # # e.g., "https://api.deepseek.com/v1"
            # Replace "XXX" with your own API key/secret
            openai_api_key="XXX",	# Your authentication credential
            # Replace "XXXX" with your specific DeepSeek model identifier
            model_name="XXXX",	    # e.g., "deepseek-r1"
        )
    # System Template
    system_template = """你是一个灾害信息结构化标注专家，请按照以下规则处理输入文本：
        1. 标注要素：
        - 受灾地点：用<location>包裹，仅标注灾害直接影响区域

        2. 标注规则：
        - 保持原文顺序和文字不变，仅添加标签
        - 同时存在多个实体时按出现顺序标注
        - 复合实体连续标注（如：<location>山区和河流下游</location>）
        - 排除非灾害相关地点（如消息来源地、非受灾区域）

        3. 特殊处理：
        - 排除修饰语（如：<location>沿海某市</location>而非"美丽的沿海某市"）
        - 区域描述需具体（如：<location>第三社区</location>而非"附近地区"）
        - 对于复合区域名称完整标注（如：<location>四川省雅安市</location>）

    """

    # Integrated Final Prompt Template
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    chain = final_prompt | llm
    return final_prompt, chain
    
    
import re
def delresult(name,ori,model_name):
    for i,item in enumerate(name):
        output = item['output']
        if '</think>' in output:
            output = output.split('</think>')[1]
        times = re.findall(r'<time>(.*?)</time>', output)
        locations = re.findall(r'<location>(.*?)</location>', output)
        disasters = re.findall(r'<disaster>(.*?)</disaster>', output)
        
        ori[i][f'{model_name}_label_json'] = {
            'time': times,
            'location': locations,
            'disaster': disasters
        } 
        ori[i][f'{model_name}_output']=output

def add_label_BIO(data, model_name,separator='\002'):
    original_count = len(data)
    filtered_data = []
    
    for item in data:
        input_str = item['input']
        bio_labels = ['O'] * len(input_str)
        
        # 遍历每个实体类型
        for entity_type in ['time', 'location', 'disaster']:
            entities = item[f'{model_name}_label_json'].get(entity_type, [])
            for entity in entities:
                start_idx = input_str.find(entity)
                if start_idx == -1:
                    continue
                end_idx = start_idx + len(entity)
                
                if end_idx > len(input_str):
                    continue
                
                # 标注B/I标签
                bio_labels[start_idx] = f'{entity_type}-B'
                for i in range(start_idx + 1, end_idx):
                    if i < len(bio_labels):
                        bio_labels[i] = f'{entity_type}-I'
        
        # 保存BIO标注结果
        item[f'{model_name}_label_BIO'] = separator.join(bio_labels)
        filtered_data.append(item)
    
    return filtered_data
    
def calculate_metrics(test_data, model_name):
    # 初始化统计变量
    total_chars = 0
    correct = 0
    tp_entities = 0
    fp_entities = 0
    fn_entities = 0

    for sample in test_data:
        # 获取真实和预测的BIO标签
        true_tags = sample['label_BIO'].split('\002')
        pred_tags = sample[f'{model_name}_label_BIO'].split('\002')
        assert len(true_tags) == len(pred_tags), "标签长度不一致"
        
        total_chars += len(true_tags)
        for t, p in zip(true_tags, pred_tags):
            if t == p:
                correct += 1
                if t != 'O':
                    tp_entities += 1
            else:
                if p != 'O':
                    fp_entities += 1
                if t != 'O':
                    fn_entities += 1

    # 计算acc
    accuracy = correct / total_chars if total_chars > 0 else 0

    # precision、recall、F1
    precision_denominator = tp_entities + fp_entities
    precision = tp_entities / precision_denominator if precision_denominator > 0 else 0

    recall_denominator = tp_entities + fn_entities
    recall = tp_entities / recall_denominator if recall_denominator > 0 else 0

    f1_denominator = precision + recall
    f1 = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }