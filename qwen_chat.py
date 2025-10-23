import json
import os
import re
from dotenv import load_dotenv
from model_config import (
    client,
    DEFAULT_MODEL, 
    DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM,
    EXTRACTION_MODEL,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_MAX_TOKENS,
    EXTRACTION_STREAM
)

def extract_entities_with_model(user_input):
    """
    使用模型从用户输入中提取人物和车内位置信息
    
    Args:
        user_input: 用户输入的文本
        
    Returns:
        dict: 包含提取的人物和车内位置信息
    """
    # 构造提取实体的提示词，专门针对汽车座舱场景
    extraction_prompt = [
        {
            "role": "system",
            "content": "你是一个车载语音助手，你的任务是从用户的输入中准确提取所有提到的人物信息和人物的车内空间位置等信息。只返回JSON格式的结果，不要添加任何解释性文字。"
        },
        {
            "role": "user",
            "content": f"请从以下用户输入中提取人物信息（姓名、称呼、身份等）和车内空间位置信息（如前座、后座、主驾、副驾等）：\\n{user_input}"
        },
        {
            "role": "assistant",
            "content": "我会严格按照要求的JSON格式返回结果："
        },
        {
            "role": "user",
            "content": 
                '''# 请按照以下格式返回:
                    {
                    "persons": [
                        {
                        "name": "人物姓名或称呼",
                        "identity": "身份描述（如果有）",
                        "location": "该人物所在的车内空间位置"
                        }
                    ]
                    }

                    # 提取规则：
                    ## 1. 人物识别原则：
                       - 只提取具体的个人姓名或称呼（如"张三"、"李老师"、"小王"、"小王"、"小斑"等）
                       - 严格禁止提取人称代词："你"、"我"、"他"、"她"、"它"、"你们"、"我们"、"他们"、"她们"、"它们"
                       - 严格禁止提取方位词作为人名："左边"、"右边"、"前排"、"后排"、"左侧"、"右侧"、"后排左边"、"后排右边"等
                       - 严格禁止提取公司名、品牌名作为人名："日产"、"丰田"、"吉利"等
                       - 当用户明确否定某身份时（包含"不是"、"别叫"等否定词），不得提取被否定的身份
                       - 不能添加用户输入中不存在的信息
                       - 每个唯一的人物只应提取一次，避免重复条目
                       
                    ## 2. 位置信息提取原则：
                       - 严格按照用户输入原文提取，不得改写或推测
                       - 只提取车内空间位置信息，如"主驾"、"副驾"、"后排"、"左座"、"右后"等
                       - 提取完整的位置描述，不要截断或简化（如用户说"后排左边"就提取"后排左边"，不要只提取"后排"）
                       - 禁止提取地理地址或地点名称
                       - 每个人物只关联一个最准确的位置描述
                       
                    ## 3. 身份信息提取原则：
                       - 仅提取与人物直接相关的描述性称谓，如"同学"、"工程师"、"老师"等
                       - 禁止将方位词作为身份信息提取
                       
                    ## 4. 特殊情况处理：
                       - 当输入中包含否定句式时，只提取被肯定提及的人物
                       - 当无法确定具体姓名或称呼时，不创建人物条目
                       - name字段不能为空，无法提取具体姓名时不添加该人物条目
                       - 如果没有任何有效信息，返回空的persons列表
                       - 避免为同一个人物创建多个条目，即使在不同位置提及

                    # 请严格按照以下思维链进行分析：
                    Step 1: 仔细阅读用户输入，识别所有可能的人名候选词
                    Step 2: 从候选词中排除禁止提取的词汇（人称代词、方位词、公司名等）
                    Step 3: 检查是否存在否定句式，排除被否定的称谓
                    Step 4: 为每个保留的人名提取完整的位置和身份信息
                    Step 5: 检查是否添加了用户输入中不存在的信息
                    Step 6: 检查是否为同一个人物创建了多个条目，确保每人只有一条记录
                    Step 7: 构造最终的JSON输出
                    
                    # 负面示例（不要这样做）：
                    示例1:
                    用户输入："我不是陈总我叫陈部长"
                    错误输出：提取了"陈总"和"陈部长"
                    正确输出：只提取"陈部长"
                    
                    示例2:
                    用户输入："我来介绍一下左边是日产的陈总给他打个招呼吧"
                    错误输出：提取了"左边"作为人名
                    正确输出：只提取"陈总"，位置为"左边"
                    
                    示例3:
                    用户输入："副驾是日产的小陈同学"
                    错误输出：提取了"日产"作为人名
                    正确输出：只提取"小陈"，身份为"同学"，位置为"副驾"
                    
                    示例4:
                    用户输入："别叫我陈总我是小陈"
                    错误输出：提取了"陈总"和"小陈"
                    正确输出：只提取"小陈"
                    
                    示例5:
                    用户输入："后排左边坐的是陈老师问个好吧"
                    错误输出：位置只提取了"后排"
                    正确输出：位置应提取"后排左边"
                    
                    示例6:
                    用户输入："我来介绍一下右边是日产的陈总给他打个招呼吧"
                    错误输出：提取了"他"作为人名或为同一个人创建了多个条目
                    正确输出：只提取一条记录：{"name": "陈总", "identity": "", "location": "右边"}
                    
                    # 正面示例（应该这样做）：
                    示例1:
                    用户输入："坐在我左边的是陈总"
                    正确输出：{"name": "陈总", "identity": "", "location": "左边"}
                    
                    示例2:
                    用户输入："副驾坐着我的妻子李四"
                    正确输出：{"name": "李四", "identity": "妻子", "location": "副驾"}
                    
                    示例3:
                    用户输入："后排坐的是陈老师"
                    正确输出：{"name": "陈老师", "identity": "", "location": "后排"}
                    
                    示例4:
                    用户输入："我叫小陈"
                    正确输出：{"name": "小陈", "identity": "", "location": ""}
                    
                    示例5:
                    用户输入："我来介绍一下右边是日产的陈总给他打个招呼吧"
                    正确输出：{"name": "陈总", "identity": "", "location": "右边"}
                '''
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=extraction_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
            stream=EXTRACTION_STREAM
        )
        
        if response:
            # 处理流式或非流式响应
            response_text = ""
            if hasattr(response, '__iter__') and not isinstance(response, list):
                # 流式响应
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
            else:
                # 非流式响应
                if response.choices and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
            
            # 尝试解析返回的JSON
            try:
                # 查找第一个 '{' 和最后一个 '}' 之间的内容
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    entities = json.loads(json_str)
                    return entities
                else:
                    return {"persons": []}
                    
            except json.JSONDecodeError:
                # 如果无法解析JSON，则返回空结果
                return {"persons": []}
        else:
            print(f"实体提取失败: {response}")
            return {"persons": []}
    except Exception as e:
        print(f"提取实体时发生错误: {e}")
        return {"persons": []}


def format_output(response_text, entities):
    """
    格式化输出回答和提取的信息
    
    Args:
        response_text: 模型的回答文本
        entities: 提取的实体信息
    """
    print("🤖 Qwen 车载助手回答:")
    print(response_text)
    print("-----------------------------------------------------------------------")
    
    # 检查是否有有效的人员信息
    persons = entities.get("persons", [])
    valid_persons = [p for p in persons if p.get("name")]
    
    if valid_persons:
        print("📋 关键人物信息提取:")
        # 先输出完整的JSON格式
        print(json.dumps(entities, ensure_ascii=False, indent=2))
        # 再输出编号列表
        print()
        for i, person in enumerate(valid_persons, 1):
            name = person.get('name', '未知')
            identity = person.get('identity', '')
            location = person.get('location', '')
            
            # 构建身份信息字符串
            identity_str = ""
            if identity:
                identity_str = f"（{identity}）"
            
            # 如果有位置信息则显示，否则只显示姓名和身份
            if location:
                print(f"   {i}. {name}{identity_str} 位于 {location}")
            else:
                print(f"   {i}. {name}{identity_str}")
    else:
        # 如果没有提取到有效信息，输出提示
        print("⚠️  未检测到明确的车内人员信息")


def chat_with_qwen():
    """
    主对话函数
    """
    print("=================================================================================")
    print("🚗 欢迎使用Qwen车载智能助手")
    print("💡 请描述车内人员情况或发出指令")
    print('📌 例如: "我是司机张三，副驾坐着我的妻子李四，后排坐着我的儿子小明"')
    print("📌 输入 'quit'、'exit' 或 '退出' 结束对话")
    print("=================================================================================")
    print()
    
    # 对话历史记录
    conversation_history = []
    
    while True:
        # 获取用户输入
        user_input = input("\n===================================================================================\n👤 用户: ").strip()
        
        # 检查退出条件
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 感谢使用Qwen车载助手，祝您行车安全!")
            break
            
        if not user_input:
            continue
            
        # 添加用户消息到对话历史
        user_message = {
            "role": "user",
            "content": user_input
        }
        conversation_history.append(user_message)
        
        try:
            # 调用OpenAI模型
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=conversation_history,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=DEFAULT_STREAM
            )
            
            if response:
                # 处理流式或非流式响应
                response_text = ""
                if hasattr(response, '__iter__') and not isinstance(response, list):
                    # 流式响应
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                                response_text += chunk.choices[0].delta.content
                else:
                    # 非流式响应
                    if response.choices and len(response.choices) > 0:
                        response_text = response.choices[0].message.content
                
                # 添加模型回复到对话历史
                assistant_message = {
                    "role": "assistant",
                    "content": response_text
                }
                conversation_history.append(assistant_message)
                
                # 只对用户输入进行信息提取，不处理模型回答
                entities = extract_entities_with_model(user_input)
                
                # 格式化输出
                format_output(response_text, entities)
                
                # 控制对话历史长度，避免过长
                if len(conversation_history) > 10:  # 最多保留10轮对话
                    conversation_history = conversation_history[-6:]  # 保留最近6条
                    
            else:
                print(f"❌ 请求失败: {response}")
                
        except Exception as e:
            print(f"❌ 调用模型时发生错误: {e}")


def demo_example():
    """
    演示示例函数
    """
    print("📝 演示示例:")
    print("用户输入: 我是司机张三，副驾坐着我的妻子李四，后排坐着我的儿子小明")
    
    # 模拟用户输入
    user_input = "我是司机张三，副驾坐着我的妻子李四，后排坐着我的儿子小明"
    
    # 模拟模型回复
    response_text = "您好张师傅！我已经了解了车内的乘坐情况。司机是您，副驾驶坐着您的妻子李四，后排坐着您的儿子小明。如果需要调节座椅、温度或其他功能，请随时告诉我。"
    
    # 模拟提取的实体（只针对用户输入）
    entities = {
        "persons": [
            {"name": "张三", "identity": "司机", "location": "主驾"},
            {"name": "李四", "identity": "妻子", "location": "副驾"},
            {"name": "小明", "identity": "儿子", "location": "后排"}
        ]
    }
    
    # 格式化输出
    format_output(response_text, entities)


if __name__ == "__main__":
    # 检查是否设置了API密钥
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your-api-key-here':
        print("⚠️  请注意: 您需要先设置有效的API密钥")
        print("   请在.env文件中设置您的OPENAI_API_KEY")
        print("   演示模式下将显示示例输出...\\n")
        demo_example()
    else:
        chat_with_qwen()