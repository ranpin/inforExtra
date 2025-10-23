import json
import pandas as pd
import io
import sys
import os
import time
from contextlib import redirect_stdout, redirect_stderr
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
        dict: 包含提取的人物和车内空间位置信息
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
            "content": "我会严格按照要求的JSON格式返回结果，不会遗漏任何字段："
        },
        {
            "role": "user",
            "content": 
                '''# 请严格按照以下格式返回，不要遗漏任何字段:
                    {
                        "persons": [
                            {
                            "name": "识别出的人物（姓名或称呼）",
                            "identity": "身份描述（如果有）",
                            "location": "该人物所在的车内空间位置"
                            }
                        ]
                    }

                    # 提取规则：
                    ## 1. 人物识别原则：
                       - 只提取具体的个人姓名或称呼（如"张三"、"李老师"、"小王"、"小王"、"小斑"等）
                       - 严格禁止提取人称代词："你"、"我"、"他"、"她"、"它"、"你们"、"我们"、"他们"、"她们"、"它们"、"人"、"大家"等
                       - 严格禁止提取方位词作为人名："左边"、"右边"、"前排"、"后排"、"左侧"、"右侧"、"后排左边"、"后排右边"等
                       - 严格禁止提取公司名、品牌名作为人名："日产"、"丰田"、"吉利"等
                       - 当用户明确否定某身份时（包含"不是"、"别叫"等否定词），不得提取被否定的身份
                       - 不能添加用户输入中不存在的信息
                       - 每个唯一的人物只应提取一次，严格避免重复条目
                       
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
                       - 当输入中包含否定句式时，只提取被肯定提及的人物，排除被否定的称谓
                       - 当无法确定具体姓名或称呼时，不创建人物条目
                       - name字段不能为空，无法提取具体姓名时不添加该人物条目
                       - 如果没有任何有效信息，返回空的persons列表
                       - 严格避免为同一个人物创建多个条目，即使在不同位置提及

                    # 请遵循上述提取规则，严格按照以下思维链进行分析：
                    - Step 1: 仔细阅读用户输入，识别所有可能的人名候选词
                    - Step 2: 按照提取规则从候选词中排除禁止提取的词汇（人称代词、方位词、公司名等）
                    - Step 3: 严格检查是否为否定句式，排除被否定的称谓
                    - Step 4: 为每个保留的人名提取完整的位置和身份信息
                    - Step 5: 严格检查是否添加了用户输入中不存在的信息
                    - Step 6: 严格检查是否为同一个人物创建了多个条目，确保每人只有一条记录
                    - Step 7: 构造最终的JSON输出，确保严格符合指定格式
                    
                    # 负面示例（避免错误提取，学习正确提取和正确输出）：
                    
                    - 示例1:
                    用户输入："副驾是日产的小陈同学"
                    错误提取：提取了"日产"作为人名
                    正确提取：只提取"小陈"作为人名，位置为"副驾"，身份为"同学"
                    正确输出：只输出一条：{"name": "小陈", "identity": "同学", "location": "副驾"}

                    - 示例2:
                    用户输入："后排左边坐的是陈老师问个好吧"
                    错误提取：位置只提取了"后排"
                    正确规则：位置应提取"后排左边"
                    正确输出：{"name": "陈老师", "identity": "", "location": "后排左边"}
                    
                    # 正面示例（应该这样做）：
                    - 示例1:
                    用户输入："坐在我左边的是陈总"
                    正确输出：{"name": "陈总", "identity": "", "location": "左边"}
                    
                    - 示例2:
                    用户输入："副驾坐着我的妻子李四"
                    正确输出：{"name": "李四", "identity": "妻子", "location": "副驾"}
                    
                    - 示例3:
                    用户输入："后排坐的是陈老师"
                    正确输出：{"name": "陈老师", "identity": "", "location": "后排"}
                    
                    - 示例4:
                    用户输入："我叫小陈"
                    正确输出：{"name": "小陈", "identity": "", "location": ""}
                    
                    - 示例5:
                    用户输入："我来介绍一下右边是日产的陈总给他打个招呼吧"
                    正确输出：{"name": "陈总", "identity": "", "location": "右边"}

                    - 示例6:
                    用户输入："我来介绍一下左边是日产的陈总给他打个招呼吧"
                    错误提取：提取了"左边"作为人名，或者提前了"他"作为人名
                    正确提取：只提取"陈总"作为人名，位置为"左边"
                    正确输出：{"name": "陈总", "identity": "", "location": "左边"}

                    - 示例7:
                    用户输入："我不是陈总我叫陈部长"
                    错误提取：输出了否定身份"陈总"
                    正确提取：只输出"陈部长"，而不输出"陈总"
                    正确输出：只输出一条：{"name": "陈部长", "identity": "", "location": ""}
                    
                    - 示例8:
                    用户输入："别叫我陈总我是小陈"
                    错误提取：输出了否定身份"陈总"
                    正确提取：只输出"小陈"，而不输出"陈总"
                    正确输出：只输出一条：{"name": "小陈", "identity": "", "location": ""}
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


def chat_with_model(conversation_history):
    """
    与模型对话
    
    Args:
        conversation_history: 对话历史
        
    Returns:
        str: 模型的回答
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
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
                        
                return response_text
            else:
                return "模型调用失败"
        except Exception as e:
            retry_count += 1
            if "RateLimitError" in str(type(e)) or "429" in str(e):
                print(f"遇到速率限制，等待5秒后重试 ({retry_count}/{max_retries})")
                time.sleep(5)
            else:
                print(f"调用模型时发生错误: {e}")
                return f"调用模型时发生错误: {e}"
    
    return "达到最大重试次数，模型调用失败"


def format_entities_output(entities):
    """
    格式化实体输出
    
    Args:
        entities: 实体字典
        
    Returns:
        str: 格式化后的输出
    """
    # 检查是否有有效的人员信息
    persons = entities.get("persons", [])
    # 只有当有姓名时才视为有效（不再进行额外过滤）
    valid_persons = [p for p in persons if p.get("name")]
    
    formatted_output = ""
    if valid_persons:
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
                formatted_output += f"{i}. {name}{identity_str} 位于 {location}\\n"
            else:
                formatted_output += f"{i}. {name}{identity_str}\\n"
    else:
        # 如果没有提取到有效信息，输出提示
        formatted_output = "未检测到明确的车内人员信息"
        
    return formatted_output.strip()


def format_output(response_text, entities):
    """
    格式化输出回答和提取的信息（与qwen_chat.py保持一致）
    
    Args:
        response_text: 模型的回答文本
        entities: 提取的实体信息
    """
    output = ""
    output += "🤖 Qwen 车载助手回答:\n"  # 使用真正的换行符
    output += response_text + "\n"
    output += "-----------------------------------------------------------------------\n"
    
    # 检查是否有有效的人员信息
    persons = entities.get("persons", [])
    # 只有当有姓名时才视为有效（不再进行额外过滤）
    valid_persons = [p for p in persons if p.get("name")]
    
    if valid_persons:
        output += "📋 关键人物信息提取:\n"
        # 先输出完整的JSON格式
        output += json.dumps(entities, ensure_ascii=False, indent=2) + "\n"
        # 再输出编号列表
        output += "\n"
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
                output += f"   {i}. {name}{identity_str} 位于 {location}\n"
            else:
                output += f"   {i}. {name}{identity_str}\n"
    else:
        # 如果没有提取到有效信息，输出提示
        output += "⚠️  未检测到明确的车内人员信息\n"
    
    return output


def process_test_set(input_file, output_file, log_file):
    """
    处理测试集Excel文件
    
    Args:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径
        log_file: 日志文件路径
    """
    # 重定向标准输出和标准错误到字符串缓冲区
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            # 读取Excel文件
            df = pd.read_excel(input_file)
            
            # 获取第一列的列名
            first_column = df.columns[0]
            
            # 重命名列
            df.rename(columns={first_column: '用户输入'}, inplace=True)
            
            # 添加需要的列
            df['JSON输出'] = ''
            df['格式化输出'] = ''
            df['模型回答'] = ''
            
            # 对话历史记录
            conversation_history = []
            
            # 打印欢迎信息（与qwen_chat.py保持一致）
            print("=================================================================================")
            print("🚗 欢迎使用Qwen车载智能助手")
            print("💡 请描述车内人员情况或发出指令")
            print('📌 例如: "我是司机张三，副驾坐着我的妻子李四，后排坐着我的儿子小明"')
            print("📌 输入 'quit'、'exit' 或 '退出' 结束对话")
            print("=================================================================================")
            print()
            
            # 处理每一行
            for index, row in df.iterrows():
                user_input = str(row['用户输入']) if not pd.isna(row['用户输入']) else ""
                
                if user_input:
                    # 打印用户输入（与qwen_chat.py保持一致）
                    print(f"===================================================================================")
                    print(f"👤 用户: {user_input}")
                    
                    # 添加用户消息到对话历史
                    user_message = {
                        "role": "user",
                        "content": user_input
                    }
                    conversation_history.append(user_message)
                    
                    # 调用OpenAI模型
                    response_text = chat_with_model(conversation_history)
                    
                    if not response_text.startswith("调用模型时发生错误") and not response_text.startswith("达到最大重试次数"):
                        # 添加模型回复到对话历史
                        assistant_message = {
                            "role": "assistant",
                            "content": response_text
                        }
                        conversation_history.append(assistant_message)
                        
                        # 只对用户输入进行信息提取，不处理模型回答
                        entities = extract_entities_with_model(user_input)
                        
                        # 格式化输出（与qwen_chat.py保持一致）
                        formatted_output = format_output(response_text, entities)
                        print(formatted_output, end='')  # 使用end=''避免重复换行
                        
                        # 保存JSON输出
                        df.at[index, 'JSON输出'] = json.dumps(entities, ensure_ascii=False, indent=2)
                        
                        # 保存格式化输出（去除转义字符）
                        entities_formatted_output = format_entities_output(entities)
                        df.at[index, '格式化输出'] = entities_formatted_output.replace("\\n", "\n")
                        
                        # 保存模型回答（去除转义字符）
                        df.at[index, '模型回答'] = response_text.replace("\\n", "\n")
                        
                        # 控制对话历史长度，避免过长
                        if len(conversation_history) > 10:  # 最多保留10轮对话
                            conversation_history = conversation_history[-6:]  # 保留最近6条
                    else:
                        print(f"❌ 请求失败: {response_text}")
                        # 即使模型调用失败，也要保存提取的信息
                        entities = extract_entities_with_model(user_input)
                        formatted_output = format_output(response_text, entities)
                        print(formatted_output, end='')  # 使用end=''避免重复换行
                        
                        df.at[index, 'JSON输出'] = json.dumps(entities, ensure_ascii=False, indent=2)
                        entities_formatted_output = format_entities_output(entities)
                        df.at[index, '格式化输出'] = entities_formatted_output.replace("\\n", "\n")
                        df.at[index, '模型回答'] = response_text.replace("\\n", "\n")
                else:
                    print(f"第{index+1}行为空，跳过")
            
            # 保存到Excel文件
            df.to_excel(output_file, index=False)
            
            # 打印结束信息（与qwen_chat.py保持一致）
            print("===================================================================================")
            print("👤 用户: quit")
            print()
            print("👋 感谢使用Qwen车载助手，祝您行车安全!")
            
        except Exception as e:
            print(f"处理测试集时发生错误: {e}")
            raise e
        finally:
            # 将捕获的输出写入日志文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== 标准输出 ===\n")
                f.write(stdout_buffer.getvalue())
                f.write("\n=== 标准错误 ===\n")
                f.write(stderr_buffer.getvalue())

if __name__ == "__main__":
    input_file = "test/测试集.xlsx"
    output_file = "test/测试集_结果.xlsx"
    log_file = "test/处理日志.txt"
    
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在")
        sys.exit(1)
    
    process_test_set(input_file, output_file, log_file)
    print(f"测试集处理完成，结果已保存到 {output_file}")
    print(f"处理日志已保存到 {log_file}")