import json
import pandas as pd
import io
import sys
import os
import time
from contextlib import redirect_stdout, redirect_stderr
from jev_extract import extract_entities
from model_config import (
    client,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM
)

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


def process_test_set(input_file, output_file, log_file, skip_chat=False):
    """
    处理测试集Excel文件

    Args:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径
        log_file: 日志文件路径
        skip_chat: True 时跳过对话模型调用，只跑 Jev 信息提取
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

                    # 调用OpenAI模型（--skip-chat 模式下跳过对话，只跑 Jev 提取）
                    if skip_chat:
                        response_text = "(--skip-chat: 未调用对话模型)"
                    else:
                        response_text = chat_with_model(conversation_history)
                    
                    if not response_text.startswith("调用模型时发生错误") and not response_text.startswith("达到最大重试次数"):
                        # 添加模型回复到对话历史
                        assistant_message = {
                            "role": "assistant",
                            "content": response_text
                        }
                        conversation_history.append(assistant_message)
                        
                        # 只对用户输入进行信息提取，不处理模型回答
                        entities = extract_entities(user_input)
                        
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
                        entities = extract_entities(user_input)
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

    skip_chat = "--skip-chat" in sys.argv
    if "--out" in sys.argv:
        output_file = sys.argv[sys.argv.index("--out") + 1]
    if "--log" in sys.argv:
        log_file = sys.argv[sys.argv.index("--log") + 1]

    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在")
        sys.exit(1)

    process_test_set(input_file, output_file, log_file, skip_chat=skip_chat)
    print(f"测试集处理完成，结果已保存到 {output_file}")
    print(f"处理日志已保存到 {log_file}")