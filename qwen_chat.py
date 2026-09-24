import json
import os
import re
from dotenv import load_dotenv
from jev_extract import extract_entities
from model_config import (
    client,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM
)

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
                entities = extract_entities(user_input)
                
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