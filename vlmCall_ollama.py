import requests
import json
import random
import base64
from datetime import datetime
from PIL import Image
import io
import time
import os
from pydantic import BaseModel

# 基础API功能，不包含业务逻辑


def get_data_engine_path():
    """获取data_engine目录的绝对路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir


def get_project_root():
    """获取项目根目录的绝对路径"""
    data_engine_path = get_data_engine_path()
    return os.path.dirname(data_engine_path)


def load_prompt_config():
    """Load English prompt configuration only."""
    config_path = "config/prompt_config_en.json"
    fallback_path = "config/prompt_config.json"  # Legacy fallback

    if not os.path.isabs(config_path):
        config_path = os.path.join(get_data_engine_path(), config_path)
    if not os.path.isabs(fallback_path):
        fallback_path = os.path.join(get_data_engine_path(), fallback_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open(fallback_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Warning: Using legacy config {fallback_path}, consider migrating to {config_path}")
                return config
        except FileNotFoundError:
            print(f"Warning: Neither {config_path} nor {fallback_path} found, using default configuration")
            return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {config_path}, using default configuration")
        return {}

# 默认配置
PROMPT_CONFIG = load_prompt_config()


class VLMRequestError(Exception):
    pass  

class VLMAPI:
    def __init__(self, model):  # qwen2.5vl:32b, llava:7b, etc.
        self.model = model
        self.api_url = "http://110.42.252.68:8080/api/generate"
        

    def encode_image(self, image_path):
        """编码图像为base64格式"""
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            if original_width == 1600 and original_height == 800:
                new_width = original_width // 2
                new_height = original_height // 2
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                resized_img.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        return base64_image
    

    def vlm_request(self,
                    systext,
                    usertext,
                    image_path1=None,
                    image_path2=None,
                    image_path3=None,
                    max_tokens=1500,
                    retry_limit=3):
        """
        发送VLM请求到Ollama API
        
        Args:
            systext: 系统提示文本
            usertext: 用户提示文本
            image_path1/2/3: 图像路径（可选）
            max_tokens: 最大token数
            retry_limit: 重试次数
        """
        # print("===== VLM SYSTEXT =====\n%s", systext)
        # print("===== VLM USERTEXT =====\n%s", usertext)
        
        # 构建完整的提示文本
        full_prompt = f"{systext}\n\n{usertext}"
        
        # 准备图像数据
        images = []
        if image_path1:
            base64_image1 = self.encode_image(image_path1)
            images.append(base64_image1)
        if image_path2:
            base64_image2 = self.encode_image(image_path2)
            images.append(base64_image2)
        if image_path3:
            base64_image3 = self.encode_image(image_path3)
            images.append(base64_image3)
        
        # 构建Ollama API请求payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.9
            }
        }
        
        # 如果有图像，添加到payload中
        if images:
            payload["images"] = images
        
        retry_count = 0
        while retry_count < retry_limit: 
            try:
                t1 = time.time()
                print(f"********* start call {self.model} *********")
                
                # 发送请求到Ollama API
                response = requests.post(self.api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("response", "")
                    
                    current_time = int(datetime.now().timestamp())  
                    formatted_time = datetime.utcfromtimestamp(current_time).strftime("%Y/%m/%d/%H:%M:%S")
                    
                    # 记录API调用（可选）
                    # record = {
                    #     "model": self.model,
                    #     "prompt": full_prompt,
                    #     "response": data,
                    #     "current_time": formatted_time
                    # }
                    # save_path = f"./data/{self.model}/apiRecord.json"
                    # save_data_to_json(record, save_path)

                    t2 = time.time() - t1
                    print(f"********* end call {self.model}: {t2:.2f} *********")
                    
                    return content
                else:
                    print(f"API request failed with status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as ex:
                print(f"Attempt call {self.model} {retry_count + 1} failed: {ex}")
                time.sleep(300)
                retry_count += 1
        
        return "Failed to generate completion after multiple attempts."
    
    def vlm_request_with_format(self,
                               systext,
                               usertext,
                               format_schema=None,
                               image_path1=None,
                               image_path2=None,
                               image_path3=None,
                               options=None,
                               retry_limit=3):
        """
        发送VLM请求到Ollama API，支持结构化输出格式
        
        Args:
            systext: 系统提示文本
            usertext: 用户提示文本
            format_schema: JSON schema for structured output (optional)
            image_path1/2/3: 图像路径（可选）
            options: 请求选项字典 (temperature, num_predict等)
            retry_limit: 重试次数
        
        Returns:
            str: API响应内容
        """
        # 构建完整的提示文本
        full_prompt = f"{systext}\n\n{usertext}"
        
        # 准备图像数据
        images = []
        if image_path1:
            base64_image1 = self.encode_image(image_path1)
            images.append(base64_image1)
        if image_path2:
            base64_image2 = self.encode_image(image_path2)
            images.append(base64_image2)
        if image_path3:
            base64_image3 = self.encode_image(image_path3)
            images.append(base64_image3)
        
        # 构建基础payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": options or {}
        }
        
        # 添加结构化输出格式
        if format_schema:
            payload["format"] = format_schema
        
        # 如果有图像，添加到payload中
        if images:
            payload["images"] = images
        
        retry_count = 0
        while retry_count < retry_limit: 
            try:
                t1 = time.time()
                print(f"********* start VLM call {self.model} *********")
                
                # 发送请求到Ollama API
                response = requests.post(self.api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("response", "")
                    
                    t2 = time.time() - t1
                    print(f"********* end VLM call {self.model}: {t2:.2f} *********")
                    
                    return content
                        
                else:
                    print(f"API request failed with status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as ex:
                print(f"Attempt VLM call {self.model} {retry_count + 1} failed: {ex}")
                time.sleep(300)
                retry_count += 1
        
        return "Failed to generate completion after multiple attempts."


def save_data_to_json(json_data, base_path):
    """保存JSON数据到文件"""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    try:
        with open(base_path, "r") as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # append
    existing_data.append(json_data)

    # write
    with open(base_path, "w") as f:
        json.dump(existing_data, f, indent=4)
    
    print("save json data to path:", base_path)


if __name__ == "__main__":
    # 测试代码
    model = "qwen2.5vl:32b"
    # model = "qwen3:32b"
    llmapi = VLMAPI(model)
    
    # 测试1: 纯文本请求
    print("=" * 50)
    print("测试1: 纯文本请求")
    print("=" * 50)
    prompt_config = PROMPT_CONFIG.get("vlm_call", {}).get("general", {})
    systext = prompt_config.get("systext", "You are a helpful assistant.")
    usertext = "Hello, can you introduce yourself?"
    
    response = llmapi.vlm_request(systext, usertext)
    print("Response:", response)
    
    # 测试2: 单张图片请求
    print("\n" + "=" * 50)
    print("测试2: 单张图片请求")
    print("=" * 50)
    prompt_config = PROMPT_CONFIG.get("vlm_call", {}).get("image_analysis", {})
    systext = prompt_config.get("systext", "You are a helpful assistant that can analyze images.")
    usertext = prompt_config.get("usertext", "请描述这张图片中你看到了什么？")
    image_path1 = os.path.join(get_project_root(), "data/test.png")
    
    response = llmapi.vlm_request(systext, usertext, image_path1=image_path1)
    print("Response:", response)