import gradio as gr
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from PIL import Image
import base64
import io
from tqdm import tqdm
import time



PHOTO_FOLDER = "photos"
DB_PATH = "photo_chroma_db"


# 初始化 API 多模态模型
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key="",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
    max_tokens=512
)

# 初始化本地嵌入模型
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="bge-small-zh-v1.5"
)

# 初始化 Chroma 向量数据库
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="my_photos",
    embedding_function=embedding_function
)

#  Prompt
PROMPT = """你是一个专业摄影师和图像理解专家，请仔细观察这张照片，用自然流畅的中文完成：

1. 先写一段50-80字像朋友圈一样的生动描述
2. 然后用【标签】列出25-35个最精准的关键词，涵盖人物、地点、时间、活动、物品、情绪、拍摄风格等

示例：
周六下午在西湖断桥边，我和闺蜜穿着旗袍拍了一组复古写真，柳树垂枝、白墙灰瓦、夕阳余晖，氛围绝了！
【标签】闺蜜, 西湖, 断桥, 杭州, 旗袍, 复古写真, 下午, 柳树, 白墙灰瓦, 夕阳, 旅游, 20多岁, 笑容, 汉服元素, 湖边, 蓝天, 幸福...
"""


# ========== 核心函数：单张照片分析 ==========
def analyze_single_photo(input_data):
    if isinstance(input_data, str):
        # 输入是文件路径
        image = Image.open(input_data).convert("RGB")
    else:
        # 输入是PIL Image对象
        image = input_data.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    base64_image = base64.b64encode(buffered.getvalue()).decode()

    message = HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
    ])

    try:
        response = llm.invoke([message])
        result = response.content

        if "【标签】" in result:
            description = result.split("【标签】")[0].strip()
            tags = result.split("【标签】")[1].strip()
        else:
            description = result.strip()
            tags = ""

        full_text = description + " " + tags.replace(",", " ")
        return {
            "path": "single_test" if not isinstance(input_data, str) else input_data,
            "description": description,
            "tags": tags,
            "search_text": full_text
        }
    except Exception as e:
        return {"error": str(e), "path": "single_test" if not isinstance(input_data, str) else input_data}


# ========== 批量处理整个相册 ==========
def batch_process_album():
    if not os.path.exists(PHOTO_FOLDER):
        os.makedirs(PHOTO_FOLDER)
        return "请把照片放入 'photos' 文件夹后再次点击"

    files = [f for f in os.listdir(PHOTO_FOLDER)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

    if not files:
        return "photos 文件夹为空！请放入照片"

    results = []
    for file in tqdm(files, desc="正在智能分析相册"):
        path = os.path.join(PHOTO_FOLDER, file)
        info = analyze_single_photo(path)

        if "error" not in info:
            # 存入向量数据库
            collection.add(
                ids=[os.path.basename(path)],
                documents=[info["search_text"]],
                metadatas=[{
                    "path": info["path"],
                    "description": info["description"],
                    "tags": info["tags"]
                }]
            )
        results.append(info)
        time.sleep(0.5)  # 避免触发API限流

    return f"批量处理完成！共处理 {len(files)} 张照片，已全部存入向量数据库，可开始搜索 →"


# ========== 自然语言搜索 ==========
def search_photos(query, top_k=12):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    images = []
    captions = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        img = Image.open(meta["path"])
        caption = f"{meta['description']}\n标签：{meta['tags']}"
        images.append(img)
        captions.append(caption)

    return images if images else "未找到匹配照片，换个说法试试~"


# ========== Gradio ==========
with gr.Blocks(theme=gr.themes.Soft(), title="智能相册管理与问答系统") as demo:
    gr.Markdown("#智能相册管理与问答系统")
    gr.Markdown("支持批量处理 + 自然语言搜索")

    with gr.Tab("1. 批量处理相册"):
        gr.Markdown(f"把所有照片放入项目根目录的 `{PHOTO_FOLDER}` 文件夹")
        process_btn = gr.Button("开始批量智能分析并建库", variant="primary")
        process_output = gr.Textbox(label="处理进度")
        process_btn.click(batch_process_album, outputs=process_output)

    with gr.Tab("2. 自然语言搜索照片"):
        search_input = gr.Textbox(label="输入你想找的照片",
                                  placeholder="例如：2024年和女朋友在海边的照片、晚上吃的火锅、戴眼镜的自拍、包含小狗在草地上的...",
                                  lines=2)
        search_btn = gr.Button("搜索照片", variant="primary")
        gallery = gr.Gallery(label="搜索结果", columns=4, height="auto")
        search_btn.click(search_photos, inputs=search_input, outputs=gallery)

    # 替换“3. 单张测试”部分的按钮绑定逻辑
    with gr.Tab("3. 单张测试"):
        single_img = gr.Image(type="pil")
        single_btn = gr.Button("分析这张照片")
        single_desc = gr.Textbox(label="描述")
        single_tags = gr.Textbox(label="标签")


        # 定义处理函数，同时返回描述和标签
        def process_single(img):
            if not img:  # 若未上传图片，返回空值
                return "", ""
            result = analyze_single_photo(img)
            # 从结果中提取描述和标签（默认空字符串避免报错）
            return result.get("description", ""), result.get("tags", "")


        # 绑定到两个输出组件
        single_btn.click(
            process_single,
            inputs=single_img,
            outputs=[single_desc, single_tags]
        )

if __name__ == "__main__":
    # 自动创建照片文件夹
    os.makedirs(PHOTO_FOLDER, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)

    demo.launch(share=True, server_port=7860)