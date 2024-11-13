import json
from tempfile import _TemporaryFileWrapper

import gradio as gr
import requests


def ask_api(api_host: str, pdf_url: str, pdf_file: _TemporaryFileWrapper, question: str, openai_key: str) -> str:
    # 验证 API 主机地址
    if not api_host.startswith('http'):
        return '[ERROR]: Invalid API Host'

    # 检查 URL 和 PDF 文件的输入情况
    if not pdf_url.strip() and not pdf_file:
        return '[ERROR]: Both URL and PDF are empty. Provide at least one.'

    if pdf_url.strip() and pdf_file:
        return '[ERROR]: Both URL and PDF are provided. Please provide only one (either URL or PDF).'

    # 验证问题字段是否为空
    if not question.strip():
        return '[ERROR]: Question field is empty'

    # 构建请求数据
    request_data = {
        'question': question,
        'envs': {
            'OPENAI_API_KEY': openai_key,
        },
    }

    # 根据输入是 URL 还是文件，发送请求
    if pdf_url.strip():
        response = requests.post(f'{api_host}/ask_url', json={'url': pdf_url, **request_data})
    else:
        with open(pdf_file.name, 'rb') as f:
            response = requests.post(f'{api_host}/ask_file', params={'input_data': json.dumps(request_data)}, files={'file': f})

    # 检查请求是否成功
    if response.status_code != 200:
        raise ValueError(f'[ERROR]: {response.text}')

    return response.json().get('result', '[ERROR]: No result returned from the API')


def main():
    # Gradio UI 标题和描述
    title = 'PDF GPT'
    description = (
        """ PDF GPT 允许你通过 Universal Sentence Encoder 和 Open AI 与 PDF 文件对话。
        相比其他工具，它能够提供更少幻觉的回答，因为嵌入技术优于 OpenAI。
        返回的回答甚至可以在方括号([])中引用信息所在的页码，增强回答的可信度，帮助快速找到相关信息。
        """
    )

    # 构建 Gradio 应用界面
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><h1>{title}</h1></center>')
        gr.Markdown(description)

        with gr.Row():
            with gr.Group():
                # 输入框：API Host
                api_host = gr.Textbox(
                    label='输入 API 主机地址',
                    value='http://localhost:8080',
                    placeholder='http://localhost:8080',
                )
                # 提示用户获取 OpenAI API Key
                gr.Markdown(
                    '<p style="text-align:center">在 <a href="https://platform.openai.com/account/api-keys">这里</a> 获取你的 OpenAI API Key</p>'
                )
                # 输入框：OpenAI API Key
                openai_key = gr.Textbox(label='输入 OpenAI API Key', type='password')
                # 输入框：PDF URL
                pdf_url = gr.Textbox(label='输入 PDF URL')
                # 显示选择 URL 或文件的提示
                gr.Markdown("<center><h4>或</h4></center>")
                # 文件上传：PDF 文件
                pdf_file = gr.File(label='上传 PDF/研究论文/书籍', file_types=['.pdf'])
                # 输入框：提问
                question = gr.Textbox(label='输入你的问题')
                # 提交按钮
                submit_btn = gr.Button(value='提交')
                submit_btn.style(full_width=True)

            with gr.Group():
                # 显示回答
                answer = gr.Textbox(label='你的问题的回答是：')

            # 点击按钮后执行的函数
            submit_btn.click(
                ask_api,
                inputs=[api_host, pdf_url, pdf_file, question, openai_key],
                outputs=[answer],
            )

    # 设置应用服务器参数
    demo.app.server.timeout = 60000  # 设置服务器最大返回时间
    demo.launch(server_port=7860, enable_queue=True)  # 启动应用，`enable_queue=True` 确保多用户请求有效


if __name__ == '__main__':
    main()
