# Copyright (c) 2022 Horizon Robotics. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import torch
import tokenization

#加载字典
tokenizer = tokenization.FullTokenizer('./raw_disf/vocab.txt') 
#加载模型
model = torch.jit.load(f"./output/ckpts/tagging_20240327_114014/checkpoint-epoch0-final/model.safetensors.script")
#标签
label_map = {"O": 0, "，": 1, "。": 2, "？": 3, "！": 4, "、":5, "《":6, "》":7}

label_map = {v:k for k, v in label_map.items()}

'''
pip install torch gradio==3.14.0 
'''

def text_punc(TextIn="请输入一段中文文字"):
    input_tokens = [c for c in TextIn if c not in label_map.values()]
    
    tokens = ['[CLS]'] + input_tokens + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids_tensor = torch.tensor([ids])
    result = model.forward(ids_tensor)[2].tolist()[0][1:-1]
    print(result)
    for i, v in enumerate(result):
        if v != 0:
            input_tokens[i] = input_tokens[i] + label_map[v]
    return ''.join(input_tokens)

def start_server():
    # input
    inputs = [gr.Textbox(),]
    output = gr.Textbox(label="Text Out")


    text = "中文文本标点"

    # description
    description = (
        "This is a text punct demo that supports only Mandarin !"  # noqa
    )

    article = (
        "<p style='text-align: center'>"
        "<a href='' target='_blank'></a>"  # noqa
        "</p>")
    interface = gr.Interface(
        fn=text_punc,
        inputs=inputs,
        outputs=[output],
        title=text,
        description=description,
        article=article,
        theme='huggingface',
    )
    interface.launch()

if __name__ == "__main__":
    start_server()
