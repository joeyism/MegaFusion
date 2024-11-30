import re
import os
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import time

def filter_emoji(desstr, restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=3, help="specify the gpu to load the model.")
    parser.add_argument("--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def inference():
    
    image_path = './samples/SDM-MegaFusion/'
    caption_path = './caption/SDM-MegaFusion/'
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    args = parse_args()
    cfg = Config(args)
    device = 'cuda:{}'.format(args.gpu_id)

    command = "[vqa] Provide a detailed and precise English description of the image displayed in 30 words."
    
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model = model.eval()
    chat = Chat(model, vis_processor, device=device)
    temperature = 0.6
    
    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )
    pattern = r'The image depicts '
    folders = sorted(os.listdir(os.path.join(image_path)))
    
    print(folders)
    for i, folder in enumerate(folders):
        if i==1:
            time.sleep(100)
        caption_folder = os.path.join(caption_path, folder)
        
        if not os.path.exists(caption_folder):
            os.mkdir(caption_folder)
        
        images = sorted(os.listdir(image_path))
        
        for image in tqdm(images):
            print(i)
            temp_path = os.path.join(image_path, image)
            # text_save_path = os.path.join(text_folder, image[:-4] + '.txt')
            print(image)
            caption_save_path = os.path.join(caption_folder, image[:-4] + '.txt')
            print(caption_save_path)
            if not os.path.exists(caption_save_path):
                # with open(text_save_path, 'r') as f:
                #     text = f.read()

                chat_state = CONV_VISION.copy()
                img_list = []
                llm_message = chat.upload_img(temp_path, chat_state, img_list)
                user_message = command
                chat.ask(user_message, chat_state)
                if len(img_list) > 0:
                    if not isinstance(img_list[0], torch.Tensor):
                        chat.encode_img(img_list)
                llm_message = chat.answer(conv=chat_state, img_list=img_list, temperature=temperature, max_new_tokens=196, max_length=800)[0]
                
                caption = llm_message.replace(pattern, "")
                caption = filter_emoji(caption)
                caption = caption.replace("\n", " ")
                
                
                # print(llm_message)
                print(caption)
                with open(caption_save_path, 'w') as f:
                    f.write(caption)
                    
            else:
                print("skip " + caption_save_path)


if __name__ == '__main__':
    inference()