import os 
import sys
import torch
import json
import time
import pickle
import random
import requests
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
     AutoConfig,
     AutoTokenizer,
     AutoModelForSeq2SeqLM,
     AutoModelForCausalLM,
)
from prompt import BehaviorTemplate 
from utils import get_item_metainfo, generate_key 

def set_seed(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

 
class PromptGenerator():
  
    def __init__(self, config):

        self.config = config 
        self.model_type = config.model_type        
        self.model_name = config.model_name_or_path

        self.generation_type = config.generation_type

        if self.generation_type == "inference":

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side='left'

            if "large" in self.model_type:
                ## lood gpt-neox20b: transformers >= 4.21.0
                ## https://github.com/huggingface/transformers/pull/17811
                cur = time.time()
                max_memory = {i: "32GB" for i in range(config.num_gpus)}
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,  
                                                    device_map='auto', 
                                                    max_memory= max_memory, 
                                                    torch_dtype=torch.float16)
                #self.model = load_parallel_model(self.model_name, config.num_gpus)
                print("loading the model takes {:.2f} seconds".format(time.time()-cur))
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                if config.num_gpus > 0:
                    self.model.cuda()
        elif self.generation_type == "api":
            ## support opt175b api generation
            ## place host url as follows
            self.host_url = ""

    def _generation(self, inputs):
        
        prompts = [each["text"] for each in inputs]

        if self.generation_type == "inference": 
            res = self.topk_sampling(prompts)
        else:
            req_obj = {"prompt": prompts, 
                    "max_tokens": self.config.max_generation_length, 
                    "temperature": self.config.temperature}
            res = self.request_api(req_obj)

        return res 
    
    def request_api(self, req_str):

        req_str = json.dumps(req_str).encode()
        headers = {'content-type': 'application/json'}
        res = requests.post(self.host_url, data=req_str, headers=headers)
        res.encoding = "utf-8"

        return res

    def topk_sampling(self, prompt):
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=self.config.max_seq_length).input_ids.to(0)
        
        ## too long tokenzied subtokens might cause cuda oom and 
        ## useless generation
        if input_ids.size()[1] > 1.5 * self.config.max_seq_length: 
            return None

        ##  Potential bugs: whether add attention masks for batch padding
        ##  https://github.com/huggingface/transformers/pull/17444
        outputs = self.model.generate(input_ids=input_ids, 
                            temperature=self.config.temperature,
                            top_p=self.config.topp,
                            pad_token_id=self.tokenizer.eos_token_id,
                            max_length=self.config.max_generation_length, 
                            num_return_sequences=self.config.num_generated_sent, 
                            do_sample=True)
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        res = np.reshape(res, (len(prompt),-1))
        return res  

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    ) 
    parser.add_argument("--output_dir", default=None, type=str, required=True, 
        help="The output dir for generated data",
    )
    parser.add_argument("--model_name_or_path", default=None, type=str,required=True,
        help="Path to pre-trained model or shortcut name selected"
    )
    parser.add_argument("--model_type", default=None, type=str, required=True, help="model type")
    parser.add_argument("--behavior_type", default=None, type=str, required=True, help="behavior type: single buy/co-buy/query")
    parser.add_argument("--prompt_type", default=None, type=str, required=True, help="prompt type: prefix or qa")
    parser.add_argument("--generation_type", default=None, type=str, required=True, help="generation type")
    parser.add_argument("--data_cache_name", default=None, type=str, help="The name of cached data")
    parser.add_argument("--max_seq_length", default=80, type=int, help="max input sequence length")
    parser.add_argument("--max_generation_length", default=100, type=int, help="max generation length")
    parser.add_argument("--temperature", default=1.0, type=float, help="generaton temperature")
    parser.add_argument("--topp", default=0.5, type=float, help="generaton top-p sampling")
    parser.add_argument("--num_generated_sent", default=3, type=int, help="number of generated sentences")
    parser.add_argument("--seed", default=42, type=int, help="seed number")
    parser.add_argument("--batch_size", default=4, type=int, help="generation batch size")
    parser.add_argument("--category", default=None, type=str, help="sampled top categories")
    parser.add_argument("--num_samples", default=None, type=int, help="number of samples for generation")
    parser.add_argument("--run_index", default=-1, type=int, help="the i-th share of data to be run")

    args = parser.parse_args()
    args.num_gpus = torch.cuda.device_count()
    set_seed(args)

    prompt_generator = PromptGenerator(args)

    category = "-".join(args.category.split())
    cached_sampled_data = os.path.join(args.data_dir, "cached_{}_{}_{}.pkl".format(args.behavior_type,
                                                      category, str(args.num_samples))) 
    if os.path.exists(cached_sampled_data):
        print("Loading samples from cached file ", cached_sampled_data)
        examples = pickle.load(open(cached_sampled_data, "rb"))
        print(len(examples))
        print(examples[:3])    
    else:
        print("Sampling data from source metadata") 
        metainfo = get_item_metainfo(args.data_dir + "/amazon_title_category.txt")
       
        all_examples = [] 
        source_file = open(args.data_dir + "/amazon_cobuy_graph.txt") 
        
        while True:
            line = source_file.readline()
            if not line:
                break

            item_a, item_b = line.strip().split("\t") 
            if item_a not in metainfo or item_b not in metainfo:
                continue
            if metainfo[item_a]["top_cate"] == args.category or metainfo[item_b]["top_cate"] == args.category:
                all_examples.append((item_a, item_b)) 

        source_file.close()
        print("Starting to sample {} examples from {} ".format(args.num_samples, len(all_examples))) 
        sampled_index = random.sample(range(len(all_examples)),args.num_samples)
        examples = [] 
        for idx in sampled_index:
            item_a, item_b = all_examples[idx]

            meta_a = {"asin": item_a, "title": metainfo[item_a]["title"], "category": metainfo[item_a]["top_cate"]}  
            meta_b = {"asin": item_b, "title": metainfo[item_b]["title"], "category": metainfo[item_b]["top_cate"]}  
            examples.append([meta_a, meta_b])

        print(examples[:3])
        pickle.dump(examples, open(cached_sampled_data, "wb"))
        print("Finishing the data sampling !") 

    if args.run_index < 0:
        gen_cache_fn = args.output_dir + "/cache/{}_{}_{}_cached_gen_all.json".format(category, args.behavior_type, args.prompt_type)
    else:   
        gen_cache_fn = args.output_dir + "/cache/{}_{}_{}_cached_gen_{}.json".format(category, args.behavior_type, args.prompt_type, str(args.run_index))

    key_cache_fn = args.output_dir +  "/cache/{}_key.json".format(category)

    print(f"Loading cache from {gen_cache_fn} and {key_cache_fn}.")

    key_cache = json.load(open(key_cache_fn, 'r', encoding='utf-8')) if os.path.isfile(key_cache_fn) else {}
    gen_cache = open(gen_cache_fn, 'w', encoding='utf-8')

    length = 0
    count = 0
    inputs = []

    long_c =0
    total_run = 10000
    if args.run_index > -1:
        start = args.run_index * total_run
        end = min((args.run_index+1) * total_run, len(examples))
        examples = examples[start:end]
  
    for sample in examples:

        prompt_templates = BehaviorTemplate(args.behavior_type, args.prompt_type, sample, "title")
        ## remove item pairs with long noisy titles
        if len(prompt_templates) > 50:
            long_c+=1
            continue
        for sent in prompt_templates.prompt:
            length += len(sent["text"].split())
            if "asin_b" in sent:
                msg = "{}-{}-{}".format(sent["asin_a"], sent["asin_b"], sent["rel"])
            else:
                 msg = "{}-{}".format(sent["asin_a"], sent["rel"])
            if msg in key_cache:
                continue   
            else: 
                inputs.append(sent)
        count += len(prompt_templates.prompt)

    if len(inputs) % args.batch_size ==0:
        batch_num = len(inputs) // args.batch_size
    else:
        batch_num = len(inputs) // args.batch_size + 1
    
    for i in tqdm(range(batch_num)):
        start_idx = i *  args.batch_size
        end_idx = min((i+1) * args.batch_size, len(inputs))
        batch_inputs = inputs[start_idx:end_idx]

        batch_outputs = prompt_generator._generation(batch_inputs)

        if batch_outputs is None:
            continue
        else:
            assert len(batch_inputs) == len(batch_outputs)
            for j in range(len(batch_inputs)):
                msg, key = generate_key(batch_inputs[j])
                key_cache[msg] = key
                json.dump({"text": msg , "key": key, "output": list(batch_outputs[j])}, gen_cache) 
                gen_cache.write("\n")

    json.dump(key_cache, open(key_cache_fn, 'w', encoding='utf-8'))
    print(float(length)/count)
    print(long_c)
    print(len(examples))
  
if __name__== "__main__":
    main()

