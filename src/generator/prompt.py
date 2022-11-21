import sys
from typing import Optional
from dataclasses import dataclass, field

PREFIX_PROMPT_DICT = {
    "open": ["because", "because"],
    "relatedTo": ["because they both are related to", "because the product is related to"],
    "isA":  ["because they both are a type of", "because the product is a type of"],
    "partOf": ["because they both are a part of", "because the product is a part of"],
    "madeOf": ["because they both are made of", "because the product is made of"],
    "similarTo": ["because they both are similar to", "because the product is similar to"],
    "createdBy": ["because they are created by", "because the product is created by"],
    "hasA": ["because they both have", "because the product has"],
    "propertyOf": ["because they both have a property of", "because the product has a property of"],
    "distinctFrom": ["because they are distinct from", "because the product is distinct from"],
    "usedFor": ["because they are both used for", "because the product is used for"],
    "can": ["because they could both", "because the product could"],
    "capableOf":["because they both are capable of", "because the product is capable of"],
    "definedAs": ["because they both are defined as", "because the product is defined as"],
    "symbolOf": ["because they both are symbols of", "because the product is symbols of"],
    "mannerOf": ["because they both are a manner of", "because the product is a manner of"],
    "deriveFrom": ["because they are derived from", "because the product is derived from"],
    "effect": ["as a result, the person will", "as a result, the person will"],
    "cause": ["because the person wants to", "because the person wants to"],
    "motivatedBy": ["and buying them was motivated by", "and buying them was motivated by"],
    "causeEffect": ["because the person wants his", "because the person wants his"]
}


class BehaviorTemplate():

    def __init__(self, behavior_type, prompt_type, metainfo, meta_type):

        self.behavior_type = behavior_type
        self.prompt_type = prompt_type
        self.metainfo = metainfo
        self.meta_type = meta_type

        if self.behavior_type == "single_buy":
            if prompt_type == "prefix":
                self.prompt_dict = PREFIX_PROMPT_DICT
                self.predicate = 'He bought a product of "[A]" '
            elif prompt_type == "qa":
                self.prompt_dict = QA_PROMPT_DICT
                self.predicte = 'Task: Please explain the intents why the person bought the two items together.\n\nProduct 1: [A]\n\n'
            self.prompt = self.single_buy_generate()
        
        elif self.behavior_type == "cobuy":
            if prompt_type == "prefix":
                self.prompt_dict = PREFIX_PROMPT_DICT
                self.predicate = 'He bought a product of "[A]" and a product of "[B]" ' 
            elif prompt_type == "qa":
                self.prompt_dict = QA_PROMPT_DICT
                self.predicate = 'Task: Please explain the intents why the person bought the two items together.\n\nProduct 1: [A]\nProduct 2: [B]\n\n'
            self.prompt = self.cobuy_generate()

    def __len__(self):
      
        return sum([len(each["title"].split()) for each in self.metainfo])

    def single_buy_generate(self):

        prompt = list()
        assert len(self.metainfo) ==1
        self.item_a = self.metainfo[0]

        for rel in self.prompt_dict.keys():
            tmp = dict()
            if self.meta_type == "title":
                text = self.predicate.replace('[A]', self.item_a["title"]) + self.prompt_dict[rel][1]
            elif self.meta_type == "concept": 
                text = self.predicate.replace('[A]', self.item_a["category"]) + self.prompt_dict[rel][1]
           
            tmp["asin_a"] = self.item_a["asin"]
            tmp["rel"]  = rel
            tmp["text"] = text
            prompt.append(tmp)  

        return prompt

    def cobuy_generate(self):

        prompt = list()
        assert len(self.metainfo) == 2
        self.item_a, self.item_b = self.metainfo

        for rel in self.prompt_dict.keys():
            tmp = dict()
            if self.meta_type == "title":
                text = self.predicate.replace('[A]', self.item_a["title"]).replace('[B]', self.item_b["title"]) + self.prompt_dict[rel][0]
            elif self.meta_type == "concept":
                text = self.predicate.replace('[A]', self.item_a["category"]).replace('[B]', self.item_b["category"]) + self.prompt_dict[rel][0]

            tmp["asin_a"] = self.item_a["asin"]
            tmp["asin_b"] = self.item_b["asin"]
            tmp["rel"]  = rel
            tmp["text"] = text
            prompt.append(tmp)

        return prompt

#meta = [{'asin': 'B00Q8F63FC', 'title': 'Acer K242HL 24&quot; LED LCD 1080p Full HD Monitor (Mercury Free)', 'category': 'Electronics'}, {'asin': 'B00Q8F63FC', 'title': 'Acer K242HL 24&quot; LED LCD 1080p Full HD Monitor (Mercury Free)', 'category': 'Electronics'}]
#example = BehaviorTemplate("cobuy", "qa", meta, "title")
#print(example.prompt)


