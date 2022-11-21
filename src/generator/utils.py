import os
import json
import hashlib
import pandas as pd
from huggingface_hub import hf_hub_url

## First-level categories: 25 
TOP_LEVEL_CATES = ['Books',
    'Clothing, Shoes & Jewelry',
    'Sports & Outdoors',
    'Kindle Store',
    'Home & Kitchen',
    'Automotive',
    'CDs & Vinyl',
    'Tools & Home Improvement',
    'Toys & Games',
    'Arts, Crafts & Sewing',
    'Electronics',
    'Office Products',
    'Grocery & Gourmet Food',
    'Movies & TV',
    'Industrial & Scientific',
    'Patio, Lawn & Garden',
    'Cell Phones & Accessories',
    'Pet Supplies',
    'Video Games',
    'Musical Instruments',
    'Appliances',
    'Software',
    'Collectibles & Fine Art',
    'Gift Cards',
    'Home & Business Services'] 

def download_model_file(model_dir, repo="facebook/opt-30b"):

    for i in range(1,8):
        url = hf_hub_url(repo_id=repo, filename="pytorch_model-0000{}-of-00007.bin".format(str(i)))
        command = "wget -P {} {}".format(model_dir, url)
        print(command)  
        os.system(command)

def generate_key(metainfo):
   
    if "asin_b" in metainfo:
        text = "{}-{}-{}".format(metainfo["asin_a"], metainfo["asin_b"], metainfo["rel"])
    else:
        text = "{}-{}".format(metainfo["asin_a"], metainfo["rel"])
        
    return text, str(hashlib.md5(text.encode('utf-8')).hexdigest())

  
def get_item_metainfo(metafile):

    product_meta = dict()

    if ".csv" in metafile:
        meta = pd.read_csv(metafile, index_col="asin")
        meta_data = meta.drop_duplicates()
        product_meta = meta_data.to_dict("index")
        
    elif ".txt" in metafile:
        infile = open(metafile)
        while True:
            line = infile.readline()
            if not line:
                break
        
            # line: asin \t title \t category
            # category: clothing | women | shoes
            info = line.strip().split("\t")
        
            asin = info[0]
            if asin not in product_meta:
                product_meta[asin] = dict()
                product_meta[asin]["title"] = info[1]
                product_meta[asin]["cate"] = info[2:]
                product_meta[asin]["top_cate"] = " ".join(info[2:]).split("|")[0]
            else:
                continue

    return product_meta
