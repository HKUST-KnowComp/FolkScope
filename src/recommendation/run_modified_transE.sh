CUDA_VISIBLE_DEVICES=0 python run_transE.py -f -d Electronics -r 05 
CUDA_VISIBLE_DEVICES=1 python run_transE.py -f -d Electronics -r 09
CUDA_VISIBLE_DEVICES=2 python run_transE.py -f -d Electronics -r 00
CUDA_VISIBLE_DEVICES=3 python run_transE.py -f -d Clothing_Shoes_and_Jewelry -r 05
CUDA_VISIBLE_DEVICES=0 python run_transE.py -f -d Clothing_Shoes_and_Jewelry -r 09
CUDA_VISIBLE_DEVICES=1 python run_transE.py -f -d Clothing_Shoes_and_Jewelry -r 00
CUDA_VISIBLE_DEVICES=2 python run_transE.py -f -d Clothing_Shoes_and_Jewelry -r 05_05
CUDA_VISIBLE_DEVICES=3 python run_transE.py -f -d Clothing_Shoes_and_Jewelry -r 09_09
CUDA_VISIBLE_DEVICES=0 python run_transE.py -f -d Electronics -r 05_05
CUDA_VISIBLE_DEVICES=1 python run_transE.py -f -d Electronics -r 09_09

