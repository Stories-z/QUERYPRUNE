import os
import time 
while(True):
	os.system("python train_search.py --gpu 1 --load_db Best_model.pth --batch_size 3 --layers 4")
	time.sleep(60)