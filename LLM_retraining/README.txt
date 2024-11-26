Content from Karpathy's github repo NanoGPT

Model -> model.py
Training -> train.py (saves fitted model as .pt file in out-movie folder)
Sampler -> sample.py (writes generated content in a txt file under Data/movies/)
Data -> Data/movies/ (movie plots data)
Config file -> config/train_movies.py
Get data ready to be trained -> Data/movies/prepare.py (creates train.bin by combining dataset_i.txt and samples.txt)

Steps needed (see script.py): 

1. Prepare data
2. Train model -> change device (if gpu available, and compile flag)
3. Compute perplexity on held-out data (val.bin)
4. Generate samples

At start of this, use prepare.ipynb file to get (i) 10 smaller splits as dataset{0,1,...,9} and (ii) test data. Currently test data uses 100 movie plots and each of the other datasets have 50 plots each. Total data has 34886 plots - using a gpu might want to use higher sample size for training and test. 

At iteration t=0: use dataset0.txt -> prepare -> train -> sample -> samples.txt
At iteration t>0: combine (dataset{t-1}.txt and samples.txt) -> prepare -> train -> samples.txt

Note: saved model (ckpt.pt file under out-movies) is a large file for github

Note: Training on gpu: change flags in script

Note: If want larger training/test size -> rerun data/movies/prepare.ipynb notebook with changed sizes