import subprocess
import torch

perplexity = []

for it in range(5):
    print(f'Starting iteration {it}')
    
    # run prepare file
    print('----(i) PREPARING TRAIN-----')
    name = '--t=' + str(it)
    res = subprocess.run(['python', 'data/movies/prepare.py', name], capture_output=True)
    print(res.stdout)

    # train the actual model
    print('----(ii) TRAINING MODEL-----')
    res = subprocess.run(['python', 'train.py', 'config/train_movies.py', '--device=cpu', '--compile=False', '--eval_iters=20', '--log_interval=1', '--block_size=64', '--batch_size=12', '--n_layer=4', '--n_head=8', '--n_embd=256', '--max_iters=500', '--lr_decay_iters=500', '--dropout=0.0'], capture_output=False)
    print('--- DONE ---')

    # extract the perplexity
    fit = torch.load('out-movie/ckpt.pt')
    perplexity.append(torch.exp(fit['best_val_loss']))
    print(f'last perplexity score on val data = {perplexity[-1]}')
    
    # generate samples
    print('----(iii) GENERATING SAMPLES-----')
    res = subprocess.run(['python', 'sample.py', '--out_dir=out-movie', '--device=cpu', '--num_samples=100', '--max_new_tokens=200'], capture_output=False)

print(perplexity)


