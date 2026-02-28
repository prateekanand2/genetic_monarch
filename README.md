# Genetic Monarch

## Usage

### Training
To train the model using the provided dataset:
```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --train_path data/train.txt \
    --valid_path data/val.txt \
    --num_latents 128 \
    --epochs 5000 \
    --batch_size 128 \
    --pseudocount 0.005 \
    --log_path logs/hclt_run.log \
    --save_path models/model.jpc
```

### Testing
To test the model:
```bash
CUDA_VISIBLE_DEVICES=2 python test.py \
    --model_path models/model.jpc \
    --test_path data/test.txt \
    --batch_size 512
```
