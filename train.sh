clear
# python main.py --batch_size=32 --dropout=0.1 --d_model=512 --nhead=8 --num_layers=4 --dim_feedforward=512 --test_size=0.2 --lr=0.0001 --epochs=1000 --seed=42 --early_stop=20 --embedding_dim=256 --key_subsequence --evaluate --model_name=saves/model_58.pth

python main.py --batch_size=32 --dropout=0.1 --d_model=256 --nhead=4 --num_layers=3 --dim_feedforward=256 --test_size=0.4 --lr=0.0003 --epochs=1000 --seed=42 --early_stop=10 --evaluate --model_name=saves/model_28.pth
