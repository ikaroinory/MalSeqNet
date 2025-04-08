import argparse
import json
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from APIDataset import APIDataset
from models import Classifier, TransformerModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--key_subsequence', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model_name', type=str)
    return parser.parse_args()


args = parse_args()


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_api_count() -> int:
    with open('data/processed/api_list.json', 'r') as f:
        api_list = json.load(f)
    return len(api_list)


def get_data_loader(batch_size: int, test_size: float, seed: int) -> tuple[DataLoader, DataLoader]:
    with open('data/processed/train_data.pkl', 'rb') as file:
        df_train_data = pickle.load(file)
        df_train_data = pd.DataFrame(
            df_train_data,
            columns=['api_sequence', 'normal_key_api_sequence', 'abnormal_key_api_sequence', 'label']
        )

    x = np.array(df_train_data['api_sequence'].tolist())
    normal_key_api_sequence = np.array(df_train_data['normal_key_api_sequence'].tolist())
    abnormal_key_api_sequence = np.array(df_train_data['abnormal_key_api_sequence'].tolist())
    y = np.array(df_train_data['label'].tolist())

    (
        x_train,
        x_test,
        normal_key_api_sequence_train,
        normal_key_api_sequence_test,
        abnormal_key_api_sequence_train,
        abnormal_key_api_sequence_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        normal_key_api_sequence,
        abnormal_key_api_sequence,
        y,
        test_size=test_size,
        random_state=seed,
    )

    train_dataset = APIDataset(x_train, normal_key_api_sequence_train, abnormal_key_api_sequence_train, y_train)
    test_dataset = APIDataset(x_test, normal_key_api_sequence_test, abnormal_key_api_sequence_test, y_test)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def get_another_env_data_loader(batch_size: int) -> DataLoader:
    with open('data/processed/another_env_train_data.pkl', 'rb') as file:
        df_train_data = pickle.load(file)
        df_train_data = pd.DataFrame(
            df_train_data,
            columns=['api_sequence', 'normal_key_api_sequence', 'abnormal_key_api_sequence', 'label']
        )

    x = np.array(df_train_data['api_sequence'].tolist())
    normal_key_api_sequence = np.array(df_train_data['normal_key_api_sequence'].tolist())
    abnormal_key_api_sequence = np.array(df_train_data['abnormal_key_api_sequence'].tolist())
    y = np.array(df_train_data['label'].tolist())

    dataset = APIDataset(x, normal_key_api_sequence, abnormal_key_api_sequence, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def train_epoch(iterator: DataLoader, model: Module, loss_fn: Module, optimizer, device: str) -> float:
    model.to(device)

    model.train()
    total_loss = 0
    for x, normal_key_api_sequence, abnormal_key_api_sequence, y in tqdm(iterator):
        x = x.to(device)
        normal_key_api_sequence = normal_key_api_sequence.to(device)
        abnormal_key_api_sequence = abnormal_key_api_sequence.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if isinstance(model, Classifier):
            output = model(x, normal_key_api_sequence, abnormal_key_api_sequence)
        else:
            output = model(x)

        loss = loss_fn(output.squeeze(), y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(iterator)


def evaluate_epoch(iterator: DataLoader, model: Module, loss_fn: Module, device: str) -> tuple[float, float]:
    model.to(device)

    model.eval()

    total_loss = 0
    accuracy = 0

    with torch.no_grad():
        for x, normal_key_api_sequence, abnormal_key_api_sequence, y in tqdm(iterator):
            x = x.to(device)
            normal_key_api_sequence = normal_key_api_sequence.to(device)
            abnormal_key_api_sequence = abnormal_key_api_sequence.to(device)
            y = y.to(device)

            if isinstance(model, Classifier):
                output = model(x, normal_key_api_sequence, abnormal_key_api_sequence)
            else:
                output = model(x)

            loss = loss_fn(output.squeeze(), y)

            total_loss += loss.item()
            # accuracy += (torch.sum(y_hat == y).item() / output.shape[0])
            accuracy += (torch.sum((output >= 0.5).squeeze() == y).item() / output.shape[0])

    return total_loss / len(iterator), accuracy / len(iterator)


def train(train_loader: DataLoader, test_loader: DataLoader, model: Module, loss_fn: Module, optimizer, epochs: int, device: str) -> None:
    best_epoch = -1
    best_epoch_loss = float('inf')
    best_epoch_accuracy = 0
    no_improvement_counter = 0

    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)

        logger.info('Evaluate.....')
        evaluate_loss, accuracy = evaluate_epoch(test_loader, model, loss_fn, device)

        logger.info(f'Epoch: {epoch + 1}')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Evaluate Loss: {evaluate_loss:.4f}')
        logger.info(f'Accuracy: {accuracy * 100:.2f}%')

        if evaluate_loss < best_epoch_loss:
            best_epoch = epoch + 1
            best_epoch_loss = evaluate_loss
            best_epoch_accuracy = accuracy
            torch.save(model.state_dict(), f'saves/model_{epoch + 1}.pth')
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= args.early_stop:
            logger.info('Early stop.')
            break

    logger.info(f'Best epoch: {best_epoch}')
    logger.info(f'Loss: {best_epoch_loss:.4f}')
    logger.info(f'Accuracy: {best_epoch_accuracy * 100:.2f}%')


def evaluate(test_loader1: DataLoader, test_loader2: DataLoader, model: Module, loss_fn: Module, device: str) -> None:
    _, accuracy = evaluate_epoch(test_loader1, model, loss_fn, device)
    logger.info(f'Accuracy in Android 7.1: {accuracy * 100:.2f}%')

    _, accuracy = evaluate_epoch(test_loader2, model, loss_fn, device)
    logger.info(f'Accuracy in Android 9: {accuracy * 100:.2f}%')


def main():
    logger.info('Loading data......')

    train_loader, test_loader = get_data_loader(
        batch_size=args.batch_size, test_size=args.test_size, seed=args.seed
    )

    if args.key_subsequence:
        model = Classifier(
            input_dim=get_api_count() + 1,
            embedding_dim=args.embedding_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            x_sequence_max_len=1000,
            normal_sequence_max_len=5000,
            abnormal_sequence_max_len=3000,
            dropout=args.dropout,
        )
    else:
        model = TransformerModel(
            input_dim=get_api_count() + 1,
            output_dim=1,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            max_len=5000,
            dropout=args.dropout,
        )

    if args.evaluate:
        model.load_state_dict(torch.load(f'{args.model_name}'))

    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if not args.evaluate:
        logger.info('Train.....')
        train(train_loader, test_loader, model, loss_fn, optimizer, args.epochs, 'cuda')
    else:
        logger.info('Evaluate.....')
        evaluate(test_loader, get_another_env_data_loader(args.batch_size), model, loss_fn, 'cuda')


if __name__ == '__main__':
    logger.remove()
    logger.add(
        'train_log.log',
        format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}',
        mode='w'
    )
    logger.add(
        sys.stdout,
        format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}'
    )

    set_seed(args.seed)

    main()
