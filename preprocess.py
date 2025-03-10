import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

divider = '-' * 100


def init_logger() -> None:
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


def preprocess_api_list() -> None:
    logger.info('Parsing API list...')

    with open('data/original/api_list.json', 'r') as file:
        api_list = json.load(file)
    api_list = list(api_list.values())

    logger.info(f'Original API count: {len(api_list)}')

    api_list = list(set(api_list))

    logger.info(f'Deduplicated Original API Count: {len(api_list)}')

    with open('data/processed/api_list.json', 'w') as file:
        file.write(json.dumps(api_list, indent=4))

    logger.info('Saved API list: data/processed/api_list.json')


def preprocess_api_index_mapping() -> None:
    logger.info('Parsing API mapping...')

    with open('data/processed/api_list.json', 'r') as file:
        api_list = json.load(file)

    api_index_mapping = {api: i + 1 for i, api in enumerate(api_list)}

    with open('data/processed/api_index_mapping.pkl', 'wb') as file:
        file.write(pickle.dumps(api_index_mapping))

    logger.info('Saved API mapping: data/processed/api_index_mapping.pkl')


def preprocess_app_api_list_label() -> None:
    logger.info('Loading API mapping......')
    with open('data/processed/api_index_mapping.pkl', 'rb') as file:
        api_index_mapping = pickle.load(file)

    logger.info('Parsing train data......')
    data = []
    with open('data/original/apis_with_tags_for_train.txt', 'r') as file:
        # with open('data/original/api_sequence_label.txt', 'r') as file:
        for line in file:
            line = line.strip()[1:-1]

            label = int(line[-1])

            line = line[:-3]

            api_sequence = json.loads(line)
            api_sequence = [api_index_mapping[api] for api in api_sequence]
            data.append((api_sequence, label))

    logger.info(f'Total train data: {len(data)}')

    with open('data/processed/train_data.pkl', 'wb') as file:
        file.write(pickle.dumps(data))

    logger.info('Saved API mapping: data/processed/train_data.pkl')


def preprocess_train_data(data_path: str, output_path: str) -> None:
    def padding(df: pd.DataFrame, column_name: str):
        max_len = max(df[column_name].apply(len))
        df[column_name] = df[column_name].apply(lambda x: x + [0] * (max_len - len(x)))
        return max_len

    def padding_sublist(list_with_sub: list[list]):
        max_len = 0
        for sub_list in list_with_sub:
            max_len = max(max_len, len(sub_list))

        for sub_list in list_with_sub:
            for _ in range(max_len - len(sub_list)):
                sub_list.append(0)

        return list_with_sub

    def flatten(two_d_list: list[list]):
        return [item for sublist in two_d_list for item in sublist]

    with open(data_path, 'r') as file:
        train_data = json.load(file)

    for item in train_data:
        df = pd.DataFrame(item['key_api_sequence'], columns=['label', 'api_sequence'])

        grouped_data = df.groupby('label')['api_sequence'].apply(list).to_dict()

        item['normal_key_api_sequence'] = flatten(
            padding_sublist(grouped_data.get(0, []))
        )
        item['abnormal_key_api_sequence'] = flatten(
            padding_sublist(grouped_data.get(1, []))
        )

        del item['key_api_sequence']

    logger.info('Padding......')
    df = pd.DataFrame(train_data)
    padding(df, 'api_sequence')
    padding(df, 'normal_key_api_sequence')
    padding(df, 'abnormal_key_api_sequence')

    logger.info('Loading API mapping......')

    with open('data/processed/api_index_mapping.pkl', 'rb') as file:
        api_index_mapping = pickle.load(file)

    logger.info('Replace api to index......')

    df['api_sequence'] = df['api_sequence'].apply(
        lambda x: [api_index_mapping.get(api, 0) for api in x]
    )
    df['normal_key_api_sequence'] = df['normal_key_api_sequence'].apply(
        lambda x: [api_index_mapping.get(api, 0) for api in x]
    )
    df['abnormal_key_api_sequence'] = df['abnormal_key_api_sequence'].apply(
        lambda x: [api_index_mapping.get(api, 0) for api in x]
    )

    with open(output_path, 'wb') as file:
        file.write(pickle.dumps(df.to_dict(orient='records')))

    logger.info(f'Train data count: {len(df)}')
    logger.info(f'Max api sequence length: {len(df.loc[0]["api_sequence"])}')
    logger.info(
        f'Max normal key api sub sequence length: {len(df.loc[0]["normal_key_api_sequence"])}'
    )
    logger.info(
        f'Max abnormal key api sub sequence length: {len(df.loc[0]["abnormal_key_api_sequence"])}'
    )

    logger.info(f'Saved train data: {output_path}')


def preprocess() -> None:
    logger.info(divider)

    preprocess_api_list()
    logger.info(divider)

    preprocess_api_index_mapping()
    logger.info(divider)

    # preprocess_app_api_list_label()
    # logger.info(divider)

    preprocess_train_data(
        'data/original/train_data_with_key_api_sequence_result_merge.json',
        'data/processed/train_data.pkl',
    )
    logger.info(divider)

    preprocess_train_data(
        'data/original/another_env_train_data_with_key_api_sequence_result.json',
        'data/processed/another_env_train_data.pkl',
    )
    logger.info(divider)


def ensure_dir_exists() -> None:
    Path('data/processed').mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    init_logger()

    logger.info('Data preprocessing...')

    ensure_dir_exists()
    preprocess()
