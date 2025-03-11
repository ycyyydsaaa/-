import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset(excel_path, test_size=0.2, random_state=42, disease_cols=None):
    """
    根据病例ID分割数据，防止同一病例同时出现在训练与测试集中

    Args:
        excel_path (str): Excel 文件路径，需包含 'paired_image' 或 'id' 列
        test_size (float): 测试集比例
        random_state (int): 随机种子
        disease_cols (list): 疾病标签列（用于分层抽样）
    Returns:
        tuple: (train_path, test_path) 训练集和测试集的 Excel 文件路径
    """
    df = pd.read_excel(excel_path)
    if 'paired_image' not in df.columns and 'id' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'paired_image' 或 'id' 列，用于构造图像文件名。")

    if 'paired_image' in df.columns:
        df = df.dropna(subset=['paired_image'])
        df['case_id'] = df['paired_image'].astype(str).str.split('_').str[0]
    else:
        df = df.dropna(subset=['id'])
        df['case_id'] = df['id'].astype(str)

    case_ids = df['case_id'].unique()

    # 分层抽样（如果指定疾病列）
    stratify = None
    if disease_cols:
        stratify = df[disease_cols].idxmax(axis=1)

    # 分割病例 ID
    train_ids, test_ids = train_test_split(
        case_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # 根据 case_id 筛选数据
    train_df = df[df['case_id'].isin(train_ids)]
    test_df = df[df['case_id'].isin(test_ids)]

    # 保存分割后的 Excel 文件
    base_path = os.path.splitext(excel_path)[0]
    train_path = f"{base_path}_train.xlsx"
    test_path = f"{base_path}_test.xlsx"
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
    if disease_cols:
        print("\n训练集疾病分布:")
        print(train_df[disease_cols].mean())
        print("\n测试集疾病分布:")
        print(test_df[disease_cols].mean())

    return train_path, test_path

