import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "/Users/caomi/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
df = pd.read_csv(file_path)

# 列名 classID 应该用于分层划分
train_data, test_data = train_test_split(df, test_size=0.3, stratify=df['classID'], random_state=42)
validation_data, test_data = train_test_split(test_data, test_size=1/3, stratify=test_data['classID'], random_state=42)

# 再次保存划分后的数据集
train_data.to_csv('/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/train_csv.csv', index=False)
validation_data.to_csv('/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/validation_csv.csv', index=False)
test_data.to_csv('/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/test_csv.csv', index=False)

# 返回新的文件路径
train_csv_path = '/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/train_csv.csv'
validation_csv_path = '/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/validation_csv.csv'
test_csv_path = '/Users/caomi/Documents/deep learning/pytorch for Audio + Music Processing/dataset/test_csv.csv'