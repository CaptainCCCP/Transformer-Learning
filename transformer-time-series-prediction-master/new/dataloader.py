import torch
import pandas as pd
import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class airDataset(Dataset):
    def __init__(self, filepath=r"C:\\Users\\MOIGE\\Desktop\\实习\\air+quality\\AirQualityUCI.csv"):
        # csv文件中没有sheet表单的说法
        print(f"reading {filepath}")  # 打印日志

        df = pd.read_csv(
            filepath,
            header=0,
            sep=';',
            encoding='utf-8',
            usecols=['Date', 'Time', 'CO(GT)', "PT08.S1(CO)", 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
            names=['Date', 'Time', 'CO(GT)', "PT08.S1(CO)", 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
            # converters={'Date': lambda x: pd.to_datetime(x), 'Time': lambda x: pd.to_datetime(x)},
            # dtype={'Date': str, 'Time': str, 'CO(GT)':np.float32, 'PT08.S1(CO)':np.int32, 'NMHC(GT)':np.float32, 'C6H6(GT)':np.int32, 'PT08.S2(NMHC)':np.int32, 'NOx(GT)':np.int32, 'PT08.S3(NOx)':np.int32, 'NO2(GT)':np.int32, 'PT08.S4(NO2)':np.int32, 'PT08.S5(O3)':np.int32, 'T':np.float32, 'RH':np.float32, 'AH':np.float32},
            skip_blank_lines=True,
            # parse_dates={'Datetime': ['Date', 'Time']},
            )
        df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
        df['C6H6(GT)'] = df['C6H6(GT)'].str.replace(',', '.').astype(float)
        df['T'] = df['T'].str.replace(',', '.').astype(float)
        df['RH'] = df['RH'].str.replace(',', '.').astype(float)
        df['AH'] = df['AH'].str.replace(',', '.').astype(float)

        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        # 将'Datetime'列转换为Unix时间戳
        df['UnixTime'] = pd.to_numeric(df['Datetime']) / 10 ** 9
        df = df.drop(['Date', 'Time', 'Datetime'], axis=1)

        # df = df.reset_index(drop=True)
        print(f'the shape of dataframe is {df.shape}')
        print(df.head())

        df = df.astype(float)
        feat = df.iloc[:, :-1].values
        label = df.iloc[:, -1:].values  # 最后一列是标签列，其他特征数据  # 这里的列和行是排除掉header和index_col之后的
        # iloc就像python里面的切片操作,此处得到的是numpy数组
        # print(label.shape)
        # print(feat.shape)
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


air_dataset = airDataset()
num_train = int(air_dataset.__len__() * 0.7)
num_val = air_dataset.__len__() - num_train
# print(num_train) 6549/9357
# print(num_val)   2808/9357

train_data, val_data = random_split(air_dataset, [num_train, num_val])
train_np = train_data.numpy()
val_np = val_data.numpy()

train_dataloder = DataLoader(train_data, batch_size=100, shuffle=False, drop_last=True)
val_dataloder = DataLoader(val_data, batch_size=100, shuffle=False, drop_last=True)

for idx, (batch_x, batch_y) in enumerate(train_dataloder):
    print(f'batch_id:{idx},{batch_x.shape},{batch_y.shape}')
    print(batch_x, batch_y)
