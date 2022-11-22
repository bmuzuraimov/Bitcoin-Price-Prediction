# importing pandas library
import pandas as pd
# importing matplotlib library
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import glob
import pandas_ta as pta
from mlayer import MLP
from mine_data import DMiner
import argparse

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

all_datasets_files = sorted(glob.glob('./docs/extracted_files/*.csv')) # from 31.12.2019 to 16.12.2022
training_set_files = all_datasets_files[:350] # from 31.12.2019 to 14.12.2020
parameter_set_files = all_datasets_files[350:700] # from 31.12.2019 to 14.12.2020
test_set_files = all_datasets_files[700:1051] # from 01.12.2021 to 16.11.2022

# Since all dataset is in daily csv files,
# for convenience sake it is better concatinate
# and save in single file

reversal_patterns = ['Double top', 'Double bottom', 'Triple top', 'Triple bottom', 'Head and shoulders', 'Head and shoulders r', 'Rounded top', 'Rounded bottom', 'Spike top', 'Spike bottom']
continuation_patterns = ['Triangle', 'Broadening', 'Diamond', 'Flag', 'Pennant', 'Wedge', 'Rectangle', 'Head and shoulders']

class TrainingDataset():
	def __init__(self, filename, window_size):
		self.filename = filename
		self.path = './docs/processed_files/datasets/'+filename
		self.raw_df_dataset = pd.read_csv(self.path)		
		self.df_dataset = self.raw_df_dataset.copy()
		self.window_size = window_size # last 10 days (10 days * 24 * 60) / 15m
		self.indicators = {
			'indicator_K': None,
			'SMA': None,
		}

	def _update_file(self):
		self.df_dataset.to_csv(self.path)

	def format_dates(self, *args, **kwargs):
		for column in args:
			try:
				self.df_dataset[column] = self.df_dataset[column].apply(lambda x: datetime.fromtimestamp(int(x)/1000))
			except(ValueError):
				print('Column already in date format or invalid dtype is passed')

	def join_indicator(self, *indicators, **kwargs):
		for indicator in indicators:
			if(indicator == None): continue
			if(indicator in self.df_dataset.columns): self.df_dataset.drop(indicator, axis=1, inplace=True)
			self.df_dataset = self.df_dataset.merge(self.indicators[indicator], how='left', on='close_time')
		# if(kwargs['inplace']):
		# 	self._update_file()

	def calc_sma(self):
		self.df_dataset['SMA'] = self.df_dataset['close'].rolling(self.window_size).mean()

	def calc_overbought_oversold_index(self, include_date=False, **kwargs):
		dict_dataset = self.df_dataset.to_dict('records')
		indicator_K_list = []
		start_index = self.window_size-1 if include_date else self.window_size
		index = start_index
		for row_i in dict_dataset[start_index:]:
			high_prices = [row_j['high'] for row_j in dict_dataset[index-start_index:index]]
			low_prices = [row_j['low'] for row_j in dict_dataset[index-start_index:index]]
			high = max(high_prices)
			low = min(low_prices)
			indicator_K = ((row_i['close'] - low) / (high - low)) * 100
			indicator_K_list.append([row_i['close_time'], indicator_K])
			index += 1
		self.indicators['indicator_K'] = pd.DataFrame(data=indicator_K_list, columns=['close_time', 'indicator_K'])

	def calc_rsi(self, include_date=False, **kwargs):
		self.df_dataset['RSI'] = pta.rsi(self.df_dataset['close'], length=self.window_size/32)

	def linear_normalization(self):
		close_prices_dict = self.df_dataset[['close_time', 'close']].copy().to_dict('records')
		volatility_list = []
		# start_index = self.window_size
		start_index = self.window_size
		index = start_index
		for row_i in close_prices_dict[start_index:]:
			close_prices = []
			delta_j = []
			delta_j_pct = []
			delta_j_u = []
			delta_j_pct_u = []
			max_price = 0
			min_price = 0
			for row_j in close_prices_dict[index-start_index:index][::-1]:
				close_prices.append(row_j['close'])
				price_delta = row_i['close'] - row_j['close']
				delta_j.append(price_delta)
				delta_j_u.append(abs(price_delta))
				delta_j_pct.append(price_delta / row_i['close'])
				delta_j_pct_u.append(abs(price_delta) / row_i['close'])
			max_price = max(close_prices)
			min_price = min(close_prices)
			volatility_list.append([row_i['close_time'], row_i['close'], delta_j, delta_j_pct, delta_j_pct_u, max_price, min_price])
			index += 1
		volatility_array = np.array(volatility_list)

		P_A = 2 / (len(close_prices_dict) - self.window_size)
		lambda_j = np.full((1, self.window_size), P_A)
		lambda_array = []
		sum_array = []
		row_sum = []
		upward_arr = volatility_array[0:, 4]
		sum_array = np.array(upward_arr[0])
		for index_j in range(1, len(upward_arr)):
			sum_array = np.add(sum_array, np.array(upward_arr[index_j]))
		row_sum = np.multiply(sum_array, lambda_j)[0]
		input_data = []
		output_data = []
		for index_i in range(len(volatility_array)):
			price = volatility_array[index_i][1]
			delta_j = volatility_array[index_i][2]
			temp_row = []
			for index_j in range(self.window_size):
				xi = delta_j[index_j] / (price * row_sum[index_j])
				temp_row.append(xi)
			di = (price - volatility_array[index_i][6]) / (volatility_array[index_i][5] - volatility_array[index_i][6])
			output_data.append(di)
			input_data.append(temp_row)
		columns = ['x'+str(i) for i in range(self.window_size)]
		input_data_df = pd.DataFrame(data=input_data, columns=columns)
		output_data_df = pd.DataFrame(data=output_data, columns=['target'])
		input_data_df.to_csv('./docs/processed_files/datasets/input_data.csv', index=False)
		output_data_df.to_csv('./docs/processed_files/datasets/output_data.csv', index=False)


	def plot(self):
		fig, axs = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
		axs[0].title.set_text(self.filename)
		axs[1].title.set_text('Overbought & oversold indicator')
		axs[2].title.set_text('RSI')
		axs[0].plot(self.df_dataset['close_time'], self.df_dataset['close'])
		axs[0].plot(self.df_dataset['close_time'], self.df_dataset['SMA'], color = 'orange')
		axs[1].plot(self.df_dataset['close_time'], self.df_dataset['indicator_K'])
		axs[1].axhline(y = 80, color = 'r', linestyle = '-')
		axs[1].axhline(y = 20, color = 'g', linestyle = '-')
		axs[2].plot(self.df_dataset['close_time'], self.df_dataset['RSI'], color = 'orange')
		axs[2].axhline(y = 70, color = 'r', linestyle = '-')
		axs[2].axhline(y = 30, color = 'g', linestyle = '-')
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, required=False)
	args = parser.parse_args()
	args.mode = 'extract_feature'
	window_size = 10
	if(args.mode == 'mine_data'):
  		miner = DMiner('BTCUSDT', '15m')
  		miner.download()
	elif(args.mode == 'process_data'):
  		miner = DMiner('BTCUSDT', '15m')
  		miner.concat_dataset('training_set.csv', training_set_files)
	elif(args.mode == 'extract_feature'):
		training_set = TrainingDataset('training_set.csv', window_size)
		training_set.format_dates('open_time', 'close_time')
		training_set.linear_normalization()
	elif(args.mode == 'train_data'):
		bitcoin_model = MLP('input_data.csv', 'output_data.csv', 0.2, 1000, window_size, 1, 1)
		bitcoin_model.train()
	
