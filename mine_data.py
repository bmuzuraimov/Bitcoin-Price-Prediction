import wget
import pandas as pd
import zipfile
from datetime import timedelta, date
import time
import glob


class DMiner:
	def __init__(self, crypto_name, interval):
		self.existing_files = glob.glob('./docs/zip_files/*')
		self.start_dt = date(2019, 12, 31)
		self.end_dt = date(2022, 11, 16)
		self.crypto_name = crypto_name
		self.interval = interval

	def __daterange(self, date1, date2):
	    for n in range(int ((date2 - date1).days)+1):
	        yield date1 + timedelta(n)

	def download(self):
		for dt in self.__daterange(self.start_dt, self.end_dt):
			confirm = input('You are about to download huge amount files, are you sure? Y/N')
			if(confirm == 'Y'):
				date = dt.strftime("%Y-%m-%d")
				file_name = '{crypto_name}-{interval}-{date}.zip'.format(crypto_name=crypto_name, interval=interval, date=date)
				url_path = 'https://data.binance.vision/data/futures/um/daily/klines/{crypto_name}/{interval}/'.format(crypto_name=crypto_name, interval=interval)
				source_url = url_path + file_name
				zip_path = './zip_files/' + file_name
				extract_path = './docs/extracted_files'
				if(zip_path not in existing_files):
					try:
						response = wget.download(source_url, zip_path)
						with zipfile.ZipFile(zip_path, 'r') as zip_ref:
						    zip_ref.extractall(extract_path)
					except:
						print('{} path doesn\' exist!'.format(source_url))
					time.sleep(0.05)
				else:
					existing_files.remove(zip_path)
			else:
				print('Action has been canceled!')

	def concat_dataset(self, output_file, dataset_files):
		df_list = []
		for daily_chart in dataset_files:
			df = pd.read_csv(daily_chart, index_col=False, header=None)
			df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
			df.set_index('open_time')
			df_list.append(df)
		result = pd.concat(df_list)
		result = result.loc[:, 'open_time':'close_time']
		result.set_index('open_time', inplace=True)
		result.to_csv('./docs/processed_files/datasets/'+output_file)


