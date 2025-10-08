import kagglehub
import shutil
import os

os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), "data")

data_directory = os.path.join(os.getcwd(), "data")

if not os.path.isdir(data_directory):
    os.mkdir(data_directory)

path = kagglehub.dataset_download("paultimothymooney/stock-market-data")
print("Path to dataset files:", path)

sp500_path = os.path.join(os.getcwd(), "data\\datasets\\paultimothymooney\\stock-market-data\\versions\\74\\stock_market_data\\sp500\\csv")

print("Moving necessary files to data directory...")

for item in os.listdir(sp500_path):
    item_path = os.path.join(sp500_path, item)
    shutil.move(item_path, data_directory)
for file in os.listdir(data_directory):
    if os.path.isdir(os.path.join(data_directory, file)):
        shutil.rmtree(os.path.join(data_directory, file))