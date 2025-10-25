from utils import *

preprocessor = StockPreprocessor()
# preprocessor.parse_data()
# print(preprocessor.get_data_arr('A.csv').shape)
x = preprocessor.parse_data()
y = preprocessor.normalize(x)
# preprocessor.get_data_arr('A.csv')
