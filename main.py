from utils import *
from models import *

# preprocessor = StockPreprocessor()
# preprocessor.parse_data()
# print(preprocessor.get_data_arr('A.csv').shape)
# x = preprocessor.parse_data()
# y = preprocessor.normalize(x)
# preprocessor.get_data_arr('A.csv')

trainer = Trainer()
trainer.train(1000)

horizon = int(5) # pred days after last ex in sequence

trainer.evaluate(horizon=horizon, stock_idx=0)