from StockPreprocessor import StockPreprocessor

preprocessor = StockPreprocessor()
train_data, test_data = preprocessor.preprocess_all()

print(train_data, test_data)