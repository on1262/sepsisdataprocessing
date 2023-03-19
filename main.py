


if __name__ == '__main__':
    dataset = MIMICDataset()
    analyzer = Analyzer(dataset)
    # analyzer._detect_adm_data("220224")
    # analyzer.feature_explore()
    analyzer.nearest_cls()
    # analyzer.lstm_cls()
    
    # analyzer.lstm_model()
    # analyzer.nearest_method()