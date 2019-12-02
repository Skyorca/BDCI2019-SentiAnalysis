For BDCI-2019 Internet news Sentiment Analysis Task, we implements several baselines here

Deep models are implemented with TF 2.0

1. sentiment dict
2. majority voting with TF-IDF+SVM/RandomForest/MultinormalNB and XGB
3. MLP
4. textCNN
5. 2-BiLSTM
6. 2-BiLSTM+textCNN
7. 2-BiLSTM+Attention
8. DCNN(not implemented yet)

deep models are parallelly run with GPU

to be updated...