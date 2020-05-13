## Gender prediction from Facebook statuses using character-level language models 

In this project, we implemented a character-based CNN as well as an LSTM language model as feature extractors on a collection of someone's Facebook statuses to predict their gender. We have also applied character base Kernel SVM and Ada Boost CNN for the sake of comparison. 

### Data 

The dataset includes two part. The first one is MNIST dataset and the second one includes 30 million status updates from 3.5 million users which is collected from the mypersonality app, a third-party facebook quiz. 

### Code
The code is divided to five categories:

1. The [Preprocessing](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/Preprocessing) file contains the code to preprocess the raw data. 
2. The [Naive_Bayes](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/Naive_Bayes) and [SVM](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/SVM) files contain the naive bayes and SVM as baseline methods for the prediction.  
3. The [CNN](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/CNN) file contains the character level CNN model. 
4. The [LSTM](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/LSTM) and [GRU](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/GRU) Files contein character level RNN models. 
5. The [CNN_Boosting](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/tree/master/CNN_Boosting) file contains Ada Boosted CNN method for the gender prediction. 

More details of the project can be found on [Description.pdf](https://github.com/MehradSm/Char-Level-Models-to-Predict-Gender/blob/master/Description.pdf) pdf file.

This project have done by Andre Cutler, Ali Siahkamari and Me, supervised by Prof. Brian Kulis.



