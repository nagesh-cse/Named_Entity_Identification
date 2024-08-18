Run the commands from code directory :-

1. To train use command 
        `python ner.py <kernel for svm> <window for pos-tags> train`
    Here 'window for pos-tags' represent the number of next and previous POS tags to look as features 

2. For testing using pre-trained model
        `streamlit run ner.py rbf 2 test`
        or
        `streamlit run ner.py linear 2 test`
        or
        `streamlit run ner.py poly 2 test`