Project Structure:

SEMANTIC-DRIFT-PROJECT/
│
├── data/
│   ├── raw/
│   │   ├── news/
│   │   │   ├── combined0.txt
│   │   │   ├── combined1.txt
│   │   │   ├── combined2.txt
│   │   │   ├── combined3.txt
│   │   │   └── combined4.txt
│   │   └── social/
│   │       ├── all_hindi_comments_doc_wise.txt
│   │       └── all_subtitles_doc_boundary.txt
│   ├── processed/
│   │   ├── news_clean.txt        
│   │   └── social_clean.txt      
│   ├── ldt/
│   │   └── hindi_ldt.csv         ❌ MISSING — needs to be downloaded
│   └── resources/
│       └── hindi_stopwords.txt
│
├── embeddings/
│   ├── news_fasttext_skipgram.bin    
│   ├── news_fasttext_skipgram.vec    
│   ├── social_fasttext_skipgram.bin  
│   └── social_fasttext_skipgram.vec  
│
├── drift/
│   ├── drift_scores.csv              ✅ 32,820 words with drift scores
│   └── rotation_matrix_R.npy         ✅ 300x300 rotation matrix
│
├── results/                          ❌ EMPTY — pending analysis
│   └── plots/
│
├── models/                           ❌ EMPTY
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_news.py         
│   │   ├── clean_social.py       
│   │   └── inspect_corpus.py     
│   ├── training/
│   │   └── train_embeddings.py   
│   ├── alignment/
│   │   └── align_embeddings.py   
│   ├── analysis/
│   │   ├── merge_ldt.py          ❌ pending LDT data
│   │   └── statistical_model.py  ❌ pending LDT data
│   └── visualization/
│       └── plot_results.py       ❌ pending
│
├── venv/
├── requirements.txt              
└── README.md