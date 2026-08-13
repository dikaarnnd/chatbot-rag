chunk_size = 128
testset_topk3
top_k   avg_context_precision   avg_context_recall      
3       0.500                   0.700                   
4       0.475                   0.800                   
5       0.440                   0.800                   
6       0.367                   0.800                   
7       0.329                   0.800                   
8       0.312                   0.800                   
9       0.289                   0.800                   
10      0.260                   0.800
11      0.236                   0.800                   
12      0.225                   0.800                   
13      0.208                   0.800                   
14      0.193                   0.800                   
15      0.180                   0.800

no. pertanyaan tidak terjawab: 2 & 9
-------------------------------------------------------
chunk_size = 256
testset_topk3
top_k   avg_context_precision   avg_context_recall      
3       0.400                   0.800                   
4       0.300                   0.800                   
5       0.240                   0.800                   
6       0.200                   0.800                   
7       0.171                   0.800                   
8       0.163                   0.800                   
9       0.144                   0.800                   
10      0.130                   0.800
11      0.118                   0.800                   
12      0.108                   0.800                   
13      0.100                   0.800                   
14      0.093                   0.800                   
15      0.087                   0.800

no. pertanyaan tidak terjawab: 2 & 9
-------------------------------------------------------

chunk_size = 384
testset_topk3
top_k   avg_context_precision   avg_context_recall 
3       0.233                   0.700                   
4       0.200                   0.800                   
5       0.160                   0.800                   
6       0.133                   0.800                   
7       0.114                   0.800                   
8       0.100                   0.800                   
9       0.089                   0.800                   
10      0.080                   0.800
11      0.073                   0.800                   
12      0.067                   0.800                   
13      0.062                   0.800                   
14      0.057                   0.800                   
15      0.053                   0.800

no. pertanyaan tidak terjawab: 2 & 9
-------------------------------------------------------

chunk_size = 512
testset_topk3
top_k   avg_context_precision   avg_context_recall      
3       0.167                   0.500                   
4       0.125                   0.500                   
5       0.120                   0.600                   
6       0.133                   0.800                   
7       0.129                   0.900                   
8       0.125                   1.000                   
9       0.111                   1.000                   
10      0.100                   1.000

no. pertanyaan tidak terjawab: -
-------------------------------------------------------

uji coba
chunk_size = 128, top_k = 4
chunk_size = 256, top_k = 3 
chunk_size = 384, top_k = 4 
chunk_size = 512, top_k = 8 

- Chunk besar (512)
    - Keunggulan:
        - Mampu menjawab pertanyaan yang membutuhkan sumber referensi banyak, karena dapat menangkap informasi lebih banyak dibandingkan chunk kecil (384-128).
    - Kelemahan:
        - Penggunaan token lebih banyak karena membutuhkan topk besar
        - LLM rentan terkena fenomena “lost in the middle”
        - Berisiko halusinasi.
        - Lemah dalam mencari informasi yang bersifat spesifik
- Chunk kecil (128-384)
    - Keunggulan
        - Mampu menjawab pertanyaan yang membutuhkan jawaban sedikit dan spesifik.
        - Informasi yang di retrieve lebih padat sehingga LLM lebih fokus dan menghasilkan jawaban akurat.
    - Kelemahan
        - Tidak dapat menangkap data yang mengandung informasi besar dan membutuhkan topk lebih besar.
        - Hasil retrieval rentan terpotong dan kehilangan konteks utuh