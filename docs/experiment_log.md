# Experiment log

python3 docs/make_experiment_log.py

Each table lists all runs of one model on the full data, best on the validation split first (lowest validation RMSE for
svd). Selection never used the test columns; they are shown only to report the chosen run. Training times of sweep runs
were measured while other jobs shared the machine; the final runs (tag final) were run alone and are the ones in the cost table.

## popularity

1 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | val_rmse | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|---|
| popularity_final | (defaults) | 0.0136 | 1.1795 | 0.0249 | 0.0126 | 0.0000 | True |

## knn

9 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| knn_final | k=3200 | 0.0379 | 0.0562 | 0.0312 | 0.3200 | True |
| knn_k3200 | k=3200 | 0.0379 | 0.0562 | 0.0312 | 0.3000 | True |
| knn_k1600 | k=1600 | 0.0377 | 0.0561 | 0.0311 | 0.3000 | False |
| knn_k800 | k=800 | 0.0374 | 0.0553 | 0.0308 | 0.2800 | False |
| knn_k400 | k=400 | 0.0366 | 0.0540 | 0.0301 | 0.2600 | False |
| knn_k200 | k=200 | 0.0358 | 0.0524 | 0.0294 | 0.2400 | False |
| knn_k100 | k=100 | 0.0351 | 0.0503 | 0.0282 | 0.2200 | False |
| knn_k50 | k=50 | 0.0341 | 0.0490 | 0.0275 | 0.2200 | False |
| knn_k20 | k=20 | 0.0330 | 0.0477 | 0.0268 | 0.1900 | False |

## als

13 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| als_factors256 | factors=256, reg=0.1, alpha=40.0, foldin_all=nan | 0.0524 | 0.0598 | 0.0324 | 931.6500 | True |
| als_final | factors=256, reg=0.1, alpha=40.0, foldin_all=False | 0.0524 | 0.0598 | 0.0324 | 795.9300 | True |
| als_f128_r0.01 | factors=128, reg=0.01, alpha=40.0, foldin_all=nan | 0.0494 | 0.0593 | 0.0320 | 134.0700 | False |
| als_f128 | factors=128, reg=0.1, alpha=40.0, foldin_all=nan | 0.0494 | 0.0593 | 0.0320 | 134.4900 | False |
| als_factors128_reg10 | factors=128, reg=10.0, alpha=40.0, foldin_all=nan | 0.0494 | 0.0595 | 0.0321 | 130.9900 | False |
| als_f128_r1.0 | factors=128, reg=1.0, alpha=40.0, foldin_all=nan | 0.0494 | 0.0593 | 0.0320 | 134.5200 | False |
| als_f128_a100 | factors=128, reg=0.1, alpha=100.0, foldin_all=nan | 0.0493 | 0.0589 | 0.0314 | 132.6000 | False |
| als_factors128_reg100 | factors=128, reg=100.0, alpha=40.0, foldin_all=nan | 0.0489 | 0.0592 | 0.0321 | 128.4900 | False |
| als_f128_a10 | factors=128, reg=0.1, alpha=10.0, foldin_all=nan | 0.0458 | 0.0565 | 0.0307 | 133.2900 | False |
| als | factors=64, reg=0.1, alpha=40.0, foldin_all=nan | 0.0449 | 0.0574 | 0.0306 | 44.8800 | False |
| als_f64 | factors=64, reg=0.1, alpha=40.0, foldin_all=nan | 0.0449 | 0.0574 | 0.0306 | 45.1100 | False |
| als_f32 | factors=32, reg=0.1, alpha=40.0, foldin_all=nan | 0.0400 | 0.0538 | 0.0290 | 16.1900 | False |
| als_factors128_reg1000 | factors=128, reg=1000.0, alpha=40.0, foldin_all=nan | 0.0210 | 0.0337 | 0.0183 | 128.9300 | False |

## svd

24 runs, ranked by validation RMSE (lower is better).

| run | settings | val_ndcg@10 | val_rmse | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|---|
| svd_final | factors=32.0, reg=0.5 | 0.0001 | 1.0968 | 0.0003 | 0.0002 | 15.9300 | True |
| svd_factors32_reg0.5 | factors=32.0, reg=0.5 | 0.0001 | 1.0968 | 0.0003 | 0.0002 | 17.6800 | False |
| svd_factors16_reg0.5 | factors=16.0, reg=0.5 | 0.0000 | 1.0970 | 0.0003 | 0.0001 | 12.5100 | False |
| svd_factors64_reg0.5 | factors=64.0, reg=0.5 | 0.0001 | 1.0970 | 0.0002 | 0.0001 | 23.8500 | False |
| svd_factors8_reg0.5 | factors=8.0, reg=0.5 | 0.0000 | 1.0970 | 0.0003 | 0.0002 | 11.4700 | False |
| svd_factors64_reg0.2 | factors=64.0, reg=0.2 | 0.0001 | 1.1020 | 0.0003 | 0.0001 | 26.5600 | False |
| svd_factors32_reg1.0 | factors=32.0, reg=1.0 | 0.0001 | 1.1054 | 0.0003 | 0.0002 | 15.4500 | False |
| svd_factors16_reg1.0 | factors=16.0, reg=1.0 | 0.0001 | 1.1054 | 0.0003 | 0.0002 | 11.6500 | False |
| svd_factors32_reg0.2 | factors=32.0, reg=0.2 | 0.0001 | 1.1090 | 0.0002 | 0.0001 | 18.2800 | False |
| svd_factors64_reg0.1 | factors=64.0, reg=0.1 | 0.0002 | 1.1137 | 0.0003 | 0.0001 | 23.6000 | False |
| svd_factors8_reg0.2 | factors=8.0, reg=0.2 | 0.0001 | 1.1210 | 0.0002 | 0.0001 | 6.2000 | False |
| svd_factors16_reg0.2 | factors=16.0, reg=0.2 | 0.0001 | 1.1225 | 0.0002 | 0.0001 | 4.8700 | False |
| svd_factors32_reg2.0 | factors=32.0, reg=2.0 | 0.0001 | 1.1270 | 0.0003 | 0.0002 | 17.2200 | False |
| svd_factors16_reg2.0 | factors=16.0, reg=2.0 | 0.0001 | 1.1270 | 0.0003 | 0.0002 | 13.3300 | False |
| svd_factors8_reg0.1 | factors=8.0, reg=0.1 | 0.0002 | 1.1340 | 0.0006 | 0.0003 | 4.6700 | False |
| svd_factors16_reg0.1 | factors=16.0, reg=0.1 | 0.0003 | 1.1379 | 0.0006 | 0.0003 | 3.7400 | False |
| svd_factors8_reg0.05 | factors=8.0, reg=0.05 | 0.0006 | 1.1412 | 0.0013 | 0.0006 | 3.7900 | False |
| svd_factors32_reg0.1 | factors=32.0, reg=0.1 | 0.0004 | 1.1426 | 0.0008 | 0.0004 | 4.9300 | False |
| svd_factors16_reg0.05 | factors=16.0, reg=0.05 | 0.0010 | 1.1466 | 0.0015 | 0.0008 | 3.3300 | False |
| svd | factors=32.0, reg=0.05 | 0.0011 | 1.1538 | 0.0017 | 0.0009 | 5.0200 | False |
| svd_factors32_reg0.05 | factors=32.0, reg=0.05 | 0.0011 | 1.1538 | 0.0017 | 0.0009 | 4.0500 | False |
| svd_factors64_reg0.05 | factors=64.0, reg=0.05 | 0.0011 | 1.1583 | 0.0019 | 0.0009 | 5.0100 | False |
| svd_factors16_reg5.0 | factors=16.0, reg=5.0 | 0.0001 | 1.1668 | 0.0003 | 0.0001 | 9.3900 | False |
| svd_factors32_reg5.0 | factors=32.0, reg=5.0 | 0.0001 | 1.1668 | 0.0002 | 0.0001 | 11.6200 | False |

## bpr

17 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| bpr_final | factors=128.0, lr=0.005, reg=0.003, epochs=80.0, patience=15.0 | 0.0398 | 0.0520 | 0.0271 | 174.3000 | True |
| bpr_patience15_epochs80_factors128_reg0.003 | factors=128.0, lr=0.005, reg=0.003, epochs=80.0, patience=15.0 | 0.0398 | 0.0520 | 0.0271 | 168.8700 | True |
| bpr_r2_f128_reg0.001 | factors=128.0, lr=0.005, reg=0.001, epochs=150.0, patience=30.0 | 0.0398 | 0.0515 | 0.0270 | 564.6900 | False |
| bpr_r2_f256 | factors=256.0, lr=0.005, reg=0.003, epochs=150.0, patience=15.0 | 0.0379 | 0.0491 | 0.0261 | 265.9000 | False |
| bpr_patience15_epochs80_factors64_reg0.003 | factors=64.0, lr=0.005, reg=0.003, epochs=80.0, patience=15.0 | 0.0357 | 0.0476 | 0.0245 | 112.5400 | False |
| bpr_patience15_epochs80_factors64_reg0.01 | factors=64.0, lr=0.005, reg=0.01, epochs=80.0, patience=15.0 | 0.0334 | 0.0471 | 0.0240 | 94.5400 | False |
| bpr_patience15_epochs80_factors128_reg0.01 | factors=128.0, lr=0.005, reg=0.01, epochs=80.0, patience=15.0 | 0.0331 | 0.0474 | 0.0243 | 70.3800 | False |
| bpr_factors64_reg0.003 | factors=64.0, lr=0.005, reg=0.003, epochs=50.0, patience=5.0 | 0.0309 | 0.0422 | 0.0220 | 74.1200 | False |
| bpr_factors64_reg0.01 | factors=64.0, lr=0.005, reg=0.01, epochs=50.0, patience=5.0 | 0.0305 | 0.0433 | 0.0223 | 38.4500 | False |
| bpr_r2_f128_lr0.01 | factors=128.0, lr=0.01, reg=0.003, epochs=150.0, patience=15.0 | 0.0289 | 0.0404 | 0.0206 | 150.0100 | False |
| bpr_factors64_reg0.02 | factors=64.0, lr=0.005, reg=0.02, epochs=50.0, patience=5.0 | 0.0289 | 0.0417 | 0.0214 | 31.4700 | False |
| bpr_factors64_reg0.006 | factors=64.0, lr=0.005, reg=0.006, epochs=50.0, patience=5.0 | 0.0287 | 0.0402 | 0.0208 | 38.8800 | False |
| bpr_factors64_reg0.04 | factors=64.0, lr=0.005, reg=0.04, epochs=50.0, patience=5.0 | 0.0242 | 0.0380 | 0.0192 | 31.9000 | False |
| bpr_factors64_reg0.001 | factors=64.0, lr=0.005, reg=0.001, epochs=50.0, patience=5.0 | 0.0169 | 0.0245 | 0.0125 | 25.2800 | False |
| bpr | factors=64.0, lr=0.005, reg=1e-05, epochs=50.0, patience=5.0 | 0.0168 | 0.0247 | 0.0130 | 21.0400 | False |
| bpr_factors64_reg0.0001 | factors=64.0, lr=0.005, reg=0.0001, epochs=50.0, patience=5.0 | 0.0166 | 0.0246 | 0.0128 | 25.9400 | False |
| bpr_factors64_reg0.1 | factors=64.0, lr=0.005, reg=0.1, epochs=50.0, patience=5.0 | 0.0127 | 0.0249 | 0.0118 | 13.5900 | False |

## ncf

12 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| ncf_final | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.003, dropout=0.2 | 0.0249 | 0.0377 | 0.0197 | 65.7800 | True |
| ncf_dropout0.2_patience5_epochs40_reg0.003 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.003, dropout=0.2 | 0.0249 | 0.0376 | 0.0195 | 134.1000 | False |
| ncf_dropout0.2_patience5_epochs40_reg0.005 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.005, dropout=0.2 | 0.0247 | 0.0375 | 0.0197 | 83.2100 | False |
| ncf_reg0.003_patience5_epochs40_dropout0.5 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.003, dropout=0.5 | 0.0243 | 0.0353 | 0.0183 | 125.2500 | False |
| ncf_reg0.001_patience5_epochs40_dropout0.5 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.001, dropout=0.5 | 0.0226 | 0.0321 | 0.0166 | 143.0900 | False |
| ncf_dropout0.2_patience5_epochs40_reg0.001 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.001, dropout=0.2 | 0.0221 | 0.0312 | 0.0159 | 212.1600 | False |
| ncf_dropout0.2_patience5_epochs40_reg0.0003 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.0003, dropout=0.2 | 0.0195 | 0.0277 | 0.0142 | 199.6100 | False |
| ncf_reg0.003_dropout0.2_patience5_epochs40_lr0.003 | factors=32.0, mlp_factors=32.0, lr=0.003, epochs=40.0, patience=5.0, reg=0.003, dropout=0.2 | 0.0189 | 0.0293 | 0.0149 | 95.8000 | False |
| ncf_dropout0.2_patience5_epochs40_reg0.01 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.01, dropout=0.2 | 0.0184 | 0.0313 | 0.0162 | 189.9500 | False |
| ncf_mlp_factors16_reg0.001_dropout0.2_patience5_epochs40_factors16 | factors=16.0, mlp_factors=16.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.001, dropout=0.2 | 0.0171 | 0.0261 | 0.0130 | 75.5800 | False |
| ncf_dropout0.2_patience5_epochs40_reg0.03 | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=40.0, patience=5.0, reg=0.03, dropout=0.2 | 0.0168 | 0.0294 | 0.0152 | 119.3600 | False |
| ncf_first | factors=32.0, mlp_factors=32.0, lr=0.001, epochs=30.0, patience=3.0, reg=nan, dropout=nan | 0.0160 | 0.0250 | 0.0127 | 136.2300 | False |

## sasrec

19 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| sasrec_blocks1_dropout0.5_hidden128 | max_len=20.0, hidden=128.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0371 | 0.0618 | 0.0329 | 107.7700 | True |
| sasrec_final | max_len=20.0, hidden=128.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0366 | 0.0632 | 0.0335 | 99.4900 | False |
| sasrec_blocks1_dropout0.5_max_len30 | max_len=30.0, hidden=64.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0365 | 0.0619 | 0.0326 | 111.0600 | False |
| sasrec_blocks1_dropout0.5 | max_len=20.0, hidden=64.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0357 | 0.0608 | 0.0322 | 208.5600 | False |
| sasrec_blocks1_dropout0.6 | max_len=20.0, hidden=64.0, blocks=1.0, dropout=0.6, lr=0.001 | 0.0356 | 0.0619 | 0.0327 | 187.1400 | False |
| sasrec_blocks1_dropout0.4 | max_len=20.0, hidden=64.0, blocks=1.0, dropout=0.4, lr=0.001 | 0.0353 | 0.0621 | 0.0327 | 135.5200 | False |
| sasrec_blocks1_dropout0.5_max_len50 | max_len=50.0, hidden=64.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0345 | 0.0606 | 0.0319 | 116.9600 | False |
| sasrec_dropout0.4 | max_len=20.0, hidden=64.0, blocks=2.0, dropout=0.4, lr=0.001 | 0.0341 | 0.0585 | 0.0310 | 170.7300 | False |
| sasrec_blocks1_dropout0.5_hidden32 | max_len=20.0, hidden=32.0, blocks=1.0, dropout=0.5, lr=0.001 | 0.0330 | 0.0581 | 0.0306 | 89.2900 | False |
| sasrec_blocks1 | max_len=20.0, hidden=64.0, blocks=1.0, dropout=0.2, lr=0.001 | 0.0327 | 0.0565 | 0.0298 | 224.5100 | False |
| sasrec_lr0.0005 | max_len=20.0, hidden=64.0, blocks=2.0, dropout=0.2, lr=0.0005 | 0.0320 | 0.0559 | 0.0291 | 163.0600 | False |
| sasrec_lr0.003 | max_len=20.0, hidden=64.0, blocks=2.0, dropout=0.2, lr=0.003 | 0.0308 | 0.0532 | 0.0278 | 129.3000 | False |
| sasrec_hidden32 | max_len=20.0, hidden=32.0, blocks=2.0, dropout=0.2, lr=0.001 | 0.0306 | 0.0541 | 0.0282 | 91.0700 | False |
| sasrec_max_len50 | max_len=50.0, hidden=64.0, blocks=2.0, dropout=0.2, lr=0.001 | 0.0299 | 0.0538 | 0.0279 | 310.3600 | False |
| sasrec_hidden128 | max_len=20.0, hidden=128.0, blocks=2.0, dropout=0.2, lr=0.001 | 0.0297 | 0.0519 | 0.0264 | 96.1600 | False |
| sasrec_dropout0.1 | max_len=20.0, hidden=64.0, blocks=2.0, dropout=0.1, lr=0.001 | 0.0297 | 0.0518 | 0.0268 | 58.1300 | False |
| sasrec_first | max_len=20.0, hidden=64.0, blocks=2.0, dropout=0.2, lr=0.001 | 0.0287 | 0.0511 | 0.0266 | 110.4100 | False |
| sasrec_blocks3 | max_len=20.0, hidden=64.0, blocks=3.0, dropout=0.2, lr=0.001 | 0.0284 | 0.0492 | 0.0256 | 123.8900 | False |
| sasrec_max_len10 | max_len=10.0, hidden=64.0, blocks=2.0, dropout=0.2, lr=0.001 | 0.0283 | 0.0519 | 0.0269 | 202.4000 | False |

## content

13 runs, ranked by validation NDCG@10.

| run | settings | val_ndcg@10 | test_recall@10 | test_ndcg@10 | fit_seconds | selected |
|---|---|---|---|---|---|---|
| content_final | lam=0.5 | 0.0225 | 0.0382 | 0.0199 | 0.0100 | True |
| content_lam0.5 | lam=0.5 | 0.0225 | 0.0382 | 0.0199 | 0.0100 | True |
| content_lam0.7 | lam=0.7 | 0.0224 | 0.0387 | 0.0200 | 0.0100 | False |
| content_lam1.0 | lam=1.0 | 0.0213 | 0.0372 | 0.0192 | 0.0100 | False |
| content_lam0.3 | lam=0.3 | 0.0208 | 0.0352 | 0.0185 | 0.0000 | False |
| content_lam1.4 | lam=1.4 | 0.0196 | 0.0345 | 0.0179 | 0.0000 | False |
| content_lam2 | lam=2.0 | 0.0176 | 0.0316 | 0.0164 | 0.0000 | False |
| content_lam3 | lam=3.0 | 0.0164 | 0.0290 | 0.0152 | 0.0000 | False |
| content_lam5 | lam=5.0 | 0.0157 | 0.0274 | 0.0144 | 0.0000 | False |
| content_first | lam=0.1 | 0.0151 | 0.0252 | 0.0132 | 136.3200 | False |
| content_lam10 | lam=10.0 | 0.0149 | 0.0259 | 0.0138 | 0.0100 | False |
| content_lam0.03 | lam=0.03 | 0.0121 | 0.0202 | 0.0104 | 0.0100 | False |
| content_lam0 | lam=0.0 | 0.0107 | 0.0183 | 0.0093 | 0.0100 | False |
