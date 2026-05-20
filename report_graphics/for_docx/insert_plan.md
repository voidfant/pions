# План вставки графиков в отчет

## Раздел 1. Трансформер

1. `1_01_01_positional_encoding_heatmap.png` — Тепловая карта позиционного кодирования
2. `1_02_02_attention_weights_head0.png` — Матрица весов самовнимания
3. `1_03_03_attention_vs_rnn_train_loss.png` — Сравнение Attention и RNN по функции потерь на обучении
4. `1_04_04_attention_vs_rnn_test_accuracy.png` — Сравнение Attention и RNN по точности на тесте
5. `1_05_05_minitransformer_loss_small.png` — MiniTransformer small: функция потерь
6. `1_06_06_minitransformer_acc_small.png` — MiniTransformer small: точность токенов
7. `1_07_07_minitransformer_final_comparison.png` — Финальное сравнение конфигураций MiniTransformer

## Раздел 2. Генеративно-состязательная сеть (GAN)

1. `2_01_01_distribution_epoch_001.png` — Распределение реальных и сгенерированных точек, epoch 001
2. `2_02_01_distribution_epoch_093.png` — Распределение реальных и сгенерированных точек, epoch 093
3. `2_03_01_distribution_epoch_186.png` — Распределение реальных и сгенерированных точек, epoch 186
4. `2_04_01_distribution_epoch_280.png` — Распределение реальных и сгенерированных точек, epoch 280
5. `2_05_02_gan_losses.png` — Кривые потерь GAN
6. `2_06_03_discriminator_confidence.png` — Динамика уверенности дискриминатора
7. `2_07_04_generated_density_hexbin.png` — Плотность сгенерированных точек

## Раздел 3. Графовая нейросеть

1. `3_01_01_gcn_vs_mlp_train_loss.png` — GCN vs MLP: функция потерь на обучении
2. `3_02_02_gcn_vs_mlp_test_accuracy.png` — GCN vs MLP: точность на тесте
3. `3_03_03_final_accuracy_bar.png` — Финальная точность методов
4. `3_04_04_gcn_embeddings_pca.png` — PCA эмбеддингов GCN
5. `3_05_05_graph_structure_subgraph.png` — Визуализация подграфа
6. `3_06_06_degree_distribution.png` — Распределение степеней узлов

## Раздел 4. Влияние архитектуры модели на прогнозирование рыночных временных рядов

Графики лежат в `4_market_architecture/`.

1. `4_market_architecture/4_00_market_experiment_pipeline.png` — Конвейер эксперимента раздела 4
2. `4_market_architecture/4_09_split_and_label_distribution.png` — Размеры обучающего, валидационного и тестового разбиений и распределение классов
3. `4_market_architecture/4_11_market_return_context.png` — Накопленная будущая доходность ret_h по тикерам
4. `4_market_architecture/4_10_feature_group_counts.png` — Группы признаков market-only датасета
5. `4_market_architecture/4_12_architecture_assumptions.png` — Сравниваемые архитектуры и их индуктивные предположения
6. `4_market_architecture/4_13_training_val_balanced_accuracy.png` — Сбалансированная accuracy нейросетевых моделей на валидации
7. `4_market_architecture/4_14_training_loss_curves.png` — Функция потерь нейросетевых моделей на обучении
8. `4_market_architecture/4_01_architecture_quality_metrics.png` — Сравнение архитектур по сбалансированной accuracy, macro-F1 и hit-rate top-10% сигналов
9. `4_market_architecture/4_08_metrics_heatmap.png` — Тепловая карта итоговых метрик
10. `4_market_architecture/4_02_quality_vs_train_time.png` — Компромисс качества и времени обучения
11. `4_market_architecture/4_03_inference_latency.png` — Скорость инференса разных архитектур
12. `4_market_architecture/4_04_parameter_complexity.png` — Параметрическая сложность моделей
13. `4_market_architecture/4_18_quality_efficiency_radar.png` — Интегральный профиль качества и эффективности
14. `4_market_architecture/4_15_score_distributions.png` — Распределение score по моделям
15. `4_market_architecture/4_06_score_vs_future_return.png` — Связь score модели и будущей доходности
16. `4_market_architecture/4_16_top10_pnl_by_model.png` — PnL top-10% наиболее уверенных сигналов
17. `4_market_architecture/4_07_equity_curves_top10.png` — Кривые капитала для top-10% сигналов
18. `4_market_architecture/4_05_best_model_confusion_matrix.png` — Матрица ошибок лучшей модели
19. `4_market_architecture/4_17_best_model_rolling_error.png` — Скользящая доля ошибок лучшей модели
