# 使用說明

此版本的 YoloV3 係參考 qqwweee 版本改寫成 tensorflow 2.0 框架，其中 voc_annotation.py 以及 kmeans.py 這兩個檔案由於時間很趕所以是直接拿他的來用，若有違反規定我馬上徹下！

## 資料集格式

1. 此版本的 YoloV3 目前僅支援 VOC2012 資料集的格式

## 訓練步驟

1. 設定資料集位置
> 1-1. 開啟 config/dataset_config.json
> 1-2. 將 "img_dir" 項目值改成您目前資料集 JPEGImages 資料夾的路徑
> 1-3. 將 "annotation_dir" 項目值改成您目前資料集 Annotations 資料夾的路徑
> 1-4. 存檔後關閉
   
2. 修改資料集類別
> 2-1. 開啟 config/model_config.json
> 2-2. 將 "class_ls" 項目值改成您目前資料集的所有類別的名稱 list
> 2-3. 存檔後關閉

3. 計算 Anchor Box 長寬比 (如果是使用 VOC2012的資料集可跳過這一步)
> 3-1. 執行以下命令之後，會看到根目錄出現 yolo_anchors.txt   
> ```
> python3 voc_annotation.py
> python3 kmeand.py
> ```
> 3-2. 開啟 config/model_config.json   
> 3-3. 將 yolo_anchors.txt 中的長寬比資料輸入到 config/model_config.json 檔案中的 "anchor_ls" 項目   
> 3-4. 存檔後關閉   

4. 產生 TFRecord 檔
> 4-1. 執行以下命令後，可以在 data 資料夾內看到 tfrecord 檔
> ```
> python3 WriteTFRecord.py
> ```

5. 開啟 config/train_config.json 可調整訓練參數
> 5-1. "learning_rate" 為學習率  
> 5-2. "lr_patient" validation loss 超過多少個 epoc 沒有創新低就降低學習率  
> 5-3. "early_stop_patient" validation loss 超過多少個 epoc 沒有創新低就停止訓練  
> 5-4. "n_epoc" 最多訓練多少個 epoc  
> 5-5. "batch_size" 每批輸入模型訓練的資料大小  
> 5-6. "num_parallel_calls" 資料前處理時，要使用多少 CPU 資源 (建議設定總核心數的一半就好)  
> 5-7. "pre_model_path" 預訓練模型的位置 (該功能還沒開放 暫時不要使用)  
> 5-8. "model_save_path" 模型訓練好之後存放的位置  
> 5-9. "log_dir" 訓練紀錄放置的位置  
> 5-10. "clipvalue" gradient 裁剪數 (目前是使用於 adam 中的 clipnorm 參數)  
> 5-11. "parallel_mode" 是否要使用多 GPU 來訓練  
> 5-12. "CUDA_VISIBLE_DEVICES" 此次訓練程式可看見的 GPU 編號  

6. 開始訓練
```
python3 Train.py
```