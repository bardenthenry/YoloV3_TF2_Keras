# 使用說明

## 資料集格式

1. 此版本的 YoloV3 目前僅支援 VOC2012 資料集的格式

## 訓練步驟

1. 設定資料集位置
> 1. 開啟 config/dataset_config.json
> 1. 將 "img_dir" 項目值改成您目前資料集 JPEGImages 資料夾的路徑
> 1. 將 "annotation_dir" 項目值改成您目前資料集 Annotations 資料夾的路徑
> 1. 存檔後關閉
   
1. 修改資料集類別
> 1. 開啟 config/model_config.json
> 1. 將 "class_ls" 項目值改成您目前資料集的所有類別的名稱 list
> 1. 存檔後關閉

1. 計算 Anchor Box 長寬比
> 1. 執行以下命令
> ```
> python3 voc_annotation.py
> python3 kmeand.py
> ```