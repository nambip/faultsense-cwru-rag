# FaultSense

FaultSense is a two-stage CWRU bearing fault diagnosis and repair advisory system.

1. A Qwen2.5-7B-Instruct LoRA classifier predicts bearing fault type and severity from serialized vibration features.
2. A FAISS-backed RAG pipeline retrieves bearing-maintenance manual passages and generates cited repair guidance.

## Files

```text
FaultSense/
  FaultSense_CWRU_EndToEnd_V1.ipynb       # data prep, augmentation, QLoRA training, evaluation
  FaultSense_CWRU_RAG_Pipeline_V2.ipynb   # RAG index, retrieval tests, full LoRA + RAG demo
  augment_cwru.py                         # training-set augmentation helper
  evaluation/                             # saved demo outputs from Drive
  finetune/                               # LoRA adapters from Drive
  rag/                                    # PDF corpus and FAISS index from Drive
FaultSense_Final_Report.md
TODO_FaultSense_Final_Submission.md
```

## Download Required Artifacts

Download `finetune/` and `rag/` from Google Drive and place them under `FaultSense/`.

**Google Drive artifacts:** (https://drive.google.com/drive/folders/1yHtabvkPp-gzhq_4R7KLahJyyKm-3LtS?usp=sharing)

Expected local layout after download:

```text
FaultSense/
  finetune/
    r8_all7_2ep/
    r16_all7_2ep/        # optional, after rank-16 run
  rag/
    corpus/
    index/
```

If you are running in Google Colab, mount Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

The notebooks expect this Drive root by default:

```text
/content/drive/MyDrive/FaultSense_Project
```

## Run The Training Notebook

Open:

```text
FaultSense/FaultSense_CWRU_EndToEnd_V1.ipynb
```

Use it to:

1. Load the CWRU feature CSV.
2. Build instruction-tuning records.
3. Split train/validation/test data.
4. Augment the training split.
5. Train the QLoRA adapter.
6. Evaluate classification accuracy.

The saved rank-8 run uses:

```text
RUN_TAG = "r8_all7_2ep"
LoRA rank = 8
LoRA alpha = 16
```

For the rank-16 ablation, change:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    ...
)

RUN_TAG = "r16_all7_2ep"
```

Save rank-16 outputs separately under:

```text
/content/drive/MyDrive/FaultSense_Project/finetune/r16_all7_2ep
```

## Run The RAG Notebook

Open:

```text
FaultSense/FaultSense_CWRU_RAG_Pipeline_V2.ipynb
```

Use it to:

1. Load the trained LoRA adapter.
2. Build or load the FAISS RAG index.
3. Test manual retrieval.
4. Run the full FaultSense pipeline.
5. Save demo outputs under `evaluation/`.

The RAG notebook expects:

```text
/content/drive/MyDrive/FaultSense_Project/finetune/r8_all7_2ep
/content/drive/MyDrive/FaultSense_Project/rag/corpus
/content/drive/MyDrive/FaultSense_Project/rag/index
```

Change the adapter path to `r16_all7_2ep` only when evaluating the rank-16 adapter.

## Save Results

Full-pipeline demo outputs are saved under `FaultSense/evaluation/`.
