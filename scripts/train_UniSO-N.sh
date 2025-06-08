python src/train_UniSO-N.py experiment=finetune_t5 


# After specified the finetuned T5 ``.pt`` path,
PT_PATH=
python src/train_UniSO-N.py \
    experiment=UniSO-N \
    ++trainer.max_epochs=200 \
    ++pt_path=$PT_PATH