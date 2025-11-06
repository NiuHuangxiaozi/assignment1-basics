




python tokenize_owe_tiny_story_dataset.py \
         --text_data_path /home/niu/code/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt \
         --tokenizer_vocab_path /home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_vocab_10000.pkl \
         --tokenizer_merge_path /home/niu/code/cs336/assignment1-basics/cs336_basics/bpe/output/TinyStoriesV2-GPT4-train_optim_merges_10000.pkl \
         --id_save_path tinystories_sample_5M_train_ids.npy \
         --save_ids_interval 100000 \
         --num_chunks 100