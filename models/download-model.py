from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SmilingWolf/wd-swinv2-tagger-v3",
    local_dir="./SmilingWolf_wd-swinv2-tagger-v3",
    local_dir_use_symlinks=False
)