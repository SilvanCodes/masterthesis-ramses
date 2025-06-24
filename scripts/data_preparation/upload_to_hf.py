from huggingface_hub import HfApi

api = HfApi()

private = False
repo_id = "sbuedenb/big_beetle_dataset-1024"  # replace with your username, dataset name
folder_path = "results/dataset"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="dataset")
