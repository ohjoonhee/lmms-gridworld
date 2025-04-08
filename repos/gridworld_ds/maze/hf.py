import os
from datasets import load_dataset
from datasets import Image
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv()

    ds = load_dataset("json", data_files="data/5x5.jsonl")
    ds["test"] = ds["train"]
    ds.pop("train")  # Remove the original train split

    ds["test"] = ds["test"].cast_column("image", Image())

    print(ds)  # Verify the change
    print(ds["test"].features)
    print(ds["test"][0])

    ds.push_to_hub("ohjoonhee/mazenav_vqa", token=os.getenv("HF_TOKEN"))
