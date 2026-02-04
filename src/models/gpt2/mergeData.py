from pathlib import Path

# Creates both train and validation data from the `data` folder.
# For each file: first 90% goes to train, last 10% goes to validation.

def createTrainValSplit(dataDir: str, trainFile: str, valFile: str, valRatio: float = 0.1):
    data_path = Path(dataDir)
    txt_files = sorted(data_path.rglob("*.txt"))
    
    with open(trainFile, "w", encoding="utf-8") as train_out, \
         open(valFile, "w", encoding="utf-8") as val_out:
        
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split at character level
            split_point = int(len(content) * (1 - valRatio))
            
            train_content = content[:split_point]
            val_content = content[split_point:]
            
            # Write train portion
            train_out.write(train_content)
            if train_content and not train_content.endswith("\n"):
                train_out.write("\n")
            
            # Write validation portion
            val_out.write(val_content)
            if val_content and not val_content.endswith("\n"):
                val_out.write("\n")
    
    print(f"Processed {len(txt_files)} files")
    print(f"  Train data: {trainFile}")
    print(f"  Validation data: {valFile}")

if __name__ == "__main__":
    scriptDir = Path(__file__).parent
    dataDir = scriptDir / "data"
    trainFile = scriptDir / "train.txt"
    valFile = scriptDir / "validation.txt"
    
    createTrainValSplit(dataDir, trainFile, valFile)
