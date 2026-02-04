from pathlib import Path

# Creates both train and validation data from the `data` folder.

# For every .txt file in the given directory, simply concatenates them into a single .txt file
def mergeTxtFiles(dataDir: str, outputFile: str):
    data_path = Path(dataDir)
    txt_files = sorted(data_path.rglob("*.txt"))
    
    with open(outputFile, "w", encoding="utf-8") as out:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                out.write(content)
                # Ensure there's a newline between files
                if not content.endswith("\n"):
                    out.write("\n")
    
    print(f"\nMerged {len(txt_files)} files into {outputFile}")

if __name__ == "__main__":
    scriptDir = Path(__file__).parent
    dataDir = scriptDir / "data"
    outputFile = scriptDir / "merged.txt"
    
    mergeTxtFiles(dataDir, outputFile)
