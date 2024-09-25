import json
import re
import os

def sum_compute(scores):
    scores_float = [float(score) for score in scores]
    if scores_float:
        average_score = sum(scores_float) / len(scores_float)
    else:
        average_score = 0
    return average_score

# Regular expression patterns
pattern_helpful = re.compile(r"帮助性:\s*(\d+\.\d+)/2")
pattern_relevent = re.compile(r"相关性:\s*(\d+\.\d+)/2")
pattern_accurate = re.compile(r"准确性:\s*(\d+\.\d+)/2")
pattern_deep = re.compile(r"深度:\s*(\d+\.\d+)/2")
pattern_creative = re.compile(r"创造性:\s*(\d+\.\d+)/1")
pattern_detail = re.compile(r"回应的细节程度:\s*(\d+\.\d+)/1")
pattern_panish = re.compile(r'惩罚[:：]\s*([-−]*\d+)\s*分')

# Directory path
directory_path = "/aifs4su/yaodong/changye/hospital/data/eval_result_8_23"  # 替换为实际目录路径

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.startswith("gpt4_eval_result") and filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        print(filename)  # 输出当前处理的文件名
        
        try:
            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Skipping file {filename} due to JSONDecodeError: {e}")
            continue
        except Exception as e:
            print(f"Skipping file {filename} due to an unexpected error: {e}")
            continue
        
        # Initialize lists to store scores for the current file
        scores_helpful, scores_relevent = [], []
        scores_accurate, scores_deep = [], []
        scores_creative, scores_detail, scores_panish = [], [], []

        # Extract scores from each file
        for item in data:
            for followup in item.get("followup_result", []):
                try:
                    response = followup.get("response", "")
                except Exception as e:
                    print(f"Error processing followup in {filename}: {e}")
                    continue

                matches_helpful = pattern_helpful.findall(response)
                scores_helpful.extend(matches_helpful)

                matches_relevent = pattern_relevent.findall(response)
                scores_relevent.extend(matches_relevent)

                matches_accurate = pattern_accurate.findall(response)
                scores_accurate.extend(matches_accurate)

                matches_deep = pattern_deep.findall(response)
                scores_deep.extend(matches_deep)

                matches_creative = pattern_creative.findall(response)
                scores_creative.extend(matches_creative)

                matches_detail = pattern_detail.findall(response)
                scores_detail.extend(matches_detail)

                matches_panish = pattern_panish.findall(response)
                scores_panish.extend(matches_panish)
        if min(len(scores_helpful), len(scores_relevent), len(scores_accurate), len(scores_deep), len(scores_creative), len(scores_detail), len(scores_panish)) <30:
            
            continue
        # Calculate and print average scores for the current file
        avg_score_helpful = sum_compute(scores_helpful)
        avg_score_relevent = sum_compute(scores_relevent)
        avg_score_accurate = sum_compute(scores_accurate)
        avg_score_deep = sum_compute(scores_deep)
        avg_score_creative = sum_compute(scores_creative)
        avg_score_detail = sum_compute(scores_detail)
        avg_score_panish = sum_compute(scores_panish)

        print(f"Results for {filename}:")
        print(f"scores_helpful: {avg_score_helpful}")
        print(f"scores_relevent: {avg_score_relevent}")
        print(f"scores_accurate: {avg_score_accurate}")
        print(f"scores_deep: {avg_score_deep}")
        print(f"scores_creative: {avg_score_creative}")
        print(f"scores_detail: {avg_score_detail}")
        print(f"scores_panish: {avg_score_panish}")
        print("-" * 50)  # 分割线，用于区分不同文件的输出
