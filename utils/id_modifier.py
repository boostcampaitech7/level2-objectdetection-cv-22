import pandas as pd

"""
    csv 파일에서 특정 칼럼의 일부를 수정하여 다시 저장
    
    불러 올 파일의 경로 : path_dataset

    불러 올 파일의 이름 : filename_input
    저장할 파일의 이름 : filename_output

    칼럼명 : column

    삭제할 문구 : string
    대체할 문구 : string_replaced
"""
# 경로 ─────────────────────────────────────────────────────────────

path_dir = '/data/ephemeral/home/outputs/'

path_input = path_dir + 'submission_.csv'
path_output = path_dir + 'submission_det.csv'

column = 'image_id'

string = '/data/ephemeral/home/dataset/'
string_replaced = ''

# ──────────────────────────────────────────────────────────────────

df = pd.read_csv(path_input)
df[column] = df[column].str.replace(string, string_replaced, regex=False)

df.to_csv(path_output, index=False)

print("수정된 CSV 파일이 저장되었습니다:", path_output)