def get_last_line_of_record(root, filename=''):
    """
    filename이 빈 문자열인 경우, record.txt에서 마지막 줄을 읽어 반환.
    그렇지 않은 경우에는 아무 작업도 하지 않음.
    """
    if filename == '':
        try:
            with open(root + "/outputs/record.txt", "r") as f:
                lines = f.readlines()
                if lines:
                    return lines[-1].strip()
                else:
                    print("record.txt 파일이 비어 있습니다.")
                    return filename
        except FileNotFoundError:
            print("record.txt 파일이 존재하지 않습니다.")
            return filename
    else:
        print("filename이 빈 문자열이 아닙니다.")
        return filename