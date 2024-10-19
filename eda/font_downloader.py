import os
import zipfile
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def install_nanumgothic():
    font_dir = "/data/ephemeral/home/dataset/fonts"
    font_path = os.path.join(font_dir, "NanumGothicCoding.ttf")
    font_zip_path = os.path.join(font_dir, "NanumGothicCoding-2.5.zip")

    if not os.path.exists(font_dir):
        os.makedirs(font_dir)

    if not os.path.exists(font_path):
        print("NanumGothic 폰트를 설치 중입니다...")

        if not os.path.exists(font_zip_path):
            subprocess.run(["wget", "https://github.com/naver/nanumfont/releases/download/VER2.5/NanumGothicCoding-2.5.zip", "-P", font_dir], check=True)
        
        try:
            with zipfile.ZipFile(font_zip_path, 'r') as zip_ref:
                zip_ref.extractall(font_dir)
            print("폰트 압축 해제 완료")
        except zipfile.BadZipFile as e:
            print(f"ZIP 파일 해제 실패: {e}")
            return
    
        #subprocess.run(["fc-cache", "-f", font_dir], check=True)

        print("NanumGothic 폰트가 성공적으로 설치되었습니다.")


def set_nanumgothic_font():
    font_path = "/data/ephemeral/home/dataset/fonts/NanumGothicCoding.ttf"

    if not os.path.exists(font_path):
        install_nanumgothic()

    return font_path

