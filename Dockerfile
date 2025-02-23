FROM python:3.12
WORKDIR /app
COPY . /app
# 安装系统依赖

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 7860
CMD ["python", "main.py"]