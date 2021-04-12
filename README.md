## 環境構築

仮想環境下で以下のコマンドを実行し、プログラムを動かすのに必要なライブラリをインストールしてください(pytorch以外)

```sh
pip install -r requirements.txt
```

pytorchは[公式HP](https://pytorch.org/get-started/locally/)で、CUDAなどのバージョンを見ながらインストールしてください

## 事前準備

jupyterで`00_setup.ipynb`を起動し、すべてのセルを実行してください

データセットのダウンロードが行われます


## Tensorboard

jupyterでは説明しませんが`main.py`にはtensorboardの使用例も載せています

Tensorboardは端末で以下のコマンドにより起動します

```sh
tensorboard --logdir [パス]
```
パスはプログラム中に記述します（例では`logs`）
