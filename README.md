# pix2pix.pytorch
Pix2Pix をPyTorch で実装しました．
[著者らのPyTorch実装](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)からpix2pixを抜き出し，実験しやすいように整理しました．ネットワークの構造，初期化など重要な部分は著者らと同様です．

# 環境
動作確認をしたのは以下の環境下です．

- Ubuntu18.04LTS
- CUDA 10.0
- cuDNN 7.4
- Python 3
- CPU : GeForce GTX 1080

必要なpythonモジュールは以下の通りです．おそらく多少バージョンが違っても大丈夫だと思います．

- torch==1.2.0
- torchvision==0.4.0
- opencv-python==4.1.1.26
- matplotlib==3.1.1
- pandas==0.25.1
- numpy==1.17.1
- tqdm==4.35.0

# 学習方法
## 準備
本プログラムでは，各行がpix2pixへの入力画像Aと出力画像Bのパスとなっているcsv ファイルを必要とします．
csv ファイルの中身は以下のようにしてください．`A_path`の列には入力画像Aへのパスを入れ，`B_path`の列には出力画像Bへのパスを入れます．ここで，各行はペアになるように注意してください．

| A_path | B_path |
|:----:|:----|
|hoge1.jpg | fuga1.png|
|hoge2.jpg | fuga2.png|
|hoge2.jpg | fuga3.png|

以上の形式のcsv ファイルを学習用は `train_df.csv`，検証用は `val_df.csv` という名前で作成してください．

## 学習
`train.py` を実行することで学習ができます．
実行は以下のように行います．

```
python train.py --dataroot hoge/df_dir -n 1 -s 1 --name edges2shoes --save_dir fuga/
```

ここでプログラムの引数は以下の通りです．
- --dataroot : 上記で作成したdataframeが入っているディレクトリへのパス
- --num : 実験番号
- --seed : seed 値
- --name : 実験名．例えば edges2shoes
- --save_dir : モデルの重みや生成画像を保存する用のディレクトリ

そのほかにもデータ拡張手法の選択や，モデルの層数，バッチサイズなどが指定できる引数があります．詳しくは以下のコマンドでhelpをみてください．
```
python train.py --help
```

実行すると，`--save_dir` で指定したディレクトリに以下のディレクトリが作成されます．
- gen_images : 学習，検証の一部（枚数は`--vis_num` で指定可能）の生成画像を保存するディレクトリ
- models : モデルが保存されるディレクトリ
- log : epochごとの学習lossを保存するディレクトリ．

## 再学習
`resume.py` を実行することで再学習できます．

使用方法は以下のコマンドで確認してください．

```
python resume.py --help
```

## テスト
テストのスクリプトは書けていません．