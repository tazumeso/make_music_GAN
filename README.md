# make_music_GAN

## 手順
1. xml2arr.pyでxmlファイルから配列を生成する。
2. make_music.pyでネットワークを学習する。
3. make_midi.pyで生成された配列をmidiファイルに変換する。

## 各コードの使用法
### xml2arr.py
after_xml_data直下にあるxmlファイルをnpyファイルに変換します。この時リズムチャネルを追加しています。

### make_music.py
ネットワークの学習を行います。
net.py:ネットワークの定義
updater.py:更新の定義
visualize.py:可視化の関数定義、２チャネルの出力に対して、チャネルを一個追加して通常の画像と形式を揃えています。

### make_midi.py
arr2midi()にnumpy配列を渡すと、midiファイルを作成してくれます。
