# AviUtl-EllipsoidClipping
![GitHub Repo License](https://img.shields.io/github/license/sing-kuro/AviUtl-EllipsoidClipping)
![GitHub Repo stars](https://img.shields.io/github/stars/sing-kuro/AviUtl-EllipsoidClipping)
![Cpp](https://img.shields.io/badge/-Cplusplus-00599C.svg?logo=cplusplus&style=plastic)
![Lua](https://img.shields.io/badge/-Lua-2C2D72.svg?logo=lua&style=plastic)

AviUtl上で月の満ち欠けなどを表現するのに使えるスクリプトです。  
円や楕円を、球や楕円体とみなして満ち欠けさせます(扇クリッピングと満ち欠け円の機能を兼ね備えていると言えばわかりやすいでしょうか)。  

## インストール
1. [Releases](https://github.com/sing-kuro/AviUtl-EllipsoidClipping/releases)から最新のellipsoid_clipping.zipをダウンロードする。
1. ダウンロードしたファイルを右クリックしてプロパティを開き、「全般」タブの下のほうにある「セキュリティ」の項目から「許可する」にチェックをいれて適用する。
1. ellipsoid_clipping.zipを解凍する。
1. 「ellipsoid_clipping」フォルダ内の「楕円体クリッピング」フォルダをまるごとAviUtlのscriptフォルダにコピー(または移動)する。

## 使い方
使用したい円か楕円の画像をAviUtlのタイムラインに読み込んでください。その画像にアニメーション効果を追加し、「楕円体クリッピング」を選択してください。  
(以降、円は楕円、球は楕円体として説明を進めます)  
以下、スライダーとチェックボックスの説明
- 基準角Z・Y・X : どこを欠けさせる(満ちさせる)か(単位は度)
- 拡張角 : どの程度欠けさせる(満ちさせる)か(単位は度)
- マスクの反転 : 透明部分と不透明部分を反転するかどうか
### 画像の大きさいっぱいに楕円がまっすぐ描かれている場合
基準角Z・Y・Xと拡張角だけを適当にいじれば良いです(普通に満ち欠けさせる場合は基準角Yは0のままになると思います)。ほかのパラメータは自動で画像の大きさに合うようになっています。

### その他の場合
このスクリプトは、円や楕円を球や楕円体とみなして、それを中心を通る二つの平面で切り取るものです。  
このときに楕円体の寸法と位置・姿勢がわかっていなければなりません。デフォルトでは画像の縦と横の長さを寸法に使い、楕円体は画像の中心にあり正面を向いているとみなすようになっています。  
これ以外の場合、デフォルトではうまくいかないので、この位置・寸法・姿勢をパラメータ設定ダイアログに入力してください。  
このとき、「幅を元のサイズに」などのチェックを外さないと値が適用されません。

以下、各パラメータの説明
- 幅を元のサイズに : チェックが入っていると「幅」パラメータは無視され、画像の横の長さが適用されます。
- 高さを元のサイズに : チェックが入っていると「高さ」パラメータは無視され、画像の縦の長さが適用されます。
- 奥行きを幅に合わ : チェックが入っていると「奥行き」パラメータは無視され、幅の値が適用されます。
- 位置を相対位置に : チェックが入っていると「X」,「Y」パラメータは無視され、それぞれ「幅」と「高さ」の半分の値が適用されます。
- X,Y : 楕円の中心の、画像に対する相対座標。
- 幅, 高さ : 楕円の長軸と短軸の長さ(円の場合はともに直径)。
- 奥行き : 楕円体の奥行き。これを変えることで満ち欠けの様子が変わる。
- 回転Z,Y,X : 楕円体の姿勢(単位は度)。

X,Y,幅,高さ,奥行きの単位はピクセル

パラメータがたくさんあって合わせるのが難しいので、あらかじめ傾きのない画像を図形に合わせて切り抜いておき、後から回転させることを推奨します。

## 自分でビルドしたい人向け
MSVC++で、ISO C++ 20 標準・OpenMPを有効にするとビルドできるはずです。


