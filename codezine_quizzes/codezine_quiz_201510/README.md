##### Swift 2.1 による[20個のカラーボックスで作る本棚は何通り？](http://codezine.jp/article/detail/9005)の解答

実行例: ボックスの数が5個の場合

```sh
% swift -O BoxPatterns.playground/Contents.swift 5
Found
┏━┓┏━┓┏━┓┏━┓┏━┓
┗━┛┗━┛┗━┛┗━┛┗━┛

Found
┏┓┏┓┏┓┏┓┏┓
┃┃┃┃┃┃┃┃┃┃
┗┛┗┛┗┛┗┛┗┛

Found
┏━┓┏━┓
┗━┛┗━┛
┏┓┏┓┏┓
┃┃┃┃┃┃
┗┛┗┛┗┛

Found
┏┓┏┓┏┓
┃┃┃┃┃┃
┗┛┗┛┗┛
┏━┓┏━┓
┗━┛┗━┛

number of patterns: 8
```

縦横を転置した場合を含め計8通り。
