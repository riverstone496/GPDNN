# GPDNN

深層学習をガウス過程として解釈したさいの振る舞いを実験するためのソースコードです．

なおこのソースコードは，単純に以下のjuliaによる実装をPythonに移植したものになります．
https://github.com/sammy-suyama/MLBlog/blob/master/src/demo_GPDNN.jl

# Demo

gpdnn.pyを実行すると，各カーネル関数を用いたガウス過程の結果を見ることができます．

参考文献
須山敦志　「ベイズ深層学習」　講談社

J. Lee, Y. Bahri, R. Novak, S. Schoenholz, J. Pennington, and J. SohlDickstein. Deep neural networks as gaussian processes. In Proceedings of the 6th International Conference on Learning Representations, 2018.
https://arxiv.org/abs/1711.00165