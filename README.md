# GCE-GNN

## Code

This is the source code for SIGIR 2020 Paper: _Global Context Enhanced Graph Neural Networks for Session-based Recommendation_.

## Requirements

- Python 3
- PyTorch >= 1.3.0
- tqdm

## Training and testing

~~~~
python build_graph.py --dataset diginetica --sample_num 12
python main.py --dataset diginetica
~~~~

## Citation

~~~~
@inproceedings{wang2020global,
    title={Global Context Enhanced Graph Neural Networks for Session-based Recommendation},
    author={Wang, Ziyang and Wei, Wei and Cong, Gao and Li, Xiao-Li and Mao, Xian-Ling and Qiu, Minghui},
    booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={169--178},
    year={2020}
}
~~~~