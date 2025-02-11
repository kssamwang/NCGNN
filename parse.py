import argparse

def get_ncgnn_args():
    # PARSER BLOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='pubmed')
    parser.add_argument('--seed', '-S', type=int, default=42)
    parser.add_argument('--hidden', '-H', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--dp1', type=float, default=0.5)
    parser.add_argument('--dp2', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--forcing', type=int, default=0, choices=[0, 1])
    parser.add_argument('--addself', '-A', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model', '-M', type=str, default='NCGNN')
    parser.add_argument('--threshold', '-T', type=float, default=3)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args