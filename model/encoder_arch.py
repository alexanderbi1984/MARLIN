import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input layer
    to_input('input.tex'),
    
    # Patch Embedding
    to_Conv("patch_embed", 768, 1, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=2, caption="Patch Embedding"),
    
    # Positional Embedding
    to_Conv("pos_embed", 768, 1, offset="(2,0,0)", to="(patch_embed-east)", height=32, depth=32, width=2, caption="Positional Embedding"),
    
    # Transformer Blocks
    to_Conv("block1", 768, 1, offset="(2,0,0)", to="(pos_embed-east)", height=32, depth=32, width=2, caption="Transformer Block 1"),
    to_Conv("block2", 768, 1, offset="(2,0,0)", to="(block1-east)", height=32, depth=32, width=2, caption="Transformer Block 2"),
    to_Conv("block3", 768, 1, offset="(2,0,0)", to="(block2-east)", height=32, depth=32, width=2, caption="Transformer Block 3"),
    to_Conv("block4", 768, 1, offset="(2,0,0)", to="(block3-east)", height=32, depth=32, width=2, caption="Transformer Block 4"),
    to_Conv("block5", 768, 1, offset="(2,0,0)", to="(block4-east)", height=32, depth=32, width=2, caption="Transformer Block 5"),
    to_Conv("block6", 768, 1, offset="(2,0,0)", to="(block5-east)", height=32, depth=32, width=2, caption="Transformer Block 6"),
    to_Conv("block7", 768, 1, offset="(2,0,0)", to="(block6-east)", height=32, depth=32, width=2, caption="Transformer Block 7"),
    to_Conv("block8", 768, 1, offset="(2,0,0)", to="(block7-east)", height=32, depth=32, width=2, caption="Transformer Block 8"),
    to_Conv("block9", 768, 1, offset="(2,0,0)", to="(block8-east)", height=32, depth=32, width=2, caption="Transformer Block 9"),
    to_Conv("block10", 768, 1, offset="(2,0,0)", to="(block9-east)", height=32, depth=32, width=2, caption="Transformer Block 10"),
    to_Conv("block11", 768, 1, offset="(2,0,0)", to="(block10-east)", height=32, depth=32, width=2, caption="Transformer Block 11"),
    to_Conv("block12", 768, 1, offset="(2,0,0)", to="(block11-east)", height=32, depth=32, width=2, caption="Transformer Block 12"),
    
    # Layer Norm
    to_Conv("norm", 768, 1, offset="(2,0,0)", to="(block12-east)", height=32, depth=32, width=2, caption="Layer Norm"),
    
    # Connections
    to_connection("patch_embed", "pos_embed"),
    to_connection("pos_embed", "block1"),
    to_connection("block1", "block2"),
    to_connection("block2", "block3"),
    to_connection("block3", "block4"),
    to_connection("block4", "block5"),
    to_connection("block5", "block6"),
    to_connection("block6", "block7"),
    to_connection("block7", "block8"),
    to_connection("block8", "block9"),
    to_connection("block9", "block10"),
    to_connection("block10", "block11"),
    to_connection("block11", "block12"),
    to_connection("block12", "norm"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main() 